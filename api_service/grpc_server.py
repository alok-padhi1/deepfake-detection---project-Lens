# grpc_server.py
import time
import logging
from concurrent import futures
import grpc

from config import settings
from auth import auth_manager
from proto import api_pb2, api_pb2_grpc

log = logging.getLogger("grpc_server")

class AnalyzeMediaServicer(api_pb2_grpc.AnalyzeMediaServicer):
    def _authenticate(self, context) -> bool:
        metadata = dict(context.invocation_metadata())
        auth_header = metadata.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing or invalid Bearer token in metadata.")
            return False
            
        token = auth_header.split(" ")[1]
        try:
            pub_key = auth_manager.get_public_key(token)
            if not pub_key:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, "Keycloak public key fetch failed.")
                return False
            from jose import jwt
            jwt.decode(
                token,
                pub_key,
                algorithms=["RS256"],
                audience=settings.KEYCLOAK_CLIENT_ID,
                issuer=f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}"
            )
            return True
        except Exception as e:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, f"Token validation failed: {e}")
            return False

    def StreamFrames(self, request_iterator, context):
        log.info("Incoming gRPC StreamFrames call received.")
        if not self._authenticate(context):
            return
            
        for frame in request_iterator:
            frame_num = frame.frame_number
            pts = frame.pts_ms
            raw_bytes = frame.frame_data
            
            # Perform live frame evaluation (e.g. call fast inference models)
            # Simulating fake/anomaly probability output for the streaming demo:
            confidence = float(abs(np.sin(frame_num * 0.1) * 0.4 + 0.1)) if 'np' in globals() else 0.15
            if confidence > 0.5:
                anomaly_flags = ["TEMPORAL_FLICKERING", "FACIAL_WARP"]
            else:
                anomaly_flags = []
                
            yield api_pb2.FrameScore(
                frame_number=frame_num,
                confidence=confidence,
                anomaly_flags=anomaly_flags
            )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    api_pb2_grpc.add_AnalyzeMediaServicer_to_server(AnalyzeMediaServicer(), server)
    listen_addr = f"[::]:{settings.GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    log.info(f"Starting LENS gRPC server on {listen_addr}")
    server.start()
    return server

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = serve()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        s.stop(0)

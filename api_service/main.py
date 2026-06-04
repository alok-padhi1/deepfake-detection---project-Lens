# main.py
import logging
import threading
import time
import uvicorn

from config import settings
import grpc_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
log = logging.getLogger("api_gateway_main")

def start_rest_api():
    log.info(f"Starting REST API Gateway on port {settings.API_PORT}")
    uvicorn.run("app:app", host="0.0.0.0", port=settings.API_PORT, log_level="info")

def start_grpc_server():
    log.info(f"Starting gRPC Server on port {settings.GRPC_PORT}")
    server = grpc_server.serve()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    rest_thread = threading.Thread(target=start_rest_api, name="REST-Thread", daemon=True)
    rest_thread.start()
    
    # Start gRPC in the main thread
    start_grpc_server()

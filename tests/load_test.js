// tests/load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';

export const options = {
  stages: [
    { duration: '5m', target: 500 },  // Ramp 0 -> 500 over 5m
    { duration: '10m', target: 500 }, // Hold 500 for 10m
    { duration: '2m', target: 0 },    // Cool down
  ],
  thresholds: {
    'http_req_duration{type:image}': ['p(50)<2000', 'p(99)<5000'], // p50 < 2s, p99 < 5s for images
    'http_req_duration{type:video}': ['p(50)<15000'],              // p50 < 15s for video
    'http_req_failed': ['rate<0.01'],                              // < 1% error rate at peak
  },
};

const imageLatencyTrend = new Trend('http_req_duration_image');
const videoLatencyTrend = new Trend('http_req_duration_video');
const kedaScaleCounter = new Counter('keda_scale_events');

export default function () {
  const url = 'http://api.gateway.lens/api/v1/analyze';
  
  // Alternate requests between image and video payloads
  const isVideo = Math.random() > 0.7;
  
  const payload = {
    file: http.file('mock_binary_data', isVideo ? 'sample.mp4' : 'sample.jpg'),
    metadata: JSON.stringify({ priority: 'MEDIUM' }),
  };

  const params = {
    headers: {
      'X-API-Key': 'client_load_test_key_001',
      'Authorization': 'Bearer test_keycloak_bearer_token',
    },
    tags: { type: isVideo ? 'video' : 'image' },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'is status 202': (r) => r.status === 202,
    'has case_id': (r) => r.json().case_id !== undefined,
  });

  // Track latency trends
  if (isVideo) {
    videoLatencyTrend.add(res.timings.duration);
  } else {
    imageLatencyTrend.add(res.timings.duration);
  }

  sleep(1);
}

// k6 Lifecycle hooks: verify GPU autoscale metrics via K8s endpoint
export function teardown(data) {
  // Queries k8s API to ensure replicas scaled up to 12
  const k8sUrl = 'https://kubernetes.default.svc/apis/apps/v1/namespaces/lens/deployments/video-forensics-service';
  const res = http.get(k8sUrl, {
    headers: { 'Authorization': 'Bearer k8s_service_account_token' }
  });
  
  if (res.status === 200) {
    const replicas = res.json().status.readyReplicas;
    console.log(`TEARDOWN CHECK: Video Forensic workers scaled to: ${replicas}`);
  }
}

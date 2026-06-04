# LENS Local Development Guide 🚀

This guide walks you through standing up the entire LENS Deepfake Detection Platform directly on your local Windows machine using Docker Compose. All components, from the AI Inference models to the Kafka messaging backbone and Next.js frontend, will spin up automatically.

## Prerequisites

1. **Docker Desktop** installed and running on Windows.
2. At least **16GB of RAM** allocated to Docker (TorchServe + Kafka + Postgres + 5 ML workers require memory).
3. Ensure ports `8000`, `3000`, `9092`, `9000`, `5432`, `6379`, and `8080` are free on your localhost.

---

## Step 1: Start the Entire LENS Stack

Open your terminal (PowerShell or Git Bash) at `d:\isafe2\` and run the following command to boot the entire system in detached mode:

```bash
docker compose --env-file .env.local up -d
```

This will pull all required images and start 13 local containers. The init-script container (`lens-init-scripts`) will automatically format your Kafka topics, MinIO buckets, Postgres schemas, and Keycloak realms.

## Step 2: Verify Service Health

Once the pull is complete, check that all containers are running and healthy:

```bash
docker compose ps
```

You should see `(healthy)` or `Up` next to core infrastructure services like `lens-kafka`, `lens-postgres`, `lens-minio`, and `lens-keycloak`.

*Note: It might take 1–2 minutes for TorchServe and Keycloak to fully boot and pass their health checks.*

## Step 3: Access the LENS UI

Once `lens-ui` is running, open your web browser and navigate to:

👉 **http://localhost:3000**

You will be greeted by the LENS login screen. 

## Step 4: Login to Keycloak

Click "Sign In". You will be redirected to the Keycloak authentication portal. Log in using the pre-provisioned testing credentials:

*   **Username:** `analyst@lens.local`
*   **Password:** `Lens@1234`

*(This user automatically carries the `Analyst` role required to view and submit cases.)*

## Step 5: Run an End-to-End Forensic Trace

1. In the UI Dashboard, click **New Submission** (or navigate to `/submit`).
2. Drag and drop any image (e.g., `.jpg`, `.png`) into the upload zone.
3. Click **Submit**.
4. The file will be streamed to local `MinIO`, and a case ID UUID will be generated.
5. Watch the dashboard—you'll see the status update via Socket.io as it progresses from `INGESTED` → `ANALYZING` → `COMPLETE`.
6. Click into the completed case to view the Bayesian Fusion decision band, confidence score, and PRNU/HFER forensic breakdowns returned by the local CPU TorchServe instances.

---

### Troubleshooting

If a specific worker fails to connect to Kafka or Postgres during the initial boot, simply restart the workers:

```bash
docker compose restart api_service image_forensics_service video_forensics_service audio_forensics_service av_sync_service fusion_service
```

To view logs for the ML fusion pipeline:
```bash
docker compose logs -f fusion_service
```

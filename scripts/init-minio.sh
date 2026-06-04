#!/bin/bash
wget -q -O /usr/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x /usr/bin/mc

echo "Waiting for MinIO to be ready..."
while ! curl -s -f http://minio:9000/minio/health/live; do
  sleep 2
done

echo "Configuring MinIO Client..."
mc alias set myminio http://minio:9000 minioadmin minioadmin

echo "Creating bucket 'lens-media'..."
mc mb myminio/lens-media --ignore-existing

echo "MinIO initialization complete."

#!/bin/bash
# Download Kafka tools if not present in alpine
apk add --no-cache openjdk11-jre wget
wget -q https://downloads.apache.org/kafka/3.5.0/kafka_2.13-3.5.0.tgz
tar -xzf kafka_2.13-3.5.0.tgz
export PATH=$PATH:/scripts/kafka_2.13-3.5.0/bin

BROKER="kafka:9092"

echo "Waiting for Kafka broker to be ready..."
while ! /scripts/kafka_2.13-3.5.0/bin/kafka-topics.sh --bootstrap-server $BROKER --list > /dev/null 2>&1; do
  sleep 2
done
echo "Kafka is ready. Creating topics..."

TOPICS=(
  "media.ingested"
  "analysis.image"
  "analysis.video"
  "analysis.audio"
  "analysis.av_sync"
  "analysis.complete"
  "drift.detected"
  "model.updated"
)

for TOPIC in "${TOPICS[@]}"; do
  /scripts/kafka_2.13-3.5.0/bin/kafka-topics.sh --create --if-not-exists --bootstrap-server $BROKER --topic $TOPIC --partitions 3 --replication-factor 1
  echo "Created topic: $TOPIC"
done

echo "Kafka initialization complete."

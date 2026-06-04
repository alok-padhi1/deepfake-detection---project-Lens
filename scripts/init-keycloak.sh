#!/bin/bash
apk add --no-cache curl jq openjdk11-jre

export PATH=$PATH:/opt/keycloak/bin
KCADM="/opt/keycloak/bin/kcadm.sh"

echo "Downloading Keycloak admin CLI..."
wget -q -O keycloak-22.0.1.tar.gz https://github.com/keycloak/keycloak/releases/download/22.0.1/keycloak-22.0.1.tar.gz
tar -xzf keycloak-22.0.1.tar.gz
KCADM="./keycloak-22.0.1/bin/kcadm.sh"

echo "Waiting for Keycloak to be ready..."
while ! curl -s -f http://keycloak:8080/health/ready; do
  sleep 2
done

echo "Authenticating with Keycloak..."
$KCADM config credentials --server http://keycloak:8080 --realm master --user admin --password admin

echo "Creating LENS realm..."
$KCADM create realms -s realm=lens -s enabled=true

echo "Creating lens-ui client..."
$KCADM create clients -r lens -s clientId=lens-ui -s enabled=true -s publicClient=false -s secret=local_dev_client_secret -s standardFlowEnabled=true -s implicitFlowEnabled=false -s directAccessGrantsEnabled=true -s 'redirectUris=["http://localhost:3000/api/auth/callback/keycloak"]'

echo "Creating Roles..."
$KCADM create roles -r lens -s name=Admin
$KCADM create roles -r lens -s name=Analyst
$KCADM create roles -r lens -s name=Submitter

echo "Creating Analyst user..."
$KCADM create users -r lens -s username=analyst@lens.local -s enabled=true -s email=analyst@lens.local -s firstName=Lens -s lastName=Analyst
$KCADM set-password -r lens --username analyst@lens.local --new-password Lens@1234

echo "Assigning Analyst role..."
$KCADM add-roles -r lens --uusername analyst@lens.local --rolename Analyst

echo "Keycloak initialization complete."

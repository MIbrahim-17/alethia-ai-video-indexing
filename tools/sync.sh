#!/bin/sh
# Push the local repo to ~/alethia on the server (code only; no secrets, data or venv).
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  setsid -w rsync -az --delete \
  --exclude '.env' --exclude '.env.*' --exclude '.git/' --exclude '.venv/' \
  --exclude 'data/' --exclude 'finevideo/' --exclude 'work/' --exclude '__pycache__/' \
  -e "ssh -o ConnectTimeout=15" "$DIR/../" ibrahim@100.112.25.180:alethia/

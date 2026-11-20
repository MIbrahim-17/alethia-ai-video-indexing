#!/bin/sh
# Run a command on the dev server: tools/ssh.sh '<command>'
DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_ASKPASS="$DIR/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  exec setsid -w ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 ibrahim@100.112.25.180 "$@"

#!/bin/sh
# Prints VM_Password from .env for SSH_ASKPASS / sudo -S. Never echo this elsewhere.
sed -n 's/^VM_Password=//p' "$(dirname "$0")/../.env" | tr -d '\r\n'

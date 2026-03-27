#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/run/sshd /root/.ssh /workspace
chmod 700 /root/.ssh

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Avoid a login-password fallback; we only want key-based root access.
if grep -q '^#\?PermitRootLogin' /etc/ssh/sshd_config; then
  sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
else
  printf '\nPermitRootLogin prohibit-password\n' >> /etc/ssh/sshd_config
fi

if grep -q '^#\?PubkeyAuthentication' /etc/ssh/sshd_config; then
  sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
else
  printf '\nPubkeyAuthentication yes\n' >> /etc/ssh/sshd_config
fi

/usr/sbin/sshd
exec sleep infinity

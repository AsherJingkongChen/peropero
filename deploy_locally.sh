#!/bin/bash
set -e

YOUR_GITHUB_USERNAME="AsherJingkongChen"
YOUR_PROJECT_NAME="vast_ai_uv_fastapi_demo"

read -p "Vast.ai Offer ID: " YOUR_VAST_OFFER_ID
if [ -z "${YOUR_VAST_OFFER_ID}" ]; then echo "Offer ID missing" >&2; exit 1; fi
read -p "SSH PubKey Path (optional, e.g. ~/.ssh/id_ed25519.pub): " YOUR_SSH_PUBLIC_KEY_PATH

ONSTART_CMD="\
export PATH=\"\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH\"; \
if [ ! -d \"/opt/${YOUR_PROJECT_NAME}\" ]; then \
    (apt-get update -y && apt-get install -y curl git) > /dev/null 2>&1 && \
    curl -LSsf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 && \
    git clone -q \"https://github.com/${YOUR_GITHUB_USERNAME}/${YOUR_PROJECT_NAME}.git\" \"/opt/${YOUR_PROJECT_NAME}\"; \
fi && \
cd \"/opt/${YOUR_PROJECT_NAME}\" && \
uv run uvicorn server:app --host 0.0.0.0 --port 8888"

echo "Creating instance..."
CREATE_OUTPUT=$(vastai create instance "${YOUR_VAST_OFFER_ID}" \
    --image pytorch/pytorch:latest --disk 16 \
    --onstart-cmd "${ONSTART_CMD}"
)

echo "${CREATE_OUTPUT}"
NEW_INSTANCE_ID=$(echo "${CREATE_OUTPUT}" | sed -E "s|.*'new_contract': ([0-9]+).*|\1|")
if [ -z "${NEW_INSTANCE_ID}" ]; then read -p "Enter Instance ID from output: " NEW_INSTANCE_ID; fi
if [ -z "${NEW_INSTANCE_ID}" ]; then echo "No Instance ID." >&2; exit 1; fi
echo "Instance ID: ${NEW_INSTANCE_ID}"

if [ -n "${YOUR_SSH_PUBLIC_KEY_PATH}" ]; then
    EXPANDED_KEY_PATH="${YOUR_SSH_PUBLIC_KEY_PATH/\~/$HOME}"
    if [ -f "${EXPANDED_KEY_PATH}" ]; then
        echo "Attaching SSH key: ${EXPANDED_KEY_PATH}"
        vastai attach ssh "${NEW_INSTANCE_ID}" "$(cat "${EXPANDED_KEY_PATH}")" >/dev/null 2>&1 || echo "Warn: Failed to attach SSH key." >&2
    else
        echo "Warn: SSH key not found at ${EXPANDED_KEY_PATH}" >&2
    fi
fi

echo "Instance creation initiated. Please wait for several minutes or monitor setup with: vastai logs ${NEW_INSTANCE_ID}"

INSTANCE_DETAILS_JSON=$(vastai show instance "${NEW_INSTANCE_ID}" --raw 2>/dev/null || echo "")
INSTANCE_SSH_HOST=$(echo "${INSTANCE_DETAILS_JSON}" | jq -r '.ssh_host // "[SSH_HOST]"')
INSTANCE_SSH_PORT=$(echo "${INSTANCE_DETAILS_JSON}" | jq -r '.ssh_port // "[SSH_PORT]"')

SSH_CMD_OPTIONS=""
if [ -n "${YOUR_SSH_PUBLIC_KEY_PATH}" ]; then
    POTENTIAL_PRIVATE_KEY_PATH="${YOUR_SSH_PUBLIC_KEY_PATH%.pub}"
    EXPANDED_PRIVATE_KEY_PATH="${POTENTIAL_PRIVATE_KEY_PATH/\~/$HOME}"
    if [ -f "${EXPANDED_PRIVATE_KEY_PATH}" ] && [[ "${YOUR_SSH_PUBLIC_KEY_PATH}" == *.pub ]]; then
        SSH_CMD_OPTIONS=" -i \"${EXPANDED_PRIVATE_KEY_PATH}\""
    fi
fi

echo "---"
echo "SSH Tunnel: ssh${SSH_CMD_OPTIONS} -L 8888:localhost:8888 root@${INSTANCE_SSH_HOST} -p ${INSTANCE_SSH_PORT}"
echo "Test: curl http://localhost:8888/"
echo "Destroy: vastai destroy instance ${NEW_INSTANCE_ID}"
exit 0

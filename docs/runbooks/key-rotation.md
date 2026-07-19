# Runbook — Signing key rotation

## When

- Scheduled (quarterly) or after any suspicion of compromise.
- Rotation is zero-downtime: old signatures still verify against the retired key.

## Prerequisites

- `core.keys.provider: file` in production.
- `core.two_person.min_approvers: 2` in production.
- Write access to the keyring file on disk (`/etc/cryptonaute/keyring.json`).

## Steps

1. Generate a new key (≥ 32 bytes, base64):
   ```bash
   python -c "import os,base64; print(base64.b64encode(os.urandom(32)).decode())"
   ```
2. Edit the keyring: add the new key under a new key_id (e.g. v4), but leave `current`
   pointed at the old key. Save.
3. Verify the file parses:
   ```bash
   python -c "from core.keys import FileKeyProvider; FileKeyProvider('/etc/cryptonaute/keyring.json')"
   ```
4. Promote the new key: set `current` to v4 in the keyring. Save.
5. Run the two-person-approved rotation command:
   ```bash
   cryptonaute --rotate-keys --actor <your-id> --approvers <a1>,<a2> --reason "scheduled"
   ```
6. Verify audit record:
   ```bash
   tail -1 data/audit.jsonl | jq .
   ```
7. Confirm the current key at runtime:
   ```bash
   curl -s localhost:9100/metrics | grep risk_current_key_id
   ```
   (or query `orchestrator.key_manager.current_key_id()` in-process.)

## Rollback

- Revert `current` to the previous key_id in the keyring.
- Run `--rotate-keys` again with the same two-person approval; the audit trail captures both
  rotations.

## Never

- Never delete a retired key from the keyring. Old signatures must remain verifiable.
- Never commit the keyring to source control. It lives on disk, protected by the OS
  (BitLocker / LUKS) or a secrets manager.

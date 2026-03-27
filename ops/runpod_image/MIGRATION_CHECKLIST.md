# Ops Migration Checklist

## Why We Are Doing This

We lost time and money to:
- missing `zstandard`
- `/tmp` data/tokenizer not staged on resumed pods
- partial dataset overlays
- ad hoc bootstrap on expensive pods
- stopping pods before pulling artifacts

This image/template plan is meant to eliminate those.

## Minimal Adoption Plan

1. Build and push our own image.
2. Create two RunPod templates from it:
   - `1xH100-smoke`
   - `8xH100-record`
3. Validate image startup on a cheap pod.
4. Run one `1xH100` smoke launch using our existing repo scripts.
5. Run one `8xH100` parity launch.
6. Only then make it the default substrate for record attempts.

## Compare Against External Template

For each of our image and the community image/template, record:
- pod ready time
- time to pass image validation
- time to pass repo preflight
- time to first training step
- final packaging compressor in logs
- any manual intervention needed

If our image is not clearly at least as reliable, fix that before adopting it.

## Stop Conditions

Do not trust the new image/template yet if:
- `zstandard` is absent
- `/tmp` staging is not verified
- one resumed-pod run fails due to missing tokenizer/data
- logs/artifacts are not easy to recover before stop

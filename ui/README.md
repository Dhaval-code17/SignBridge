# SignBridge UI

Model-independent PWA shell for:

Camera → Model 1 → [T,960] → Model 2 → sign sequence → Model 3 → English → TTS

## Run locally

From this folder:

```powershell
python -m http.server 8000
```

Then open:

http://localhost:8000

## Current behavior

- Camera start/stop/reset
- Camera permission error handling
- Mock Model 2/3 pipeline test
- English text output
- Browser speech synthesis
- Copy translation
- FPS/latency placeholders
- PWA manifest + service worker
- Activity log

## Integration points

Replace the mock path in `app.js` with actual API calls when Person 2 and Person 3 expose their inference endpoints.

Model 1 contract:

`[T,960] float32`

Do not pass the 128-D contrastive projection or classifier logits to Model 2.

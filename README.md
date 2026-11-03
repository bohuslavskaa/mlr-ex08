# Leaf classifier — live camera demo (MLR exercise 8)

**Live page: <https://bohuslavskaa.github.io/mlr-ex08/>**

Open the URL on a phone (or scan the QR code at the bottom of the page) and point
the **rear camera** at a leaf. Every class probability is updated live, and the
model runs entirely in the browser — frames are never uploaded anywhere.

Classes: `apple`, `cherry`, `chestnut`, `maple`.

## What it does

`index.html` is a single self-contained page: plain JavaScript, no build step,
inference by [onnxruntime-web] loaded from the jsDelivr CDN (WebAssembly,
single-threaded because GitHub Pages does not send the cross-origin-isolation
headers that shared-memory threads require).

Preprocessing follows [`model_info.json`](model_info.json) exactly — the file is
fetched at runtime and is the single source of truth for class names and
normalization constants:

| step | detail |
| --- | --- |
| 1. crop | central square of the frame (`side = min(width, height)`) |
| 2. resize | that square to `input_size` × `input_size` = 96 × 96 |
| 3. scale | RGB bytes to 0–1 (`/ 255`) |
| 4. standardize | each channel with `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]` |
| 5. layout | float32, channel-first: `1 × 3 × 96 × 96` |

The exported graph ends at the classifier GEMM (input `image` → output `logits`,
no `Softmax` node), so the page applies a numerically stable softmax itself to
turn logits into the displayed probabilities.

Large camera and photo frames are downscaled in repeated halving steps rather
than one jump, which keeps the 96 × 96 input area-averaged instead of aliased.

## Features

- **Live rear-camera classification** — `getUserMedia({facingMode: 'environment'})`,
  `playsinline` + `muted` so iOS Safari shows the preview inline instead of
  opening the fullscreen player, and camera start behind a tap (a user gesture
  is required on iOS). Live results are lightly smoothed over time; the badge
  shows fps and per-frame inference time.
- **Photo upload as a fallback** — any camera- or permission problem, or just a
  saved picture: `Upload photo` classifies one still, unsmoothed. EXIF
  orientation is honoured via `createImageBitmap(..., {imageOrientation:
  'from-image'})`, with an `<img>` fallback for browsers that reject the option.
- **All four probabilities, always visible** — a bar and a percentage per class,
  plus the top class as a headline.
- **Self-check** — the “Self-check mode” checkbox classifies the bundled
  `check_image.jpg` with the identical pipeline and prints the deviation from
  `check_expected.json` on the page.
- **Switch camera** — appears when the device reports more than one camera.

## Verified

`check_image.jpg` (96 × 96, so the central-square crop and the resize are both
identity) through the deployed page:

| class | page | `check_expected.json` |
| --- | --- | --- |
| apple | 0.829591 | 0.829588 |
| cherry | 0.167949 | 0.167952 |
| chestnut | 0.002456 | 0.002456 |
| maple | 0.0000037 | 0.0000037 |

Max absolute deviation **2.8 · 10⁻⁶** — float32/WebAssembly vs. the reference
run. The same numbers come out of the real “Upload photo” control
(`apple = 83.0%`). The QR code was generated locally and verified by screenshotting
the rendered page and decoding the symbol back to the URL above.

Browsers: the checks above were run in Blink (headless Chrome); the page is
written for **Safari on iPhone** — iOS 16.4 or newer, since that is where
onnxruntime-web's WASM-SIMD build is supported. Camera access needs HTTPS, which
GitHub Pages provides.

## Files

| file | role |
| --- | --- |
| `index.html` | the whole app: markup, styles, preprocessing, inference, UI, QR code |
| `leaf_model.onnx` | MobileNetV3 classifier, 4 classes, 6.3 MB |
| `model_info.json` | class names, input size, normalization constants |
| `check_image.jpg` | 96 × 96 reference image for the self-check |
| `check_expected.json` | expected probabilities for that image |

## Run it locally

Same-origin `fetch()` of the model needs a real HTTP server — opening the file
with `file://` will not work:

```sh
python3 -m http.server 8000
# then open http://localhost:8000/
```

`localhost` counts as a secure context, so the camera works locally too.

## QR code

The QR symbol at the bottom of the page is an inline SVG computed locally
(qrcode-generator, run as a build-time step) and verified by decoding a
screenshot of the rendered page. No third-party QR image service is involved, so
the page makes no request to anyone but jsDelivr.

[onnxruntime-web]: https://www.npmjs.com/package/onnxruntime-web

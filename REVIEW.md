# REVIEW – ex08 leaf classifier page

## 1. It runs on my phone

**What I did.** Opened the GitHub Pages URL on an iPhone in Safari, pressed *Start camera* and pointed the rear camera at four different leaves (a real leaf in my hand and leaf pictures on a screen).

**What I found.** The page shows the live camera image (about 6.7 fps, 40–44 ms per frame) and the probability of all four classes (apple, cherry, chestnut, maple), updated continuously.

![maple](screenshots/phone_maple.png)
Maple leaf held in my hand: the page shows maple 89.7%.

![cherry](screenshots/phone_cherry.png)
Branch with three leaves on a screen: cherry 96.6%.

![chestnut](screenshots/phone_chestnut.png)
Two elongated serrated leaves: chestnut 43.1%, apple 38.8%. The model is unsure here, but the top class is chestnut.

![apple](screenshots/phone_apple.png)
Stock photo of a leaf on a screen: apple 90.2%.

One small flaw: the hint "Point your phone's rear camera at a leaf." stays visible on top of the live video. It does not affect the classification.

**What was fixed.** Nothing was needed for the camera itself.

---

## 2. The same numbers as the notebook

**What I did.** Compared the page with `check_expected.json` using the original 96×96 `check_image.jpg` from the repository folder (2816 bytes, checked with `ls -l`) in three ways: *Upload photo* in Chrome on my Mac, *Upload photo* in Safari on my iPhone, and the self-check mode on the iPhone.

| Class | Expected | Mac Chrome (upload) | iPhone Safari (upload) | iPhone self-check (embedded raw pixels) |
| --- | --- | --- | --- | --- |
| apple | 0.829588 | 0.8296 (page: 83.0%) | 0.8148 (page: 81.5%) | 0.829591 |
| cherry | 0.167952 | 0.1679 (page: 16.8%) | 0.1825 (page: 18.3%) | 0.167949 |
| chestnut | 0.002456 | 0.0025 (page: 0.25%) | 0.0027 (page: 0.27%) | 0.002456 |
| maple | 0.000004 | 0.000004 | 0.000004 | 0.000004 |
| **Max abs. difference** | – | ≈ 3e-6 (agrees) | 1.48e-2 (differs) | 2.77e-6 (PASS, limit 5e-4) |

Logits printed by the page: Mac Chrome `[4.264, 2.667, -1.558, -8.057]`, iPhone Safari `[4.202, 2.706, -1.516, -8.065]`.

**What I found.**

- My first self-check on the iPhone failed with a maximum difference of 1.48e-2 (apple 0.8148 instead of 0.8296), although the predicted class was still apple. Uploading the original file on the iPhone gives the same numbers, while Chrome on the Mac agrees with the reference to three decimals.
- `check_image.jpg` is already 96×96 and the notebook transform has no Resize or CenterCrop, so crop and resize cannot be the cause.
- According to the coding agent's step-by-step test in Chrome and in WebKit (the engine of Safari), only the JPEG decoding differs: Apple's ImageIO changes 3834 of 27648 bytes by up to 5 units, while Chrome decodes bit-identically to the reference. Every canvas step is an identity for a 96×96 image. I did not repeat this analysis myself; I only reproduced the final numbers on the page (83.0% on the Mac, 81.5% on the iPhone).
- An earlier upload gave apple 82.6% in both browsers (logits 4.235, 2.664, -1.540, -8.077). That was not the original file, so I discarded it. I do not know which copy of the image I had used.

**What was fixed.** The self-check now classifies the raw 96×96 RGB bytes of `check_image.jpg`, embedded in the page as base64, so no JPEG decoding is involved. It passes on the iPhone and on the Mac with 2.77e-6, and the limit was tightened to 5e-4. Limitation: this checks the model and the preprocessing, not the browser's JPEG decoder. Uploaded JPEGs on Apple devices remain decoder-limited (about 1.5e-2 here); this is a property of the decoder, not of my preprocessing.

---

## 3. The preprocessing matches my model

**What I did.** Read `index.html` and compared it with `model_info.json` (input_size 96, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) and with the transform in `Copy_of_ex08_student.ipynb` (the cell with `transforms.Compose`, before section "2b. The network").

| Requirement | Where in `index.html` | Lines | Matches |
| --- | --- | --- | --- |
| Central square | `centralSquare`: `side = Math.min(srcW, srcH)`, offsets `(src - side) / 2`; used in `grabCentralSquareRgba` | 204–207, 372 | Yes |
| Resize to 96×96 | `info.input_size`; downscaling loop `while (curSide > size)` in `grabCentralSquareRgba`; pixels read with `getImageData` | 383, 392 | Yes |
| Divide by 255 | `normalizeRgbaToChw` | 217–219 | Yes |
| Standardize with mean/std | same function; values come from `info`, not hard-coded | 214–219 | Yes |
| Channel order R, G, B, shape 1×3×96×96 | `out[i]`, `out[plane + i]`, `out[2 * plane + i]`; `new window.ort.Tensor('float32', data, [1, 3, size, size])` | 217–219, 401 (camera and upload), 734 (self-check) | Yes |
| Same for raw RGB bytes (self-check) | `normalizeRgbToChw`: 3 bytes per pixel without alpha (`j += 3` instead of `j += 4`), same formula and channel order | 226–237 | Yes |

**What I found.** The notebook transform is `ToTensor()` + `Normalize(mean, std)` with the same mean and std as `model_info.json` and no Resize or CenterCrop, because the training images are already 96×96. `ToTensor` divides by 255 and gives channel-first order, which is what the page does. Crop and resize are needed only for camera frames and non-square uploads. The page downscales frames in steps with `drawImage`; there is no reference output for camera frames, so this part cannot be compared numerically. The two normalization functions differ only in the input format (RGBA for camera and upload, RGB for the embedded self-check pixels).

**What was fixed.** Nothing in the preprocessing needed fixing.

---

## 4. Nothing leaves the phone

**What I did.** Searched `index.html` for `fetch`, `XMLHttpRequest`, `sendBeacon`, `WebSocket`, `POST`, `body`, `src=`, `import` and `https://`. Then opened Chrome DevTools → Network, cleared the list, reloaded the page, uploaded a photo and looked at the requests (`screenshots/network.png`).

**What I found.**

- **Code.** There are six `fetch` matches: the helper `fetchWithProgress` and calls that load `leaf_model.onnx`, `model_info.json`, `check_image.jpg` and `check_expected.json` from the page's own site. None of them sets `method` or `body`. The only `body` in the script is `res.body.getReader()`, which reads the downloaded model. There is no `XMLHttpRequest`, `sendBeacon`, `WebSocket`, `POST` or `import`. A photo chosen with *Upload photo* is displayed through `URL.createObjectURL(file)`, a local blob URL.
- **External URLs.** `cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.min.js` (the inference library) and the same folder for its WebAssembly files (`ORT_WASM_PATHS`). They are needed to run the network in the browser. The CDN sees the request (IP address, headers), but never an image. The other `https://` strings are the page's own address (`og:url`, canonical link, QR label and link).
- **Network tab.** Loading the page makes 6 requests: the page, `ort.min.js`, `model_info.json`, `leaf_model.onnx`, `ort-wasm-simd-threaded.jsep.mjs` and the `.wasm` file, all from `github.io` and `cdn.jsdelivr.net`, all GET. After clearing the list and uploading a photo, no new request appeared.
- **QR code.** It is not generated by JavaScript or by an external service. It is static inline SVG in the HTML (inside `<!--QR-BLOCK-START-->`, in `<span id="qrSymbol">`: an `<svg viewBox="0 0 41 41">` with a `<path>`). I checked that the SVG contains no external links.
- **DevTools warnings.** The Issues tab shows two warnings about `SharedArrayBuffer`, coming from the onnxruntime-web file; GitHub Pages cannot set the cross-origin isolation headers. They do not send any data and the page works. No errors.

**What was fixed.** Nothing needed fixing. Optionally, the onnxruntime files could be hosted in the repository, so that the page would not need the CDN at all.
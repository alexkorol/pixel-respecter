# Pixel Perfecter — Web Demo

Drag-and-drop page that runs the **actual Python reconstruction core** in the
browser via [Pyodide](https://pyodide.org) (numpy + OpenCV + Pillow compiled
to WebAssembly). No server-side processing — images never leave the machine.

## Run locally

```bash
# from the repo root (the page fetches ../pixel_perfecter/*.py)
python -m http.server 8748
# open http://localhost:8748/webdemo/
```

## Deploy

Any static host that serves the repo works — e.g. GitHub Pages on the repo
root: Settings → Pages → deploy from branch, then share
`https://<user>.github.io/pixel-perfecter/webdemo/`.

First load fetches ~15 MB of WASM runtime from the jsDelivr CDN (cached
afterwards). A 1024px image reconstructs in ~4–5 s in WASM (vs <1 s native).

## Notes

- The demo loads `pixel_perfecter/{__init__,grid,mesh,colors,reconstructor,palettes}.py`
  straight from the repo, so it is always in sync with the core — there is no
  ported/duplicated algorithm to maintain.
- `window.demoLoad(url)` processes an image by URL and resolves with the
  stats line; used for automated smoke checks.

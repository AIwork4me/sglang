# Experimental YuE2 HTTP route

`http_api.py` registers `POST /v1/audio/generations` on the multimodal
diffusion server when the resolved pipeline is `Yue2Pipeline`.

It is registered by `python/sglang/multimodal_gen/runtime/entrypoints/http_server.py`
(the router lives in `entrypoints/yue2_api.py`), so the route stays out of the
core audio-generation API until the upstream request schema is finalized.

## Routes

Both routes are equivalent:

- `POST /v1/audio/generations`
- `POST /v1/audio/music/generations`

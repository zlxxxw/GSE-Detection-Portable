"""Camera I/O utilities (Windows-friendly OpenCV capture settings).

This project often benchmarks camera FPS with external tools (e.g. vendor apps),
but OpenCV defaults can silently pick a suboptimal backend/pixel format and
produce very low effective FPS. These helpers make the capture settings explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CameraConfig:
    camera_id: int = 0
    width: int = 2560
    height: int = 1440
    fps: int = 60
    backend: str = "dshow"  # dshow | msmf | any
    fourcc: str = "MJPG"  # MJPG is often required for high fps on USB cameras
    buffersize: int = 1


def _backend_to_cv2(backend: str) -> int:
    backend = (backend or "any").lower()
    if backend in ("any", "default"):
        return 0
    if backend in ("dshow", "directshow"):
        return cv2.CAP_DSHOW
    if backend in ("msmf", "mediafoundation"):
        return cv2.CAP_MSMF
    raise ValueError(f"Unsupported backend: {backend}")


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    """Open and configure an OpenCV camera capture.

    Notes (Windows):
      - For many UVC cameras, high resolution/high FPS requires MJPG.
      - OpenCV may default to MSMF; DSHOW is often more stable for UVC.
      - Setting FOURCC before width/height/fps can matter for some drivers.
    """
    cap = cv2.VideoCapture(cfg.camera_id, _backend_to_cv2(cfg.backend))
    if not cap.isOpened():
        return cap

    if cfg.buffersize is not None and cfg.buffersize > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.buffersize))

    if cfg.fourcc:
        fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc.upper())
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.width))
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.height))
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, int(cfg.fps))

    return cap


def get_actual_capture_params(cap: cv2.VideoCapture) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }


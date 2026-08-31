from dataclasses import dataclass

@dataclass
class BoundingBox:
    corners_2D: list(float)
    corners_3D: list(float)

    centers_2D: list(float)
    centers_3D: list(float)
	
    # In this order width, (length), height
    size_3D: list(float)
    size_2D: list(float)

    label: str
    confidence_score: float

    nuscenes_camera_tokens: list(str) | None
    nuscenes_sample_token: str | None

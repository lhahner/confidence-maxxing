from dataclasses import dataclass

@dataclass
class BoundingBox:
    corners_2D: list()
    corner_3D: list()

    centers_2D: list()
    centers_3D: list()

    size_3D: list()
    size_2D: list()

    label: str
    confidence_score: float

    nuscenes_camera_tokens: list() 
    nuscenes_sample_token: str

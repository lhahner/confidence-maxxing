from dataclasses import dataclass, field

@dataclass
class BoundingBox:
    label: str
    confidence_score: float

    nuscenes_camera_token: str 

    corners_2D: list[float] = field(default_factory=list)
    corners_3D: list[float] = field(default_factory=list)

    centers_2D: list[float] = field(default_factory=list)
    centers_3D: list[float] = field(default_factory=list)

    size_2D: list[float] = field(default_factory=list)
    size_3D: list[float] = field(default_factory=list)

    nuscenes_sample_token: str = field(default_factory=str)

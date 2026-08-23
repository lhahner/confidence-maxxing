from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

def detector_box_lidar_to_camera(
    box,
    lidar_to_global,
    cam_pose,
    cam_cs,
):
    T = np.asarray(lidar_to_global)
    R_lidar_global = T[:3, :3]
    t_lidar_global = T[:3, 3]
    box.rotate(Quaternion(matrix=R_lidar_global))
    box.translate(t_lidar_global)
    box.translate(
        -np.asarray(cam_pose["translation"])
    )
    box.rotate(
        Quaternion(cam_pose["rotation"]).inverse
    )
    box.translate(
        -np.asarray(cam_cs["translation"])
    )
    box.rotate(
        Quaternion(cam_cs["rotation"]).inverse
    )
    return box

def detection_to_box(det):
    x, y, z, yaw, length, width, height, score = det["bbox_3d"]

    return Box(
        center=[x, y, z],
        size=[width, length, height],  # nuScenes expects w, l, h
        orientation=Quaternion(
            axis=[0, 0, 1],
            radians=yaw,
        ),
        score=score,
        name=det["label"],
    )


def parse_args():
    pass

def load_3D_detections(path):
    pass

def transform_3D_to_2D():
    x, y, z, yaw, length, width, height, score = det["bbox_3d"]
     Box(
        center=[x, y, z],
        size=[width, length, height],  # nuScenes expects w, l, h
        orientation=Quaternion(
            axis=[0, 0, 1],
            radians=yaw,
        ),
        score=score,
        name=det["label"],
    )


def load_2D_detections(path):
    pass

def compute_center_based_distance():
    pass

def associate(detections_3D_in_2D, detections_2D):
    pass 

def maxx(association_map):
    pass

def main():
    args = parse_args()

    detections_3D = load_3D_detections(path=args.3d_detection_path)
    for detection_3D in detections_3D:
        detection_3D_in_2D = transform_3D_to_2D(detections_3D=detection)
        
        detections_2D = load_2D_detections(path=args.2d_detection_path, sample_token="")
        for detection_2D in detections_2D:
            center_based_distance = compute_center_based_distance(detection_3D_in_2D, detection_2D)
            if center_based_distance > 1.5:
                continue 
            if detection_3D_in_2D["confidence"] < 0.2:
                detection_3D_in_2D["confidence"] += detection_2D["confidence"]

    with open(OUTPUT_FILE, "w") as file:
        io.write(maxxed_detections, file)

if __name__ == "__main__":
    main()

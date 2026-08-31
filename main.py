import argparse
import json
import numpy as np
import math

from data_classes.bounding_box import BoundingBox
from nuscenes.nuscenes import NuScenes
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

DEFAULT_PROJECT_ROOT = Path(
    "/projects/scc/UGOE/UXEI/UMIN/scc_umin_baum/mthesis_lennart_hahner/dir.project"
)
CAMERAS = [
	"CAM_FRONT_LEFT",
	"CAM_FRONT",
	"CAM_FRONT_RIGHT",
	"CAM_BACK_LEFT",
	"CAM_BACK",
	"CAM_BACK_RIGHT",
	]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--nuscenes_root", type=Path, default=f"{DEFAULT_PROJECT_ROOT}/datasets/nuscenes-mini")
    parser.add_argument("--detection_path_3D", 
                        type=Path, 
                        default="./detections/transfusion_opencpdet_nuscenes-mini_simpletrack_format_val.json")
    parser.add_argument("--detection_path_2D", 
                        type=Path, 
                        default="./detections/predictions.json")
    parser.add_argument("--iou-based",
                        type=bool,
                        default=True)
    parser.add_argument("--center-based",
                        type=bool,
                        default=False)
    return parser.parse_args()

def load_detections(path):
    with open(path, "r") as f:
         data = json.load(f)
    return data

def transform_3D_to_2D(detection, 
		       lidar_to_global, 
                       cam_pose, 
                       cam_cs,
                       imsize=(1600, 900), 
		       min_depth=0.1,
		       sample_token="",
		       camera_token=""):
    x, y, z, yaw, length, width, height, score = detection["bbox_3d"]
    box = Box([x, y, z], [width, length, height],
              Quaternion(axis=[0, 0, 1], radians=yaw),
              score=score, name=detection["label"])

    T = np.asarray(lidar_to_global)
    box.rotate(Quaternion(matrix=T[:3, :3]))
    box.translate(T[:3, 3])

    box.translate(-np.asarray(cam_pose["translation"]))
    box.rotate(Quaternion(cam_pose["rotation"]).inverse)
    box.translate(-np.asarray(cam_cs["translation"]))
    box.rotate(Quaternion(cam_cs["rotation"]).inverse)

    corners = box.corners()
    if np.any(corners[2, :] <= min_depth):
        return None

    K = np.asarray(cam_cs["camera_intrinsic"], dtype=float)
    pts = view_points(corners, K, normalize=True)[:2, :]

    cx, cy = view_points(box.center.reshape(3, 1), K, normalize=True)[:2, 0]

    rx1, ry1 = pts[0].min(), pts[1].min()
    rx2, ry2 = pts[0].max(), pts[1].max()

    W, H = imsize
    if rx2 <= 0 or rx1 >= W or ry2 <= 0 or ry1 >= H:
        return None
    x1, x2 = np.clip([rx1, rx2], 0, W - 1)
    y1, y2 = np.clip([ry1, ry2], 0, H - 1)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return BoundingBox(
		corners_2D=[float(x1), float(y1), float(x2), float(y2)],
    	 	corners_3D=corners,
		centers_2D=[float(cx), float(cy)],
		size_3D=[width, length, height],
		size_2D=[x2 - x1, y2 - y1],
		label=detection["label"],
		confidence_score=float(score),
		nuscenes_camera_tokens=camera_token,
		nuscenes_sample_token=sample_token)	

def is_bounding_box_in_camera_frame(detection, nusc, camera_token, intrinsic, cam_sd):
    x, y, z, yaw, length, width, height, score = detection["bbox_3d"]
    box = Box(
        center=[x, y, z],
        size=[width, length, height],
        orientation=Quaternion(axis=[0, 0, 1], radians=yaw)
    )
    box = box_global_to_camera(
            nusc,
            box,
            camera_token
    )
    visible = box_in_image(
            box,
            intrinsic,
            (cam_sd["width"], cam_sd["height"]),
            vis_level=BoxVisibility.ANY
    )
    return visible

# TODO
def mutate_detection():
    pass

def box_global_to_camera(nusc, box, camera_token):
    cam_sd = nusc.get("sample_data", camera_token)

    ego_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        cam_sd["calibrated_sensor_token"]
    )

    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)

    box.translate(-np.array(calibrated_sensor["translation"]))
    box.rotate(Quaternion(calibrated_sensor["rotation"]).inverse)

    return box

def compute_iou_based_distance(transformed_2D_bboxes, native_2D_bboxes):
    matching_boxes = []
    for transformed_2D_bbox in transformed_2D_bboxes:
        if transformed_2D_bbox is None:
            continue
        
        for native_2D_bbox in native_2D_bboxes:
            iou = ops.box_iou(torch.tensor([native_2D_bbox]), torch.tensor([transformed_2D_bbox]))
            if iou >= 0.4:
                matching_boxes.append((transformed_2D_bbox, native_2D_bbox))
    return matching_boxes

def compute_center_based_distance(box_a, box_b):
    matching_boxes = []

    for a_box in box_a:
        if a_box is None:
            continue
        
        for b_box in box_b:
            distance_norm = math.dist(a_box.center, b_box.center) / math.sqrt(a_box.wlh[0]**2 + a_box.wlh[2]**2)
            if distance_norm < 0.25:
                matching_boxes.append(a_box)
    return matching_boxes

def main():
    args = parse_args()
    detections_3D = load_detections(path=args.detection_path_3D)
    nusc = NuScenes(version=str(args.version), dataroot=args.nuscenes_root, verbose=False)

    for detection_3D in detections_3D["frames"]:
        for camera_pose in CAMERAS: 
            sample_token = detection_3D["sample_token"]
            
            cam_token = nusc.get("sample", sample_token)["data"][camera_pose]
            cam_sd = nusc.get("sample_data", cam_token)
            cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"]) 
            cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            intrinsic = np.asarray(cam_cs["camera_intrinsic"])
            
            updated_detections = []
            # Identify all 3D detections in the current camera frame and transform to 2D
            for detection in detection_3D["detections"]: 
                if not is_bounding_box_in_camera_frame(detection, nusc, cam_token, intrinsic, cam_sd):
                    continue
                if detection.score < 0.1:
                    continue
                detection_3D_in_2D = transform_3D_to_2D(detection=detection, 
                                                        lidar_to_global=detection_3D["lidar_to_global"], 
                                                        cam_pose=cam_pose, 
                                                        cam_cs=cam_cs,
                                                        intrinsic=intrinsic)
                # Load all 2D Detections for that camera frame, 
                # TODO loading all detections on every detection again makes no sense
                detections_2D = [] 
                for detection in load_detections(path=args.detection_path_2D):
                    camera_token = detection["image_id"]
                    lidar_token_2D_based = nusc.get("sample", nusc.get("sample_data", 
                                                                       camera_token 
                                                                      )["sample_token"]
                                                    )["data"]["LIDAR_TOP"]
                    lidar_token_3D_based = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
                    if lidar_token_2D_based  == lidar_token_3D_based:
                       box = to_Box(bbox=detection["bbox"], # TODO to_Box not exist
                                    score=detection["score"], 
                                    label=detection["category_id"])
                       detections_2D.append(box)
            
                if len(detections_2D) == 0:
                    continue
                
                # Does the current detection match any of the loaded YOLO detections? 
                matches = associate_detections() # TODO not yet defined
                
                # If the matches are larger then 1, means more than one bounding box represent the same object
                # Ideally this is considered a to be a tuple of the two detections first being TF and snd YOLO
                if len(matches) > 1:
                   matches = apply_nms(matches) # TODO Not Yet defined
                
                # If YOLO is confident here mutate the loaded transfusion detection
                new_detection_score: float
                if matches[1]["score"] > 0.5:
                    new_detection_score = detection["score"] + matches[1]["score"]
                    detection["score"] = new_detection_score
                updated_detections.append(detection)
            detection_3D["detections"] = updated_detections

    with open(OUTPUT_FILE, "w") as file:
        io.write(detections_3D, file)

if __name__ == "__main__":
    main()

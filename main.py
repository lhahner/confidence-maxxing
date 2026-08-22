def parse_args():
    pass

def load_3D_detections(path):
    pass

def transform_3D_to_2D():
    pass

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
    detections_3D_in_2D = transform_3D_to_2D(detections_3D=detections_3D)

    detections_2D = load_2D_detections(path=args.2d_detection_path)

    assoication_map = asscoiate(detections_3D_in_2D, detections_2D)
    maxxed_detections = maxx(association_map==assoication_map)

    with open(OUTPUT_FILE, "w") as file:
        io.write(maxxed_detections, file)

if __name__ == "__main__":
    main()

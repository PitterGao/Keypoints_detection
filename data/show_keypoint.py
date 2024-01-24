import os
import json
import cv2

image_folder = "./images/"
annotation_folder = "./annotations/"

image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.endswith(".png")]

for image_name in image_files:
    image_path = os.path.join(image_folder, image_name + ".png")
    json_path = os.path.join(annotation_folder, image_name + ".json")

    if not os.path.exists(json_path):
        print(f"图像 {image_name} 没有标注文件。")
        continue

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"无法打开标注文件 {json_path}，图像 {image_name} 没有标注。")
        continue

    image = cv2.imread(image_path)
    shapes = data["shapes"]
    keypoints = []
    for shape in shapes:
        if shape["shape_type"] == "point":
            point = shape["points"][0]
            keypoints.append(point)

    for point in keypoints:
        x, y = point
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow("Image with Keypoints", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

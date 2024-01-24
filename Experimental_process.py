import os
import cv2
import json
import random
import argparse
from tools.cam_pth import CAMExtractor_onnx, CAMExtractor_pth
from tools.uilts import display


def split_images_keypoints(image_path, json_path, num):
    """Split the image and map key points"""
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    block_width = width // num
    block_height = height // num

    with open(json_path, 'r') as file:
        data = json.load(file)

    shapes = data["shapes"]
    keypoints = []
    for shape in shapes:
        if shape["shape_type"] == "point":
            point = shape["points"][0]
            keypoints.append(point)

    segmented_images = []
    segmented_keypoints = []

    for row in range(num):
        for col in range(num):
            start_x = col * block_width
            end_x = start_x + block_width
            start_y = row * block_height
            end_y = start_y + block_height

            block = image[start_y:end_y, start_x:end_x]

            adjusted_keypoints = []
            for point in keypoints:
                x, y = point
                if start_x <= x < end_x and start_y <= y < end_y:
                    adjusted_x = x - start_x
                    adjusted_y = y - start_y
                    adjusted_keypoints.append((int(adjusted_x), int(adjusted_y)))

            # for point in adjusted_keypoints:
            #     x, y = point
            #     cv2.circle(block, (int(x), int(y)), 5, (0, 255, 0), -1)

            segmented_images.append(block)
            segmented_keypoints.append(adjusted_keypoints)

    return segmented_images, segmented_keypoints


def keypoint_detection_rate(weights, split_images, key_points, detection_num, size):
    """The key point detection probability of the image is calculated"""
    # CAM = CAMExtractor_pth(model_weights=weights, en_glcm=True)
    CAM = CAMExtractor_onnx(model_weights=weights, en_glcm=True)

    indices = random.sample(range(len(split_images)), detection_num)
    selected_images = [split_images[i] for i in indices]
    selected_keypoints = [key_points[i] for i in indices]

    num_keypoints = 0
    num_detected_keypoints = 0

    for i, image in enumerate(selected_images):
        keypoints = selected_keypoints[i]

        # 生成热力图
        input_img = cv2.resize(image, size)
        heatmap = CAM(input_img)

        height, width, channels = image.shape
        heatmap = cv2.resize(heatmap, (width, height))

        # cv2.imshow("heat", heatmap)
        # cv2.imshow("ori_img", image)
        # cv2.waitKey(0)

        for point in keypoints:
            x, y = point
            if heatmap[y, x] >= 0.5:
                num_detected_keypoints += 1
            num_keypoints += 1

    # 计算关键点概率
    detection_rate = num_detected_keypoints / num_keypoints if num_keypoints != 0 else 0

    return detection_rate


def main(opt):
    image_files = [os.path.splitext(file)[0] for file in os.listdir(opt.img_dir) if file.endswith(".png")]

    num = 0
    rate_all = 0.0
    for image_name in image_files:
        image_path = os.path.join(opt.img_dir, image_name + ".png")
        json_path = os.path.join(opt.json_dir, image_name + ".json")

        if not os.path.exists(json_path):
            print(f"Warning： {image_name} Key points are not marked")
            print("------------------------------------------------------")

            continue

        try:
            with open(json_path, "r") as f:
                segmented_images, segmented_keypoints = split_images_keypoints(image_path, json_path, opt.split_num)
        except FileNotFoundError:
            print(f"无法打开标注文件 {json_path}，图像 {image_name} 没有标注。")
            continue

        # Map key points to the artwork and display
        # display(segmented_images, segmented_keypoints)

        if opt.choose_model is True:
            rate = keypoint_detection_rate(opt.weights_onnx, segmented_images, segmented_keypoints,
                                           opt.detection_num, opt.SIZE)
        else:
            rate = keypoint_detection_rate(opt.weights_pth, segmented_images, segmented_keypoints,
                                           opt.detection_num, opt.SIZE)

        print(f"Picture-{image_name} rate:{rate * 100}%")
        print("------------------------------------------------------")

        num += 1
        rate_all += rate

    print(f"total:{rate_all/num * 100}%")
    print("------------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num', default=2, help='The number of times the artwork is averaged')
    parser.add_argument('--detection_num', default=2, help='Random extraction of a specified number of images')
    parser.add_argument('--SIZE', default=(512, 512), help='Enter the image size into the model to generate CAM')
    parser.add_argument('--img_dir', default='./data/images/', help='Image folder')
    parser.add_argument('--json_dir', default='./data/annotations/', help='Key points json folder')
    parser.add_argument('--choose_model', default=True, help='choose resnet101.pth or camnet.onnx')
    parser.add_argument('--weights_pth', default='./weights/resnet101.pth', help='pth')
    parser.add_argument('--weights_onnx', default='./weights/camnet.onnx', help='onnx')
    args = parser.parse_args()
    main(args)

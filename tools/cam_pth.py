import torch
import cv2
from Keypoint_detection.tools import fast_glcm
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
import onnxruntime as ort
import numpy as np


class CAMExtractor_pth:
    def __init__(self, model_weights, en_glcm, size=(512, 512)):
        self.model = torch.load(model_weights).to('cuda')
        self.model.eval()
        self.en_glcm = en_glcm
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        input_tensor = preprocess_image(img)
        tensor = input_tensor.to('cuda')
        norm_img = img / 255.0

        target_layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        CAMExtractor = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)

        cam = CAMExtractor(input_tensor=tensor, targets=None, aug_smooth=True)
        cam = 1 - cam
        cam = cam.squeeze()

        cam_show = cv2.applyColorMap(np.uint8(255 * (1 - cam)), cv2.COLORMAP_JET)

        if self.en_glcm:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_glcm = fast_glcm.fast_glcm_mean(gray, 0, 255, 8, 5)
            weight = mean_glcm / mean_glcm.max()
            weight = 1 - weight

            cam = np.where(cam > 0.5, 1, cam)
            weight = np.where(weight > 0.5, 1, weight)
            grayscale_cam = 0.5 * weight + 0.5 * cam
            grayscale_cam = np.where(grayscale_cam > 0.5, 1, grayscale_cam)
            visualization = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)

            # cv2.imshow("visualization", visualization)
            # cv2.imshow("cam", cam_show)
            # cv2.waitKey(0)

            return 1 - grayscale_cam

        # cam = np.where(cam > 0.5, 1, cam)
        visualization = show_cam_on_image(norm_img, cam, use_rgb=True)
        return 1 - cam


class CAMExtractor_onnx:
    def __init__(self, model_weights, en_glcm, size=(512, 512)):
        self.model = ort.InferenceSession(model_weights, providers=['CUDAExecutionProvider'])
        self.en_glcm = en_glcm
        self.size = size

    def __call__(self, img):
        img = cv2.resize(img, self.size)
        input_tensor = preprocess_image(img)
        tensor = input_tensor.numpy()
        norm_img = img / 255.0

        cam = self.model.run(None, {"input": tensor})[0]
        cam = 1 - cam
        cam = cam.squeeze()

        cam_show = cv2.applyColorMap(np.uint8(255 * (1 - cam)), cv2.COLORMAP_JET)

        if self.en_glcm:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_glcm = fast_glcm.fast_glcm_mean(gray, 0, 255, 8, 5)
            weight = mean_glcm / mean_glcm.max()
            weight = 1 - weight

            cam = np.where(cam > 0.5, 1, cam)
            weight = np.where(weight > 0.5, 1, weight)
            grayscale_cam = 0.5 * weight + 0.5 * cam
            grayscale_cam = np.where(grayscale_cam > 0.5, 1, grayscale_cam)
            visualization = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)

            # cv2.imshow("visualization", visualization)
            # cv2.imshow("cam", cam_show)
            # cv2.waitKey(0)

            return 1 - grayscale_cam

        # cam = np.where(cam > 0.5, 1, cam)
        visualization = show_cam_on_image(norm_img, cam, use_rgb=True)
        return 1 - cam


if __name__ == '__main__':
    image = cv2.imread("../data/images/Micritic_limestone.2.png")
    CAM = CAMExtractor_pth(model_weights="../weights/resnet101.pth", en_glcm=True)
    # CAM = CAMExtractor_onnx(model_weights="../weights/camnet.onnx", en_glcm=True)
    cam = CAM(image)
    cv2.imshow("cam", cam)
    cv2.waitKey(0)

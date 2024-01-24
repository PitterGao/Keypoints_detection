import cv2
import copy

def display(segmented_images, segmented_keypoints):
    for i, image in enumerate(segmented_images):
        cloned_image = copy.deepcopy(image)
        keypoints = segmented_keypoints[i]

        for point in keypoints:
            x, y = point
            cv2.circle(cloned_image, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.imshow(f"Segmented Image {i}", cloned_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
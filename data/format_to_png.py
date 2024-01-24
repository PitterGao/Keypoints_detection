import os
from PIL import Image

folder_path = "images"
image_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)

    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        target_path = os.path.splitext(image_path)[0] + ".png"
        image.save(target_path, "PNG")

        print(f"文件 {image_file} 转换成功。")

    except Exception as e:
        print(f"文件 {image_file} 转换失败: {str(e)}")

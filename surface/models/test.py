from PIL import Image

image_path = '/home/tuslam/surface_normal_server/surface/models/test_baseline/2011_09_26_drive_0001_sync_02/0000000001.jpg'
try:
    image = Image.open(image_path)
    image.show()  # 这将显示图片
    print(f'成功读取图片，尺寸为: {image.size}')
except IOError as e:
    print(f'读取图片时出错: {e}')
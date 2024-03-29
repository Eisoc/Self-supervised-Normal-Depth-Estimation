import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from models.MotionFusionNet import MotionFusionNet


image_path = 'data/imgs/val/image/13_1.png'
flow_path = 'data/imgs/val/flow/13_1.png'
left_image_path = 'data/imgs/val/left/13_1.png'

model = MotionFusionNet()
model.load_state_dict(torch.load('checkpoints/checkpoints/best.pt'))
model = model.to('cuda')
model.eval()

img = Image.open(image_path)
flow = Image.open(flow_path)

trans = transforms.ToTensor()
img = trans(img).to('cuda').unsqueeze(0)
flow = trans(flow).to('cuda').unsqueeze(0)

pred = model(img, flow).to('cpu').squeeze(0)
pred = torch.argmax(pred, dim=0)

color = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 0)]).astype(np.uint8)
img_label = color[pred]
img_label = Image.fromarray(np.uint8(img_label))

img_row = Image.open(left_image_path)
img = Image.blend(img_row, img_label, 0.3)
img.save('result.png')
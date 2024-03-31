import math

import torch
from torch.utils.data import DataLoader
import torchmetrics as tm

from models.MotionFusionNet import MotionFusionNet
from data.dataset.kittimotion import KITTIMotion

cm = tm.ConfusionMatrix(task='multiclass', num_classes=3).to('cuda')

def calculate_iou(outputs, targets):
    confusion_matrix = cm(outputs, targets)

    intersection = confusion_matrix.diag()
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection

    iou = intersection / union

    miou = torch.mean(iou)

    return miou.item()

model = MotionFusionNet()
model.load_state_dict(torch.load('checkpoints/checkpoints/best.pt'))
model = model.to('cuda')
model.eval()

dataset = KITTIMotion('data/imgs', train=False)
val_iter = DataLoader(dataset, 4, True)

count = 0
mIoU = 0
with torch.no_grad():
    for image, flow, label in val_iter:
        image = image.to('cuda')
        flow = flow.to('cuda')
        label = label.to('cuda')
        pred = model(image, flow)
        tmp = calculate_iou(pred, label)
        if not math.isnan(tmp):
            mIoU += tmp
        count += 1
print(f'mIoU:{mIoU / count}')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from model.MotionFusionNet import MotionFusionNet
from data.dataset.kittimotion import KITTIMotion

torch.cuda.set_device(0)
device = 'cuda'
epochs = 800

model = MotionFusionNet()
model = model.to(device)
model.train()

train_iter = DataLoader(KITTIMotion('data', True), batch_size=16, shuffle=True)
loss_func = CrossEntropyLoss()
optimizer = Adam([{'params': model.parameters(), 'initial_lr': 0.01}], lr=0.01)
lr_scheduler = StepLR(optimizer, 3000, 0.9, 400)
step = 1
for epoch in range(epochs):
    for idx, (image, flow, label) in enumerate(train_iter):
        image = image.to(device)
        flow = flow.to(device)
        label = label.to(device)
        pred_label = model(image, flow)
        loss = loss_func(pred_label, label)
        loss.backward()
        print(f'epoch: {str(epoch + 1)}, epoch_step:{str(idx + 1)}, global_step: {step}, loss: {loss.item()}')
        step += 1
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
    torch.save(model.state_dict(), f'epoch-{str(epoch + 1)}.pt')

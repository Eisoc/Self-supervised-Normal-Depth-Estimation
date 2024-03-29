import glob
from PIL import Image
import numpy as np
import os
from functools import cmp_to_key

def cmp(x: str, y: str):
    x = int(os.path.split(x)[1].split('.')[0])
    y = int(os.path.split(y)[1].split('.')[0])

    return x - y

train_labels = glob.glob('processed_data/val/label/*.png')
train_lefts = glob.glob('processed_data/val/left/0/*.png')
train_rights = glob.glob('processed_data/val/right/0/*.png')

train_labels = sorted(train_labels, key=cmp_to_key(cmp))[0:44]
train_lefts = sorted(train_lefts, key=cmp_to_key(cmp))[0:44]
train_rights = sorted(train_rights, key=cmp_to_key(cmp))[0:44]

for index, path in enumerate(train_labels):
    img = Image.open(path)
    img = img.resize((1280, 384))
    img = np.array(img)
    img1, img2, img3 = img[:, 0:768], img[:, 256:1024], img[:, 512:1280]
    img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')
    img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
    img3 = Image.fromarray(img3.astype('uint8')).convert('RGB')
    # os.remove(path)
    (path, _) = os.path.split(path)
    img1.save(os.path.join('val/label', f'{str(index + 1)}_1.png'))
    img2.save(os.path.join('val/label', f'{str(index + 1)}_2.png'))
    img3.save(os.path.join('val/label', f'{str(index + 1)}_3.png'))

for index, path in enumerate(train_lefts):
    img = Image.open(path)
    img = img.resize((1280, 384))
    img = np.array(img)
    img1, img2, img3 = img[:, 0:768, :], img[:, 256:1024, :], img[:, 512:1280, :]
    img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')
    img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
    img3 = Image.fromarray(img3.astype('uint8')).convert('RGB')
    # os.remove(path)
    (path, _) = os.path.split(path)
    img1.save(os.path.join('val/left', f'{str(index + 1)}_1.png'))
    img2.save(os.path.join('val/left', f'{str(index + 1)}_2.png'))
    img3.save(os.path.join('val/left', f'{str(index + 1)}_3.png'))

for index, path in enumerate(train_rights):
    img = Image.open(path)
    img = img.resize((1280, 384))
    img = np.array(img)
    img1, img2, img3 = img[:, 0:768, :], img[:, 256:1024, :], img[:, 512:1280, :]
    img1 = Image.fromarray(img1.astype('uint8')).convert('RGB')
    img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
    img3 = Image.fromarray(img3.astype('uint8')).convert('RGB')
    # os.remove(path)
    (path, _) = os.path.split(path)
    img1.save(os.path.join('val/right', f'{str(index + 1)}_1.png'))
    img2.save(os.path.join('val/right', f'{str(index + 1)}_2.png'))
    img3.save(os.path.join('val/right', f'{str(index + 1)}_3.png'))
#####################################################
# resnet50_image.py
# training and validating ResNet50 on
# the BEE4 image dataset.
# bugs to vladimir kulyukin, chris allred on canvas
#####################################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ]),
}

image_datasets = {
'train':
    datasets.ImageFolder('data/BEE4/train', data_transforms['train']),
    'validation':
    datasets.ImageFolder('data/BEE4/valid', data_transforms['validation'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size=32,
                                shuffle=True, num_workers=4),
    'validation':
    torch.utils.data.DataLoader(image_datasets['validation'],
                                batch_size=32,
                                shuffle=False, num_workers=4)
}

from sys import platform
device = 'cuda:0' if torch.cuda.is_available() else ('mps' if platform == "darwin" else 'cpu')

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)).to(device)

lossfun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

def train_model(model, lossfun, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = lossfun(outputs, labels)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))
    return model

model_trained = train_model(model, lossfun, optimizer, num_epochs=3)
torch.save(model_trained.state_dict(), 'resnet50_bee4.h5')

model = models.resnet50(pretrained=False).to(device)
model.fc = nn.Sequential(nn.Linear(2048, 128),
                         nn.ReLU(inplace=True),
                         nn.Linear(128, 2)).to(device)
model.load_state_dict(torch.load('resnet50_bee4.h5'))

validation_img_paths = ['data/BEE4/valid/bee/32_1180_yb.png',
                        'data/BEE4/valid/bee/80_46_yb.png',
                        'data/BEE4/valid/nobee/192_168_4_5-2017-05-13_16-38-06_27_223_18.png',
                        'data/BEE4/valid/nobee/192_168_4_8-2017-05-08_15-45-28_187_261_195.png']
img_list = [Image.open(img_path) for img_path in validation_img_paths]

validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                for img in img_list])

pred_logits_tensor = model(validation_batch)
pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

for i, img in enumerate(img_list):
    title_str = f'{100*pred_probs[i,0]:.0f}% Bee, {100*pred_probs[i,1]:.0f}%␣noBee'
    print(title_str)

samples_size=5
# randomly sample validation images and display the network performance on each image.
for _ in range(2):
    rand_idx = np.random.randint(0, len(image_datasets['validation']), size=samples_size)
    for i, idx in enumerate(rand_idx):
        # get the image without transform
        untransformed, _ = image_datasets['validation'][idx]
        sample_image, class_label = image_datasets['validation'][idx]
        with torch.no_grad():
            outputs = model(sample_image.unsqueeze(0).to(device))
            _, preds = torch.max(outputs, 1)
            pred_probs = F.softmax(outputs, dim=1).cpu().data.numpy().squeeze()
        performance_str = f'{image_datasets["validation"].classes[preds]}: {pred_probs[preds]*100:.2f}%\ntrue: {image_datasets["validation"].classes[class_label]}'
        print('------\n{}\n'.format(performance_str))
        
print('Done...')

# Author: Armin Masoumian (masoumian.armin@gmail.com)

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from keras.datasets import cifar10
import numpy as np
from torch.utils.data import Dataset, DataLoader
from resnet import *
from mri_dataset import *
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import datetime
from densenet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

model = ResNet50()
model = model.cuda()

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = MRI_DATASET(
    train=True,
    returns="all",
    intersect_idx=None,
)

valid_dataset = MRI_DATASET(
    train=False,
    returns="all",
    intersect_idx=None,  # TODO: support validation intersect indices
)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=3)
valid_loader = DataLoader(valid_dataset, batch_size=3, shuffle=False, num_workers=10)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-05)

print("Hello")
print(len(train_dataset))

train_size = len(train_dataset)
start_time = time.time()
batch_size = 3
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #2d daten 3d machen
        #inputs = inputs.unsqueeze(2)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs.squeeze(), labels.float().squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, epoch + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')
end_time = time.time()
elapsed_time = end_time - start_time

correct = 0
total = 0
dog_negetive = 0
cat_positive = 0
dog_positive = 0
cat_negative = 0
predict = []
correct2 = []
valid_indices = []

all_outputs = []
all_labels = []

with torch.no_grad():
    for data in valid_loader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        #inputs = inputs.unsqueeze(2)

        outputs = model(inputs)
        print(outputs.size())
        outputs = torch.sigmoid(outputs)

        all_outputs.append(outputs.cpu())
        all_labels.append(labels.cpu())

        total += labels.size(0)
        predict.append(outputs.tolist())
        correct2.append(labels.tolist())

all_outputs = torch.cat(all_outputs).numpy()
all_labels = torch.cat(all_labels).numpy()

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"dense_50w3_output_{current_time}.txt"

fpr, tpr, _ = roc_curve(all_labels, all_outputs)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# Plot speichern mit Uhrzeit im Dateinamen
plot_filename = f"dense_50w3_auroc_plot_{current_time}.png"
plt.savefig(plot_filename)

with open(filename, 'w') as f:
    # Schreiben der Variablenwerte in die Datei
    f.write(f"Time: {elapsed_time}\n")
    f.write(f"Prediction: {predict}\n")
    f.write(f"Targer: {correct2}\n")

print(total)


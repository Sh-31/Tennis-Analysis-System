import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import cv2

class KeypointsDataset(Dataset):
    def __init__(self, img_dir:str, data_dir:str):
        self.img_dir = img_dir

        with open(data_dir, "r") as f:
            self.data = json.load(f)

        self.transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize( # Normalizes the tensor using the mean and standard deviation values (these are the standard normalization values for ImageNet).
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):    
        item = self.data[idx]
        img = cv2.imread(f"{self.img_dir}/{item['id']}.png")
        heigth , width , _ = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)

        # when resizing image ( transforms.Resize((224,224)) )
        # the x , y points of keypoints also 
        # need to adjust

        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)
        kps[::2]  *= 224.0 / width  # Adjust x coordinates
        kps[1::2] *= 224.0 / heigth # Adjust y coordinates

        return img , kps

def train_one_epoch(epoch_index):
    running_loss = []

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move to device

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) 
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    return running_loss


if __name__ == "__main__":

    # install keypoints datasets commands and unzip
    # !wget --header="Host: drive.usercontent.google.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header="Accept-Language: en-US,en;q=0.9,ar;q=0.8" --header="Cookie: HSID=Ag2OIHvsd2Wub4C7z; SSID=AWnBcQKwDHiTrZAU1; APISID=pltrFZgE9lJ0o1gq/AN9feEHYvs8oHd519; SAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; __Secure-1PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; __Secure-3PAPISID=zgF45F21ZPWzYWZw/AgUMJ8b7QQXuWGn19; SID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-uttBbVDRolhF-hY16nwHXw0gACgYKAWISAQASFQHGX2MivNTw_E_toJuIRy6LMpKNOBoVAUF8yKpFSmvq7AMjvEWeNc50Zff40076; __Secure-1PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-utbSY2jBY1VXuw8gYl5hIO2QACgYKAXsSAQASFQHGX2MihVCJ1PwLozGqZgdSatM9QhoVAUF8yKpgrsTvI8i_UE-YHpoN7Gx-0076; __Secure-3PSID=g.a000fwgYx1PcnW-rFyFhg3x6mQHzCrwXz-KFhoOLogUl7YTWI-utwVfPl2imdPimZJ9tdDZGQAACgYKAUESAQASFQHGX2MiEJ49mV4jME2kttDAV5hwWBoVAUF8yKp80mIgju1lu-q4nI7VsFDM0076; NID=511=efI9IZpxtyJ7Dw1MAUXU8FlzS5jXGewY4Er8HliWc3A0RSWdgvNDyKY66ETjgRyTGWPbWODSmiSeYSBab5SPHVwqbJxd6ZeGW2f6BkHi61UKksXPH0CVJRM1hKpMjHPU5qw7tboM2Mi87NrosV8COB-GCLulLLbjOoSAEQewTe8NVZ5Owq8IkwvxFGfJkmUKEMkFWrw9yb5nTDl3wbZEsGFI92iEdNTSxSRovNCIPN2US-SCFdQ0m2BtvwdiWZbgnn7dSQ8yPA145Kk2BA-ATpJNJ6SJHEHLQY-9CPail9D5qgJgxR925EUg5RGCpEu9wS5xbA62KTa19wAvbAq7Dk3TWc-iX4p1s7ESFyDC7yMpFxiFPJjqkWwFi_ZfiK2TW2t0TQ60DFBxqOytQaLyHrkEvD-CQPVj6OCOP22cZY0Cu61HaAQgFO9pXH-kJUlywzVdbirJumN5gswyaQ49b3KdLcG0Jb7brOMTM24T2nGtQ10hJzsnTwX7dBk3ujqQrI_DGuURvPassPUrIZ0; AEC=Ae3NU9MOEGeKAZjP6INpOYbyMraWAWztmx5pJB_1ILu1furiTy1K37k15u0; __Secure-1PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; __Secure-3PSIDTS=sidts-CjEBYfD7Z9twEKTWJ9gU7KG-rLbxJGNRQIoG3wH6JVu6yiCC2fsRrm7tN8L6d5WlILrnEAA; 1P_JAR=2024-02-18-08; SIDCC=ABTWhQExCxkfmwCkG1RaEgz8U1ZkPeh3HmLMUdMt8S5cNSsLY5U5rAL6wlvq7dtjRw7zrtAbqsFI; __Secure-1PSIDCC=ABTWhQH0jLeRIS6Tu3LS8DXB5Q3gGDq9LTmlk60FKu795Bf0UbzsOcYWVAE96clq5aAL8i724Q0; __Secure-3PSIDCC=ABTWhQHIFcyv3nZYwp78WXEQal71jCE_ZsGT5lXs8VLr7XDIfFqHcLTIPz4HxzJb9ZnYQ5l2s9eU" --header="Connection: keep-alive" "https://drive.usercontent.google.com/download?id=1lhAaeQCmk2y440PmagA0KmIVBIysVMwu&export=download&authuser=0&confirm=t&uuid=3077628e-fc9b-4ef2-8cde-b291040afb30&at=APZUnTU9lSikCSe3NqbxV5MVad5T%3A1708243355040" -c -O 'tennis_court_det_dataset.zip'
    # unzip tennis_court_det_dataset.zip 

    root_path = "/teamspace/studios/this_studio/Tennis-Analysis-system/fine_tuning/"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = KeypointsDataset(root_path + "data/Keypoints_dataset/images", root_path + "data/Keypoints_dataset/data_train.json")
    val_dataset = KeypointsDataset(root_path + "data/Keypoints_dataset/images",   root_path + "data/Keypoints_dataset/data_val.json")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc =  torch.nn.Linear(model.fc.in_features, 14*2)
    
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

    epoch_number = 0
    EPOCHS = 25
    best_vloss = 1_000_000.
    BATCH_SIZE = 128 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        running_loss = train_one_epoch(epoch_number)
        avg_loss = np.mean(running_loss) # loss per batch
        writer.add_scalar('Loss/train', avg_loss, epoch_number + 1)

        running_vloss = 0.0
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)  
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels) 
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                          {'Training' : avg_loss, 
                           'Validation' : avg_vloss},
                            epoch_number + 1
                          )
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}.pth'.format(timestamp, epoch_number + 1)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    model_save_name =  'model_last.pth'   
    torch.save(model.state_dict(), model_save_name)            
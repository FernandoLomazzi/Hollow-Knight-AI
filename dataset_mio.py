from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class LazyVideoDataset(Dataset):
    def __init__(self, video, df, actions):
        # Asume left 0 right 2
        self.video = video
        self.labels = torch.tensor(df[actions].astype(float).values)

        self.preprocess = transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return self.labels.shape[0] - 2

    def __getitem__(self, idx):
        #print(idx)
        # 0 1 2 3 4 5 6 7 8 9
        # 0     1     2     3
        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        pattern = torch.empty((3, 3, 224, 224))
        #print(idx)
        for i in range(3):
            ret, frame = self.video.read()
            assert(ret)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame = Image.fromarray(frame)
            #plt.imshow(frame)
            #plt.show()
            frame = self.preprocess(frame)
            pattern[i] = frame
        return (pattern, self.labels[idx+2])


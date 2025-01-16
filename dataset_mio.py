from torch.utils.data import Dataset
import torch
from torchvision import transforms
import cv2

class LazyVideoDataset(Dataset):
    def __init__(self, video, df, actions, w = 3):
        self.video = video
        self.labels = torch.tensor(df[actions].astype(float).values)
        self.w = w
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Use this methods for training dataset
    def __len__(self):
        return self.labels.shape[0] - self.w
    
    def __getitem__(self, idx):
        # 0 1 2 3 4 5 6 7 8 9
        # 0     1     2     3
        self.video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        pattern = torch.empty((self.w, 3, 224, 224))
        for i in range(self.w):
            ret, frame = self.video.read()
            assert(ret)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.preprocess(frame)
            pattern[i] = frame
        return (pattern, self.labels[idx+self.w])

    """
    # Use this methods for fine tuning dataset

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        # 0 1 2 3 4 5 6 7 8 9
        # 0     1     2     3
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.w*idx)
        pattern = torch.empty((self.w, 3, 224, 224))
        for i in range(self.w):
            ret, frame = self.video.read()
            assert(ret)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.preprocess(frame)
            pattern[i] = frame
        return (pattern, self.labels[idx])
    """
    

    

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import pandas as pd
import numpy as np
import cv2
from torch.nn.functional import pad

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset  # Import Dataset class
from tqdm.auto import tqdm

class VideoDataset(Dataset):
    def __init__(self, video_folder, label_file, transform=None):
        self.video_folder = video_folder
        self.labels = pd.read_csv(label_file)
        self.transform = transform

        # Map activities to numerical labels
        # self.activity_to_label = {activity: i for i, activity in enumerate(self.labels['activity'].unique())}
        self.labels['label'] = self.labels['activity'].map(activity_to_label)

        # Calculate the max sequence length
        self.max_length = self.labels.apply(
            lambda row: row['frame_end'] - row['frame_start'] + 1, axis=1
        ).max()

    def __len__(self):
        return len(self.labels)

    def pad_frames(self, frames, max_length):
        current_length = frames.size(0)
        if current_length < max_length:
            padding = (0, 0, 0, 0, 0, 0, 0, max_length - current_length)
            frames = pad(frames, padding, mode='constant', value=0)
        return frames

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        frame_start, frame_end = int(row['frame_start']), int(row['frame_end'])
        video_name = f"{row['file_id']}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Warning: Cannot open video: {video_path}. Skipping this sample.")
            return None  # Skip this sample

        # Process the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Failed to open video: {video_path}")
            return None

        frames = []
        for i in range(frame_end + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {i} from video {video_name}")
                break
            if frame_start <= i <= frame_end:
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No valid frames found in range {frame_start}-{frame_end} for {video_name}")

        frames = np.array(frames)
        frames = torch.tensor(frames, dtype=torch.float32).unsqueeze(1)  # Convert to float and add channel dimension
        frames = self.pad_frames(frames, self.max_length)  # Pad frames

        mask = torch.zeros(self.max_length, dtype=torch.float32)
        mask[:len(frames)] = 1

        return {
            'frames': frames,
            'mask': mask,
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

# Initialize MobileNetV2 1.0x with Jester pre-trained weights
class MobileNetV2_3D_Jester(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV2_3D_Jester, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU6(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU6(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        batch_size, T, C, H, W = x.size()
        x = x.view(batch_size, C, T, H, W)
        features = self.features(x)  # Extract intermediate features
        flattened = features.view(batch_size, -1)  # Flatten the feature map
        if return_features:
            return flattened  # Return features for SMOTE
        x = self.classifier(flattened)
        return x

# use pandas instead of cached
filenames = [f'iccv_activities_3s/midlevel.chunks_90.split_0.{name}.csv' for name in ['train_reduce', 'test', 'val']]
train_labels, val_labels, test_labels = [pd.read_csv(label_file) for label_file in filenames]

# Combine all labels to create a consistent mapping
all_labels = pd.concat([train_labels, val_labels, test_labels])

# Create a mapping from activity to numerical label
activity_to_label = {activity: i for i, activity in enumerate(all_labels['activity'].unique())}

# Paths for labels and preprocessed videos
train_label_file = "iccv_activities_3s/midlevel.chunks_90.split_0.train_reduce.csv"
val_label_file = "iccv_activities_3s/midlevel.chunks_90.split_0.val.csv"
test_label_file = "iccv_activities_3s/midlevel.chunks_90.split_0.test.csv"
preprocessed_video_folder = "a_column_driver_preprocessed_128/"

# Transformations applied within the dataset
transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# Create datasets using preprocessed videos
train_dataset = VideoDataset(preprocessed_video_folder, train_label_file, transform=transform)
val_dataset = VideoDataset(preprocessed_video_folder, val_label_file, transform=transform)
test_dataset = VideoDataset(preprocessed_video_folder, test_label_file, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) #Shuffle can be True if needed
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define paths and configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(pd.read_csv(train_label_file)['activity'].unique())  # Calculate number of classes based on the training labels

# Load pre-trained weights for Jester model

# Load the checkpoint
weights_path = os.path.join(os.getcwd(), "mobilenetv2_jester.pth")
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)

# Extract only the state_dict
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Filter out the unnecessary keys (like optimizer and epoch)
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):  # If the model was wrapped in nn.DataParallel, remove 'module.'
        key = key[7:]
    new_state_dict[key] = value

# Load the filtered state_dict into the model
model = MobileNetV2_3D_Jester(num_classes).to(device)
model.load_state_dict(new_state_dict, strict=False)

# Reinitialize the classifier layer
model.classifier = nn.Linear(128, num_classes)

model = model.to(device)  # Send the updated model to the device

# do it for all datasets
# Extract features and labels
loader_list = [train_loader]#, test_loader, val_loader]
for index, loader in enumerate(loader_list):
    features_list = []
    labels_list = []

    for batch in tqdm(loader):
        frames, labels = batch['frames'].to(device), batch['label'].to(device)

        with torch.no_grad():
            # Extract features using the model
            features = model(frames, return_features=True).cpu().numpy()  # Shape: [batch_size, 128]
            labels_np = labels.cpu().numpy()

            # Append to lists
            features_list.append(features)
            labels_list.append(labels_np)

    # Concatenate all batches
    features_all = np.concatenate(features_list, axis=0)  # Shape: [num_samples, 128]
    labels_all = np.concatenate(labels_list, axis=0)  # Shape: [num_samples]
    # save arrays
    np.save(f'loader_feature_t_{index}',features_all)
    np.save(f'loader_label_t_{index}',labels_all)
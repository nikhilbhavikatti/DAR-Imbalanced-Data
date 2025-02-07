import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import copy
import json
import seaborn as sns

def plot_accuracy(checkpoints, train_accuracies, val_accuracies, test_accuracies, model_name, save_path):
    """
    Plots accuracy over checkpoints for training and validation and saves the figure.
    Automatically selects the corresponding accuracy values from all epochs.
    """
    selected_train_accuracies = [train_accuracies[i-1] for i in checkpoints]
    selected_val_accuracies = [val_accuracies[i-1] for i in checkpoints]
    selected_test_accuracies = [test_accuracies[i-1] for i in checkpoints]
    
    plt.figure(figsize=(20, 16))
    plt.plot(checkpoints, selected_train_accuracies, marker='o', markersize=8, linewidth=2, label="Train Accuracy")
    plt.plot(checkpoints, selected_val_accuracies, marker='s', markersize=8, linewidth=2, label="Validation Accuracy")
    plt.plot(checkpoints, selected_test_accuracies, marker='^', markersize=8, linewidth=2, label="Test Accuracy")
    plt.xlabel("Checkpoints", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title(f"Accuracy Over Checkpoints - {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f"{save_path}/accuracy_{model_name}.png", bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, labels, save_path, normalize=True):
    """
    Plots a confusion matrix with color intensity representing the values and saves the figure.
    If normalize is True, the confusion matrix values are normalized as percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix to percentages if needed
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize each row
        cm = np.round(cm * 100).astype(int)  # Convert to percentage
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
                xticklabels=labels, yticklabels=labels, linewidths=1, linecolor='gray', annot_kws={"size": 16})
    for i in range(len(labels)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Actual", fontsize=16)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(f"{save_path}/confusion_matrix_{model_name}.png", bbox_inches='tight')
    plt.close()

def plot_accuracy_bar_val(models, final_accuracies, save_path):
    """
    Plots a bar chart comparing final accuracy of different models and saves the figure.
    """
    plt.figure(figsize=(20, 16))
    
    x_positions = np.arange(len(models))  # Explicitly define numerical x positions

    plt.bar(x_positions, final_accuracies, color=['blue', 'green', 'red', 'purple', 'orange'] * (len(models) // 5 + 1))
    plt.xlabel("Models", fontsize=14)
    plt.ylabel("Final Accuracy (%)", fontsize=14)
    plt.title("Final Accuracy Comparison", fontsize=16)

    # Set xticks at correct positions
    plt.xticks(ticks=x_positions, labels=models, rotation=60, ha='right', fontsize=12, fontweight='bold')
    
    plt.ylim([min(final_accuracies) - 5, 100])

    # Add labels above bars
    for i, v in enumerate(final_accuracies):
        plt.text(x_positions[i], v + 0.5, f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()  # Ensure layout fits within the figure
    plt.savefig(f"{save_path}/accuracy_comparison_val.png", bbox_inches='tight')
    plt.close()

def plot_accuracy_bar_test(models, final_accuracies, save_path):
    """
    Plots a bar chart comparing final accuracy of different models and saves the figure.
    """
    plt.figure(figsize=(20, 16))
    
    x_positions = np.arange(len(models))  # Explicitly define numerical x positions

    plt.bar(x_positions, final_accuracies, color=['blue', 'green', 'red', 'purple', 'orange'] * (len(models) // 5 + 1))
    plt.xlabel("Models", fontsize=14)
    plt.ylabel("Final Accuracy (%)", fontsize=14)
    plt.title("Final Accuracy Comparison", fontsize=16)

    # Set xticks at correct positions
    plt.xticks(ticks=x_positions, labels=models, rotation=60, ha='right', fontsize=12, fontweight='bold')
    
    plt.ylim([min(final_accuracies) - 5, 100])

    # Add labels above bars
    for i, v in enumerate(final_accuracies):
        plt.text(x_positions[i], v + 0.5, f"{v:.2f}%", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()  # Ensure layout fits within the figure
    plt.savefig(f"{save_path}/accuracy_comparison_test.png", bbox_inches='tight')
    plt.close()


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

# 2. Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

# 3. Feature Dataset
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features and labels to PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)  # Ensure features are floats
        self.labels = torch.tensor(labels, dtype=torch.long)  # Ensure labels are integers (for classification)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return tensors directly, which can be moved to CPU or CUDA using .to(device)
        return self.features[idx], self.labels[idx]
    
# 3.5 Define a custom dataset to load the sparse data
class SparseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.clip_files = sorted(os.listdir(data_dir))  # List all .npy files in the directory

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, idx):
        clip_path = os.path.join(self.data_dir, self.clip_files[idx])
        clip_data = np.load(clip_path, allow_pickle=True).item()  # Load the dictionary
        frames = torch.tensor(clip_data['frames'], dtype=torch.float32).squeeze(0)  # Convert to tensor
        features = torch.tensor(clip_data['features'], dtype=torch.float32)  # Convert to tensor
        labels = torch.tensor(clip_data['label'], dtype=torch.long).squeeze()  # Convert to tensor
        return frames, features, labels

# 4. Load features and labels
train_features, val_features, test_features = [np.load(f'loader_feature_{i}.npy') for i in range(3)]

# use pandas instead of cached
filenames = [f'iccv_activities_3s/midlevel.chunks_90.split_0.{name}.csv' for name in ['train', 'test', 'val']]
train_labels, val_labels, test_labels = [pd.read_csv(label_file) for label_file in filenames]

# Combine all labels to create a consistent mapping
all_labels = pd.concat([train_labels, val_labels, test_labels])

# Create a mapping from activity to numerical label
activity_to_label = {activity: i for i, activity in enumerate(all_labels['activity'].unique())}

# Apply the mapping to transform the labels in all datasets
train_labels = train_labels['activity'].map(activity_to_label)
val_labels = val_labels['activity'].map(activity_to_label)
test_labels = test_labels['activity'].map(activity_to_label)

train_labels, val_labels, test_labels = [np.load(f'loader_label_{i}.npy') for i in range(3)]

# Define your model (assuming MobileNetV2_3D_Jester is already defined)
num_classes = len(all_labels['activity'].unique())
pretrained_model = MobileNetV2_3D_Jester(num_classes)

# Load the checkpoint
weights_path = os.path.join(os.getcwd(), "mobilenetv2_jester.pth")
checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
#checkpoint = torch.load("content/mobilenetv2_jester.pth", map_location=torch.device('cpu'), weights_only=True)

# Extract the state_dict, handling cases where the checkpoint contains 'state_dict' or not
state_dict = checkpoint.get('state_dict', checkpoint)

# Remove 'module.' prefix if the model was wrapped in nn.DataParallel
state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

# Load the state_dict into the model, ignoring mismatches in the final layer
pretrained_model.load_state_dict(state_dict, strict=False)

# Reinitialize the final layer (classifier)
pretrained_model.classifier = nn.Linear(128, num_classes)

# 6. SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
smote_features, smote_labels = smote.fit_resample(train_features, train_labels)

# 7. Datasets and Dataloaders
train_dataset = FeatureDataset(train_features, train_labels)
smote_dataset = FeatureDataset(smote_features, smote_labels)
val_dataset = FeatureDataset(val_features, val_labels)
test_dataset = FeatureDataset(test_features, test_labels)

# 7. Weighted Sampler
weights = "sample_weights.json"
with open(weights, "r") as f:
    weights = json.load(f)
#class_counts = np.bincount(train_labels)
#weights = 1. / class_counts[train_labels]
sampler = WeightedRandomSampler(weights, len(weights))

# 8. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# 9. Training loop
criteria = [nn.CrossEntropyLoss(), FocalLoss()]
dataloaders = {
    'normal': DataLoader(train_dataset, batch_size=16, shuffle=True),
    'weighted': DataLoader(train_dataset, batch_size=16, sampler=sampler),
    'smote': DataLoader(smote_dataset, batch_size=16, shuffle=True)
}
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
learning_rates = [1e-2, 1e-3, 1e-4]

num_epochs = 1000
checkpoint_interval = 20

# Create a directory to save checkpoints
os.makedirs("checkpoints", exist_ok=True)

# Create a directory to save plots
os.makedirs("plots_new", exist_ok=True)
save_dir = "plots_new"

train_accuracies_dict = {}
val_accuracies_dict = {}
test_accuracies_dict = {}
final_accuracies_val = []
final_accuracies_test = []
models_trained = []
checkpoints_list = list(range(1, num_epochs + 1, checkpoint_interval))

for criterion in criteria:
    for loader_name, dataloader in dataloaders.items():
        for lr in learning_rates:
            model_name = f"{loader_name}_{criterion.__class__.__name__.lower()}_{lr:.0e}"
            train_accuracies = []
            val_accuracies = []
            test_accuracies = []
            # Reinitialize the model
            model = copy.deepcopy(pretrained_model).to(device)
            model.features.eval()
            model.classifier = nn.Linear(128, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Initialize variables to track the best validation accuracy
            best_val_accuracy = 0.0
            best_test_accuracy = 0.0

            # Wrap the epoch loop with tqdm for epoch-level progress tracking
            epoch_progress = tqdm(range(num_epochs), desc=f"Training {loader_name} ({criterion.__class__.__name__}, lr={lr})", unit="epoch")

            for epoch in epoch_progress:
                model.classifier.train()
                running_loss = 0.0
                correct_train = 0
                total_train = 0

                for batch_idx, batch in enumerate(dataloader):
                    features, labels = batch[0].to(device), batch[1].to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model.classifier(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, preds = torch.max(outputs, dim=1)
                    correct_train += (preds == labels).sum().item()
                    total_train += labels.size(0)

                train_accuracy = correct_train / total_train
                train_accuracies.append(train_accuracy * 100)

                # Validation phase
                model.eval()
                correct_val = 0
                total_val = 0
                y_true, y_pred = [], []

                with torch.no_grad():
                    for batch in val_loader:
                        features, labels = batch[0].to(device), batch[1].to(device)
                        outputs = model.classifier(features)
                        _, preds = torch.max(outputs, dim=1)
                        correct_val += (preds == labels).sum().item()
                        total_val += labels.size(0)
                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(preds.cpu().numpy())

                val_accuracy = correct_val / total_val
                val_accuracies.append(val_accuracy * 100)

                # Test phase
                correct_test = 0
                total_test = 0
                y_test_true, y_test_pred = [], []

                with torch.no_grad():
                    for batch in test_loader:
                        features, labels = batch[0].to(device), batch[1].to(device)
                        outputs = model.classifier(features)
                        _, preds = torch.max(outputs, dim=1)
                        correct_test += (preds == labels).sum().item()
                        total_test += labels.size(0)
                        y_test_true.extend(labels.cpu().numpy())
                        y_test_pred.extend(preds.cpu().numpy())

                test_accuracy = correct_test / total_test
                test_accuracies.append(test_accuracy * 100)

                # Update progress bar
                epoch_progress.set_postfix(
                    loss=f"{running_loss / len(dataloader):.4f}",
                    train_accuracy=f"{train_accuracy:.4f}",
                    val_accuracy=f"{val_accuracy:.4f}",
                    test_accuracy=f"{test_accuracy:.4f}"
                )

                # Save the model checkpoint every 200 epochs
                if (epoch + 1) % checkpoint_interval == 0:
                    checkpoint_name = f"{loader_name}_{criterion.__class__.__name__.lower()}_{lr:.0e}_{epoch+1:04d}.pth"
                    checkpoint_path = os.path.join("checkpoints", checkpoint_name)
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

                # Save the best model based on validation accuracy
                if val_accuracy > best_val_accuracy:
                    best_y_true = y_true
                    best_y_pred = y_pred
                    best_val_accuracy = val_accuracy
                    best_model_name = f"best_{loader_name}_{criterion.__class__.__name__.lower()}_{lr:.0e}.pth"
                    best_model_path = os.path.join("checkpoints", best_model_name)
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model saved: {best_model_path}")

                # Save the best model based on test accuracy
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy

            train_accuracies_dict[model_name] = train_accuracies
            val_accuracies_dict[model_name] = val_accuracies
            test_accuracies_dict[model_name] = test_accuracies
            final_accuracies_val.append(best_val_accuracy * 100)
            final_accuracies_test.append(best_test_accuracy * 100)
            models_trained.append(model_name)
            plot_confusion_matrix(best_y_true, best_y_pred, model_name, list(activity_to_label.keys()), "plots_new")

            print(f"Training completed for {loader_name} with {criterion.__class__.__name__} and lr={lr}")

# Save all accuracies as NumPy arrays
np.save(os.path.join(save_dir, "train_accuracies.npy"), train_accuracies_dict)
np.save(os.path.join(save_dir, "val_accuracies.npy"), val_accuracies_dict)
np.save(os.path.join(save_dir, "test_accuracies.npy"), test_accuracies_dict)
np.save(os.path.join(save_dir, "final_accuracies_val.npy"), final_accuracies_val)
np.save(os.path.join(save_dir, "final_accuracies_test.npy"), final_accuracies_test)

print(f"Accuracies saved in {save_dir}/ as .npy files")
# Generate plots for all models
for model_name in models_trained:
    plot_accuracy(checkpoints_list, 
                  train_accuracies_dict[model_name], 
                  val_accuracies_dict[model_name], 
                  test_accuracies_dict[model_name],
                  model_name,  "plots_new")
plot_accuracy_bar_val(models_trained, final_accuracies_val, "plots_new")
plot_accuracy_bar_test(models_trained, final_accuracies_test, "plots_new")

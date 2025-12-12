import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class MelDataset(Dataset):
    def __init__(self, data_dir, max_len=1500):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mel = torch.load(self.files[idx])

        # Convert multi-channel to mono
        if mel.shape[0] > 1:
            mel = torch.mean(mel, dim=0, keepdim=True)

        # Pad or truncate to fixed length
        if mel.shape[2] > self.max_len:
            mel = mel[:, :, :self.max_len]
        elif mel.shape[2] < self.max_len:
            pad_size = self.max_len - mel.shape[2]
            mel = nn.functional.pad(mel, (0, pad_size))

        label = 1 if 'fake' in self.files[idx].lower() else 0
        return mel, label


# Simple CNN Model
class VoiceCNN(nn.Module):
    def __init__(self):
        super(VoiceCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.relu = nn.ReLU()

        # FC layer size = 32 channels * 32 freq * 375 time
        self.fc1 = nn.Linear(32 * 32 * 375, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#############################################
#  ✔ TRAINING SHOULD RUN ONLY IF EXECUTED   #
#############################################
if __name__ == "__main__":
    data_dir = "Data/processed"
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001

    dataset = MelDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VoiceCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), "voice_cnn_model.pth")
    print("Training complete and model saved!")



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from utils import createDataLoaders

df = pd.read_csv("../data/sleep.csv")

df = df.drop(columns=["Person ID"])
df[["Systolic", "Diastolic"]] = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df = df.drop(columns=["Blood Pressure"])
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

categorical_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
df = pd.get_dummies(df, columns=categorical_cols)

y = df["Quality of Sleep"].values
X = df.drop(columns=["Quality of Sleep"]).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

class SleepNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output(x)
        return x

train_loader, test_loader = createDataLoaders(X_train, y_train, X_test, y_test, batch_size=32)

input_dim = X_train.shape[1]
model = SleepNet(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_y in train_loader:
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

model.eval()
test_losses = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)
        test_losses.append(loss.item())

print(f"Test MSE: {sum(test_losses)/len(test_losses):.4f}")

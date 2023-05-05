import json
import torch
import random
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import os

SCENES_JSON_PATH = 'scenes-train.json'
MODEL_PATH = './model_saves/pixel2world.pth'

batch_size = 32

class Pixel2World(nn.Module):
  def __init__(self):
    super().__init__()
    self.map = nn.Sequential(
      nn.Linear(5, 8),  nn.ReLU(),
      nn.Linear(8, 10), nn.ReLU(),
      nn.Linear(10, 3), nn.Tanh(),
    )

  def forward(self, x):
    return self.map(x)

def save_model(model):
  torch.save(model.state_dict(), MODEL_PATH)

def load_model():
    if not os.path.exists(MODEL_PATH):
        print('Training pixel to world model')
        train_pixel2world_model()
    model = Pixel2World()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def train_pixel2world_model():
    with open(SCENES_JSON_PATH, 'r') as f:
        scenes_data = json.load(f)
    X, y = [], []
    for scene0, scene1 in scenes_data['scenes']:
        for obj in (scene0['objects'] + scene1['objects']):
            X.append(obj['bbox'])
            y.append(obj['3d_coords'])
    X = torch.tensor(X)
    y = torch.tensor(y)

    print("X Shape", X.shape)
    print("Y Shape", y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_epochs = 400
    model = Pixel2World()
    criteria = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    val_loss_min = 1.00

    for epoch in range(num_epochs):
        for b in range(len(X_train)//batch_size):
          X = X_train[b*batch_size:b*(batch_size+1)]
          Y = Y_train[b*batch_size:b*(batch_size+1)]
          optimizer.zero_grad()
          Y_pred = model(X)
          loss = criteria(Y_pred, Y)
          loss.backward()
          optimizer.step()
        if epoch%10 == 0:
            print("Epoch:", epoch)
            print(f"Training Loss : {loss.item():.6f}")
            with torch.no_grad():
                Y_pred = model(X_test)
                loss = criteria(Y_pred, Y_test)
                print(f"Validation Loss : {loss.item():.6f}")
                print("True World Coordinates     ", Y_test[:5])
                print("Predicted World Coordinates", Y_pred[:5])
                if loss < val_loss_min:
                    save_model(model)
                    val_loss_min = loss
                    print("Model saved!")

if __name__ == '__main__':
    train_pixel2world_model()
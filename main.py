import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pre import split_data
from sklearn.metrics import r2_score

device = torch.device("cuda")


# y type is like [R, M]

def data_to_tensor(df, y):
    x = df
    y_d = torch.zeros((len(y), 2)).to(device)
    x_data = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float).to(device)
    for i in range(x.shape[0]):
        if list(y)[i] == "R":
            y_d[i][0] = 1
            y_d[i][1] = 0
        else:
            y_d[i][0] = 0
            y_d[i][1] = 1
        aux = x.iloc[i]
        for j in range(x.shape[1]):
            x_data[i][j] = aux[j]

    return TensorDataset(x_data, y_d)


X_train, X_test, Y_train, Y_test = split_data()

train_set = DataLoader(
    data_to_tensor(X_train, Y_train),
    batch_size=23,
    shuffle=True
)

test_set = DataLoader(
    data_to_tensor(X_test, Y_test)
)

temp = pd.concat([X_test, Y_test], axis=1)
temp = temp.reset_index().drop(columns=["index"])
temp.to_csv("Test.csv")

class Modelo(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        out = self.out(x)

        return torch.softmax(out, dim=1)


model = Modelo(60, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()

def train(input, real):
    optimizer.zero_grad()

    out = model(input)
    lost = loss(out, real)

    lost.backward()

    optimizer.step()

    return lost


epochs = 120
for epoch in range(epochs):
    epoch_loss = 0
    for i, (input, real) in tqdm(enumerate(train_set)):
        epoch_loss += train(input, real)
    
    print("Epoch: ", epoch, " Loss: ", epoch_loss / i)


results = pd.DataFrame(columns=["Real", "Probability_of_Rock", "Probability_of_Metal"])
real_r = []
prob_r = []
prob_m = []
for i, (input, real) in enumerate(test_set):
    with torch.no_grad():
        output = model(input)
        real_r.append(torch.argmax(real[0]).item())
        prob_r.append(output[0][0].item())
        prob_m.append(output[0][1].item())
results["Real"] = real_r
results["Probability_of_Rock"] = prob_r
results["Probability_of_Metal"] = prob_m
results.to_csv("results.csv")
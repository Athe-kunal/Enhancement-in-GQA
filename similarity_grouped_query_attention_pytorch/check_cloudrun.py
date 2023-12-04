import torch
import torch.nn as nn
import torch.optim as optim
import time

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 50),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.fc(x)

def train(model, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data = torch.randn(64, 100).to(device)
    target = torch.randint(0, 10, (64,)).to(device)

    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    print(loss)
    return end_time - start_time

device = torch.device('cuda:0')
model = DummyModel().to(device)
time_taken_single_gpu = train(model, device)
print(f"Time taken on single GPU: {time_taken_single_gpu:.2f} seconds")

if torch.cuda.device_count() > 1:
    model = DummyModel()
    model = nn.DataParallel(model)
    model.to('cuda')
    time_taken_multi_gpu = train(model, 'cuda')
    print(f"Time taken on multiple GPUs: {time_taken_multi_gpu:.2f} seconds")
else:
    print("Multiple GPUs are not available.")

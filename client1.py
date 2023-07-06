import socket
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Define the model architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
trainset = datasets.MNIST('./data', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))

# Create a socket for server-client communication
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'
port = 12345
client_socket.connect((host, port))

# Send a message to the server to indicate client connection
client_socket.send(b"Client connected")

# Receive a message from the server
message = client_socket.recv(1024)
print(message.decode())

# Define the model and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model on the client
for epoch in range(15):
    model.train()
    running_loss = 0.0

    for images, labels in trainset:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print('Epoch: {} Loss: {:.4f}'.format(epoch + 1, running_loss / len(trainset)))

# Send the updated model weights to the server
client_socket.send(b"Model weights")

# Receive the aggregated model from the server
model_weights = client_socket.recv(1024)
model.load_state_dict(torch.load(model_weights))

# Close the socket
client_socket.close()




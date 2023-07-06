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

# Create the model and optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the server training function
def train_on_batches(model, criterion, optimizer, data_batches):
    model.train()
    running_loss = 0.0

    for images, labels in data_batches:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(data_batches)

# Perform federated learning
def federated_learning(model, criterion, optimizer, data_loaders, num_epochs):
    for epoch in range(num_epochs):
        aggregated_loss = 0.0

        for client_id, data_loader in data_loaders.items():
            client_loss = train_on_batches(model, criterion, optimizer, data_loader)
            aggregated_loss += client_loss

            print(f"Client {client_id}: Epoch {epoch + 1} Loss: {client_loss}")

        print(f"Aggregated Loss: {aggregated_loss}")

# Create a socket for server-client communication
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'
port = 12345
server_socket.bind((host, port))
server_socket.listen(2)  # Listen for a maximum of 2 clients

print("Server is waiting for clients to connect...")

# Accept connections from clients
client_sockets = []
for _ in range(2):
    client_socket, addr = server_socket.accept()
    print("Client connected:", addr)
    client_sockets.append(client_socket)

# Split trainset into two client datasets
split_idx = len(trainset) // 2
client1_dataset = torch.utils.data.Subset(trainset, range(0, split_idx))
client2_dataset = torch.utils.data.Subset(trainset, range(split_idx, len(trainset)))

# Create data loaders for each client dataset
client1_loader = torch.utils.data.DataLoader(client1_dataset, batch_size=64, shuffle=True)
client2_loader = torch.utils.data.DataLoader(client2_dataset, batch_size=64, shuffle=True)

# Create data loader dictionary for clients
data_loaders = {
    "client1": client1_loader,
    "client2": client2_loader
}

# Perform federated learning on the server
federated_learning(model, criterion, optimizer, data_loaders, num_epochs=15)

# Save the trained model
torch.save(model.state_dict(), 'server_model.pth')

# Close the sockets
for client_socket in client_sockets:
    client_socket.close()
server_socket.close()


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Przygotowanie danych
transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# 2. Definicja prostego autoenkodera
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, 2)    # Tu kompresujemy do 2 wymiarów (do wizualizacji)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

# 3. Trening
for epoch in range(10):
    for images, _ in trainloader:
        output, _ = autoencoder(images)
        loss = criterion(output, images.view(-1, 28*28))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 4. Wydobywanie i wizualizacja "mapy pojęć"
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000, shuffle=False)
images, labels = next(iter(testloader))
with torch.no_grad():
    _, encoded = autoencoder(images)
encoded = encoded.numpy()
labels = labels.numpy()

plt.figure(figsize=(8,6))
for i in range(10):
    idx = labels == i
    plt.scatter(encoded[idx, 0], encoded[idx, 1], label=str(i), alpha=0.6, s=20)
plt.legend()
plt.title('Emergentna mapa pojęć: autoenkoder na MNIST')
plt.xlabel('Kod 1'); plt.ylabel('Kod 2')
plt.show()

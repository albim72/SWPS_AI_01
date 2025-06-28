import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Tworzymy dane: x od 0 do 99, y = 2x + 1
x_train = torch.arange(0, 100, dtype=torch.float32).reshape(-1, 1)  # shape (100, 1)
y_train = 2 * x_train + 1                                          # shape (100, 1)

# 2. Definiujemy prosty model liniowy
model = nn.Linear(in_features=1, out_features=1)

# 3. Ustawiamy funkcję straty i optymalizator
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 4. Trening modelu przez 1000 epok
for epoch in range(1000):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoka {epoch} | Strata: {loss.item():.4f}")

# 5. Sprawdźmy wyniki na kilku przykładowych danych
x_test = torch.tensor([[150.0], [200.0], [250.0]])
y_test_pred = model(x_test)

print("\nTestowe predykcje:")
for i in range(x_test.shape[0]):
    print(f"x = {x_test[i].item():.1f} → y = {y_test_pred[i].item():.2f}")

# 6. Opcjonalnie: wykres
with torch.no_grad():
    plt.scatter(x_train.numpy(), y_train.numpy(), label="Prawdziwe dane")
    plt.plot(x_train.numpy(), model(x_train).numpy(), color='red', label="Predykcja modelu")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model regresji liniowej: y = 2x + 1")
    plt.grid(True)
    plt.show()

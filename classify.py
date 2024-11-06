from dataset import ShapeNetDataset
import torch
from model import CNN3D

batch_size = 16
num_epochs = 40
save_dir = "Classifier.keras"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dataset = ShapeNetDataset(device)

train_set, test_set = torch.utils.data.random_split(dataset, [0.75, 0.25])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = CNN3D().to(device)
# model.load_state_dict(torch.load(save_dir, weights_only=True)) <-- Load saved model

loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

def validation_accuracy():
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct/ total * 100

# Training loop ------------------------------------------------------------------------------------

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    acc = validation_accuracy()
    print(f"Epoch: {epoch + 1}. Loss: {loss.item()} Validation Accuracy : {acc}")

torch.save(model.state_dict(), save_dir)
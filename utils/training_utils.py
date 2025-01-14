import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from utils.parse_args import parse_args
from utils.setup import DeviceLoader
from classes.EarlyStopping import EarlyStopping
from torch.utils.data import DataLoader

args = parse_args()

def _print(*pargs):
    if args['verbose']:
        print(*pargs)
    if args['log']:
        file_path = 'results/' + args['model'] + '.log'
        with open(file_path, 'a') as f:
            print(*pargs, file=f)

def confusion_matrix(y: Tensor, y_pred: Tensor, class_count: int) -> Tensor:
    confusion_matrix = torch.zeros(class_count, class_count, dtype=torch.int64)
    for true, prediction in zip(y, y_pred):
        confusion_matrix[true, prediction] += 1
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix: Tensor, class_names: list[str]) -> None:
    class_count = len(class_names)
    _, ax = plt.subplots()
    _ = ax.imshow(confusion_matrix, cmap='summer')
    ax.set_xticks(range(class_count))
    ax.set_yticks(range(class_count))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='center')
    ax.xaxis.set_label_position('top') 

    for i in range(class_count):
        for j in range(class_count):
            _ = ax.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Display a colorbar for the confusion matrix
    cbar = ax.figure.colorbar(ax.imshow(confusion_matrix, cmap='summer'), ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Number of images', rotation=-90, va='bottom')

    # Save image
    if args['save_fig']:
        plt.savefig(f'./results/{args['model']}.png')

    plt.show()

def train_one_epoch(model: nn.Module, trainloader: DeviceLoader,
                optimizer: torch.optim.Optimizer, loss: nn.modules.loss) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0

    # Every 10 percent of the batches, print the average loss
    print_batch = int(len(trainloader) / 10)
    for batch, (X, y) in enumerate(trainloader, 0):
        optimizer.zero_grad()
        predictions = model(X)
        current_loss = loss(predictions, y)
        current_loss.backward()
        optimizer.step()

        correct += (torch.argmax(predictions, dim=1) == y).sum().item()
        loss_value = current_loss.item()
        running_loss += loss_value
        total_loss += loss_value 
        if (batch + 1) % print_batch == 0:
            avg_loss = running_loss / print_batch
            _print(f'\t[{int(((batch + 1) / print_batch) * 10)}% of Batches]: Loss = {avg_loss:.3f}')
            running_loss = 0.0
    accuracy = correct / (len(trainloader) * model.batch_size)
    return accuracy, total_loss

def validation_loss(model: nn.Module, validationloader: DeviceLoader,
                loss: nn.modules.loss) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in validationloader:
            predictions = model(X)

            correct += (torch.argmax(predictions, dim=1) == y).sum().item()
            total_loss += loss(predictions, y).item()
    accuracy = correct / (len(validationloader) * model.batch_size)
    return accuracy, total_loss

def train(model: nn.Module, trainloader: DataLoader,
          validationloader: DataLoader, optimizer: torch.optim.Optimizer, 
          lossfn: nn.modules.loss, device: torch.device, epochs: int = 10) -> None:

    trainloader = DeviceLoader(device, trainloader, 'training')
    validationloader = DeviceLoader(device, validationloader, 'validation')

    _print("Starting training")
    early_stop = EarlyStopping(patience=5, delta=0.5)

    for epoch in range(epochs):
        _print(f'=========== Epoch: {epoch + 1} ===========')
        t_acc, total_tloss = train_one_epoch(model, trainloader, optimizer, lossfn)
        v_acc, total_vloss = validation_loss(model, validationloader, lossfn)

        _print(f'Train Loss: {(total_tloss / len(trainloader)):.2f},\tValidation Loss: {(total_vloss / len(validationloader)):.2f}')
        _print(f'Train Accuracy: {(100 * t_acc):.2f}%,\tValidation Accuracy: {(100 * v_acc):.2f}%\n')
        if early_stop(total_tloss, total_vloss):
            _print('Early stopping')
            break

def test(model: nn.Module, testloader: DataLoader,
         lossfn: nn.modules.loss, device: torch.device) -> tuple[float, Tensor]:

    # If it can fit the validation data, it can fit test data
    testloader = DeviceLoader(device, testloader, 'validation')

    conf_matrix = torch.zeros(model.class_count, model.class_count, dtype=torch.int64)
    model.eval()
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for X, y in testloader:
            pred = model(X)
            test_loss += lossfn(pred, y).item()
            _, predicted = torch.max(pred, 1)
            conf_matrix += confusion_matrix(y, predicted, model.class_count)
            correct += (predicted == y).sum().item()
    accuracy = correct / (len(testloader) * model.batch_size)
    _print(f'Accuracy: {(100 * accuracy):.2f}%, Avg. Loss: {(test_loss / len(testloader)):.2f}')
    return accuracy, conf_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils.parse_args import parse_args, ELOAD
from utils.setup import device_data_loader
from classes.EarlyStopping import EarlyStopping
args = parse_args()

def _print(*pargs):
    if args['verbose']:
        print(*pargs)
    if args['log']:
        file_path = 'results/' + args['model'] + '.log'
        with open(file_path, 'a') as f:
            print(*pargs, file=f)

def confusion_matrix(y, y_pred, class_count):
    confusion_matrix = torch.zeros(class_count, class_count, dtype=torch.int64)
    for true, prediction in zip(y, y_pred):
        confusion_matrix[true, prediction] += 1
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, class_names):
    _, ax = plt.subplots()
    _ = ax.imshow(confusion_matrix, cmap='summer')
    ax.set_xticks(range(len(class_names))) 
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='center')
    ax.xaxis.set_label_position('top') 

    for i in range(len(class_names)):
        for j in range(len(class_names)):
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
    
        

# def precision_recall(confusion_matrix):
#     precision = torch.zeros(confusion_matrix.size(0))
#     recall = torch.zeros(confusion_matrix.size(0))
#     for i in range(confusion_matrix.size(0)):
#         precision[i] = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
#         recall[i] = confusion_matrix[i, i] / confusion_matrix[i, :].sum()
#     return precision, recall

def train_one_epoch(model: nn.Module, trainloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, loss: nn.modules.loss, 
                    device: torch.device) -> None:
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    
    # Every 10 percent of the batches, print the average loss
    print_batch = int(len(trainloader) / 10)
    for batch, (X, y) in enumerate(trainloader, 0):
        if args['full_device_load'] == ELOAD.NONE:
            X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        current_loss = loss(pred, y)
        current_loss.backward()
        optimizer.step()
        
        correct += (torch.argmax(pred, dim=1) == y).sum().item()
        loss_value = current_loss.item()
        running_loss += loss_value
        total_loss += loss_value 
        if (batch + 1) % print_batch == 0:
            avg_loss = running_loss / print_batch
            _print(f'\t[Batch: {((batch + 1)):3d}]: Loss = {avg_loss:.3f}')
            running_loss = 0.0
    accuracy = correct / (len(trainloader) * model.batch_size)
    return accuracy, total_loss

def validation_loss(model: nn.Module, validationloader: torch.utils.data.DataLoader,
                    loss: nn.modules.loss, device: torch.device) -> torch.Tensor:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for X, y in validationloader:
            if args['full_device_load'] != ELOAD.TRAINING_VALIDATION:
                X, y = X.to(device), y.to(device)
            pred = model(X)
            
            correct += (torch.argmax(pred, dim=1) == y).sum().item()
            current_loss = loss(pred, y)
            total_loss += current_loss.item()
    accuracy = correct / (len(validationloader) * model.batch_size)
    return accuracy, total_loss

def train(model: nn.Module, trainloader: torch.utils.data.DataLoader,
          validationloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          lossfn: nn.modules.loss, device: torch.device, epochs: int = 10) -> None:

    if args['full_device_load'] >= ELOAD.TRAINING:
        _print('Loading the training data into the GPU memory')
        trainloader = device_data_loader(device, trainloader)
    if args['full_device_load'] == ELOAD.TRAINING_VALIDATION:
        _print('Loading the validation data into the GPU memory')
        validationloader = device_data_loader(device, validationloader)
    
    _print("Starting training")
    early_stop = EarlyStopping(patience=5, delta=0.5)

    for epoch in range(epochs):
        _print(f'=========== Epoch: {epoch + 1} ===========')
        t_acc, total_tloss = train_one_epoch(model, trainloader, optimizer, lossfn, device)
        v_acc, total_vloss = validation_loss(model, validationloader, lossfn, device)
        
        _print(f'Train Loss: {(total_tloss / len(trainloader)):.2f},\tValidation Loss: {(total_vloss / len(validationloader)):.2f}')
        _print(f'Train Accuracy: {(100 * t_acc):.2f}%,\tValidation Accuracy: {(100 * v_acc):.2f}%\n')
        if early_stop(total_tloss, total_vloss):
            _print('Early stopping')
            break
    
    
def test(model: nn.Module, testloader: torch.utils.data.DataLoader, 
         lossfn: nn.modules.loss, device: torch.device) -> torch.Tensor:
    
    if args['full_device_load'] >= ELOAD.TRAINING:
        """ If it can fit the training data, it can fit the test data """
        testloader = device_data_loader(device, testloader)
    
    conf_matrix = torch.zeros(model.class_count, model.class_count, dtype=torch.int64)
    model.eval()
    correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for X, y in testloader:
            if args['full_device_load'] == ELOAD.NONE:
                X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += lossfn(pred, y).item()
            _, predicted = torch.max(pred, 1)
            conf_matrix += confusion_matrix(y, predicted, model.class_count)
            correct += (predicted == y).sum().item()
    accuracy = correct / (len(testloader) * model.batch_size)
    _print(f'Accuracy: {(100 * accuracy):.2f}%, Avg. Loss: {(test_loss / len(testloader)):.2f}')
    return accuracy, conf_matrix
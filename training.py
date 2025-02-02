from torch import nn, optim
from torchsummary import summary
from utils.parse_args import parse_args
from utils.setup import get_device, get_model
from utils.training_utils import plot_confusion_matrix, train, test
from classes.CovidDataloader import covid19_dataloaders, COVID19Dataset

args = parse_args()
device = get_device()

dataset = COVID19Dataset()
class_names = dataset.get_classes()

trainloader, validationloader, testloader = covid19_dataloaders()

X, y = trainloader.dataset[0]
net = get_model(args, len(class_names))

# Summary of the model
summary(net, X.size())

# Train and evaluate the model
train(net, trainloader=trainloader, validationloader=validationloader, 
        epochs=net.epochs, optimizer=optim.Adam(net.parameters(), lr=net.lr, betas=(0.9, 0.99)),
        device=device, lossfn=nn.CrossEntropyLoss())

# Test the model
_, conf_matrix = test(net, testloader=testloader, lossfn=nn.CrossEntropyLoss(), device=device)

# Print and display the confusion matrix
print(conf_matrix)
plot_confusion_matrix(conf_matrix, class_names=class_names)
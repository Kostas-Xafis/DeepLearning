import torch
from torch.utils.data import DataLoader
from utils.parse_args import parse_args

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_to_device(X, y, device):
    return X.to(device), y.to(device)

def device_data_loader(device: torch.device, dl: DataLoader) -> list[tuple[torch.Tensor, torch.Tensor]]:
    new_dl = []
    for X, y in dl:
        new_dl.append(load_to_device(X, y, device))
    return new_dl

def get_model(args: dict[str, any], class_count: int) -> torch.nn.Module:
    device = get_device()
    if args['model'] == 'cnn1':
        from models.CNN1 import CNN1
        return CNN1(args, class_count).to(device)
    elif args['model'] == 'cnn2':
        from models.CNN2 import CNN2
        return CNN2(args, class_count).to(device)
    elif args['model'] == 'cnn-basicblock':
        from models.CNN3 import CNN3
        return CNN3(args, class_count).to(device)
    elif args['model'] == 'resnet50':
        from models.ResNet50 import Resnet50
        return Resnet50(args, class_count).to(device)
    elif args['model'] == 'resnet50-pretrained':
        from models.ResNet50 import Resnet50Pretrained
        return Resnet50Pretrained(args, class_count).to(device)
    else:
        raise ValueError(f'Unknown model: {args["model"]}')

class DeviceLoader:
    def __init__(self, device: torch.device = None, dataloader: DataLoader = None, loader_type: str = None):
        self.device = device if device is not None else get_device()
        self.args = parse_args()
        self.fully_loaded = False

        self.dataset = dataloader
        self.fully_loaded = False
        if self.__should_load(loader_type):
            print(f'Loading {loader_type} data to device: {self.device}')
            self.dataset = device_data_loader(self.device, dataloader)
            self.fully_loaded = True

    def __iter__(self):
        if self.fully_loaded:
            self.current = 0
        else:
            self.dataset_iter = iter(self.dataset)
        return self

    def __next__(self):
        if self.fully_loaded:
            if self.current < len(self.dataset):
                self.current += 1
            else:
                raise StopIteration
            return self.dataset[self.current - 1]
        else:
            return load_to_device(*self.dataset_iter.__next__(), self.device)

    def __should_load(self, loader_type: str):
        if self.args is None or self.args['full_device_load'] == 'none':
            return False

        if self.args['full_device_load'] == loader_type or self.args['full_device_load'] == 'training_validation':
            return True

        return False

    def __len__(self):
        return len(self.dataset)
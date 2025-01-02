import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_to_device(X, y, device):
    return X.to(device), y.to(device)

def get_model(args: dict[str, any], class_count: int, img_size: torch.Size) -> torch.nn.Module:
    if args['model'] == 'cnn1':
        from models.CNN1 import CNN1
        return CNN1(args, class_count, img_size).to(get_device())
    elif args['model'] == 'cnn2':
        from models.CNN2 import CNN2
        return CNN2(args, class_count, img_size).to(get_device())
    elif args['model'] == 'resnet50':
        from models.ResNet50 import Resnet50
        return Resnet50(args, class_count).to(get_device())
    elif args['model'] == 'resnet50-pretrained':
        from models.ResNet50 import Resnet50Pretrained
        return Resnet50Pretrained(args, class_count).to(get_device())
    elif args['model'] == 'cnn-basicblock':
        from models.CNN3 import CNN3
        return CNN3(args=args, class_count=class_count, img_size=img_size).to(get_device())
    else:
        raise ValueError(f'Unknown model: {args["model"]}')

def device_data_loader(device: torch.device, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    new_dataloader = []
    for X, y in dataloader:
        new_dataloader.append(load_to_device(X, y, device))
    return new_dataloader
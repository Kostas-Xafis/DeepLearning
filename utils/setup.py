import torch

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_to_device(X, y, device):
    return X.to(device), y.to(device)

def get_model(args: dict[str, any], total_classes: int, img_size: torch.Size) -> torch.nn.Module:
    if args['model'] == 'cnn1':
        from models.cnn1 import CNN1
        return CNN1(args, total_classes, img_size).to(get_device())
    elif args['model'] == 'cnn2':
        from models.cnn2 import CNN2
        return CNN2(args, total_classes, img_size).to(get_device())
    elif args['model'] == 'cnn3':
        # from models.cnn3 import CNN3
        # return CNN3(args)\
        raise NotImplementedError('CNN3 not implemented yet')
    elif args['model'] == 'resnet50':
        # from models.resnet50 import ResNet50
        # return ResNet50(args)
        raise NotImplementedError('ResNet50 not implemented yet')
    else:
        raise ValueError(f'Unknown model: {args["model"]}')

def device_data_loader(device: torch.device, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
    new_dataloader = []
    for X, y in dataloader:
        new_dataloader.append(load_to_device(X, y, device))
    return new_dataloader
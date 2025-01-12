import os
import math
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

DATASET_DIR = 'dataset/COVID-19_Radiography_Dataset'

class COVID19Dataset(Dataset):
    def __init__(self, root_dir=DATASET_DIR, transform=None, load=True):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_names = []
        if not load:
            return
        self.load_datapaths()

    def load_datapaths(self):
        # Get all subdirectories
        sub_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) ]

        # Load all image paths and labels
        for label_idx, subdir in enumerate(sub_dirs):
            subdir_path = os.path.join(self.root_dir, subdir)
            self.label_names.append(subdir)
            
            sub_dir_paths = os.listdir(subdir_path)
            for img_name in sub_dir_paths:
                self.image_paths.append(subdir_path + '/' + img_name)
                self.labels.append(label_idx)
    
    def display_batch(self, indexes: list[int]):
        sample_size = len(indexes)
        rows = math.ceil(sample_size ** 0.5)
        cols = math.ceil(sample_size / rows)
        
        _, axes = plt.subplots(rows, cols, figsize=(15, 15))
        
        for ax, idx in zip(axes.flatten(), indexes):
            image, label = self[idx]
            ax.imshow(image.permute(1, 2, 0))
            ax.set_title(self.label_names[label])
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Bar chart for total amount of images for each label
        label_counts = [self.labels.count(i) for i in range(len(set(self.labels)))]
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(label_counts)), label_counts, tick_label=self.label_names)
        plt.xlabel('Labels')
        plt.ylabel('Number of images')
        plt.title('Number of images per label')
        plt.show()

    def get_classes(self):
        if len(self.label_names) == 0:
            self.labels = []
            self.label_names = []
            # Get all subdirectories
            dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d)) ]

            # Load all image paths and labels
            for label_idx, subdir in enumerate(dirs):
                subdir_path = os.path.join(self.root_dir, subdir)
                self.label_names.append(subdir)

        return self.label_names
    
    def set_transform(self, transform: transforms.Compose):
        self.transform = transform
        
    def add_transform(self, transform):
        if isinstance(transform, transforms.Compose):
            self.transform = transform
            return
        if self.transform is None:
            self.transform = transforms.Compose([transform])
        elif isinstance(self.transform, transforms.Compose):
            self.transform.transforms.append(transform)
        else:
            self.transform = transforms.Compose([self.transform, transform])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


def get_mean_std(dataset: COVID19Dataset = None, batches=None):
    batch_size = 32
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

        if nb_samples >= batches * batch_size:
            break

    mean /= nb_samples
    std /= nb_samples

    return mean, std

def get_covid19_single_dataloader(batch_size=64):
    dataset = COVID19Dataset(transform=transforms.ToTensor())
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_covid19_split(ds_size = 100, image_size = 100):
    dataset = COVID19Dataset()
    if image_size != 100:
        img = dataset[0][0]
        # Resize by actual percentage and not percentage squared
        w,h = img.width, img.height
        nw = int(w - (w * ((100 - image_size) / 2 / 100)))
        nh = int(h - (h * ((100 - image_size) / 2 / 100)))
        print('New image size:', nw, 'x', nh)
        dataset.add_transform(transforms.Compose([transforms.ToTensor(), transforms.Resize((nw, nh))]))
    else:
        dataset.add_transform(transforms.ToTensor())
    generator = torch.Generator().manual_seed(42)

    if ds_size == 100:
        return random_split(dataset, [0.6, 0.2, 0.2], generator=generator)
    keep_ds = ds_size / 100
    return random_split(dataset, [keep_ds * 0.6, keep_ds * 0.2, keep_ds * 0.2,  1 - keep_ds], generator=generator)

def covid19_dataloaders():
    from utils.parse_args import parse_args
    
    args = parse_args()
    ds_size = args['dataset_size']
    batch_size = args['batch_size']
    image_size = args['image_resize']

    split = get_covid19_split(ds_size, image_size)
    train, val, test = split[:3] if ds_size != 100 else split
    
    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=False)
    val = DataLoader(val, batch_size=batch_size, shuffle=False)

    return (train, val, test)
    
def estimate_dataset_memory(tensor: list[torch.Tensor, int], size: int = 1):
    single_sample_memory = \
        (tensor[0].numel())\
            * torch.tensor([], dtype=torch.float32).element_size() \
            + 1 * torch.tensor([], dtype=torch.int32).element_size()
    
    return size * single_sample_memory / 1000**3


# Example usage
if __name__ == "__main__":
    import random
    dataset = COVID19Dataset(transform=transforms.ToTensor())
    
    dataset.display_batch(random.sample(range(len(dataset)), 25))
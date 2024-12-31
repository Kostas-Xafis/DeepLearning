import sys
import argparse
from torchvision import transforms
from classes.CovidDataloader import COVID19Dataset, estimate_dataset_memory

class ELOAD:
    NONE = 1
    TRAINING = 2
    TRAINING_VALIDATION = 3

def parser(arg_specs: dict[str, tuple]):
    parser = argparse.ArgumentParser()
    
    help_format = lambda arg: f'{arg[2]} - {arg[0].__name__.capitalize()} - (default: {arg[1]}).' 
    
    for arg, arg_type in arg_specs.items():
        help = help_format(arg_type)
        if type(arg_type[0]) == bool:
            parser.add_argument(f'--{arg}', action=argparse.BooleanOptionalAction, help=help, default=arg_type[1])
        else:
            parser.add_argument(f'--{arg}', type=arg_type[0], help=help, default=arg_type[1])

    args = parser.parse_args()
    return vars(args)


def parse_cnn_args():
    arg_specs = {
        'model': (str, 'cnn1', 'Cnn model to use. "cnn1"|"cnn2"|"cnn3"|"resnet50"'),
        'batch_size': (int, 64, 'Batch size'),
        'epochs': (int, 20, 'Number of epochs'),
        'lr': (float, 10e-3, 'Learning rate'),
        'full_device_load': (int, 1, 'Load the dataset into the GPU memory: 1 (None), 2 (Training data), 3 (Training & Validation data)'),
        'dataset_size': (int, 1, 'Size of the dataset. 2 (Full), 1 (Small)'),
        # 'h|help': (bool, False, 'Print this message.')
    }
    args = parser(arg_specs)
    if len(args) == 0:
        return None
    if 'help' in args:
        for arg, arg_type in arg_specs.items():
            if arg == 'help':
                print(f'--{arg}: Print this message.') 
            print(f'--{arg}: {arg_type[0]} (default: {arg_type[1]}).')
        return None
    return args

def gather_args():
    args = {}
    # Get the model argument first:
    print('Choose a model to use: ')
    print('(1.) Cnn1')
    print('(2.) Cnn2')
    print('(3.) Cnn3')
    print('(4.) ResNet50')
    inp = input('Enter a number: ')
    inp = int(inp)
    if inp > 4 or inp < 1:
        print('Invalid input. Exiting...')
        return None
    args['model'] = ['cnn1', 'cnn2', 'cnn3', 'resnet50'][int(inp)-1]
    
    if args['model'] == 'resnet50':
        args['batch_size'] = 64
        args['epochs'] = 5
        args['lr'] = 10e-4
    elif args['model'] == 'cnn1':
        args['batch_size'] = 64
        args['epochs'] = 20
        args['lr'] = 10e-3
        pass
    elif args['model'] == 'cnn2':
        args['batch_size'] = 64
        args['epochs'] = 20
        args['lr'] = 10e-3
    else:
        args['batch_size'] = 64
        args['epochs'] = 20
        args['lr'] = 10e-3
    
    inp = input('\nUse the full dataset or a smaller one (1:Full or 2:Small): ')
    if inp != '2' and inp != '1':
        print('Invalid input. Exiting...')
        return None
    args['dataset_size'] = 'Full' if inp == '1' else 'Small'
    
    # Ask the user if he wants to load the full dataset into the GPU memory or not
    dataset_inp_size = 1 if args['dataset_size'] == 'Full' else 0.1
    dataset = COVID19Dataset(transform=transforms.ToTensor())
    dataset_size = len(dataset)
    data_sample = dataset[0]
    training_data_size = estimate_dataset_memory(data_sample, int(dataset_size * 0.6 * dataset_inp_size))
    validation_data_size = estimate_dataset_memory(data_sample, int(dataset_size * 0.2 * dataset_inp_size))
    
    print(f'\nEstimated VRAM usage for the training data: {training_data_size:.2f} GB')
    print(f'Estimated VRAM usage for the training & validation data: {training_data_size + validation_data_size:.2f} GB')
    print('Load the full dataset into the GPU memory?')
    print('(1.) None')
    print('(2.) Training data')
    print('(3.) Training & Validation data')
    inp = input('Εnter a number: ')
    
    args['full_device_load'] = int(inp)
    
    if args['full_device_load'] < 1 or args['full_device_load'] > 3:
        print('Invalid input. Exiting...')
        return None
            
    return args


final_args = None
def parse_args():
    global final_args
    if final_args is not None:
        return final_args
    
    total_args = sys.argv
    if len(total_args) == 1:
        final_args = gather_args()
    else:
        final_args = parse_cnn_args()
    
    if final_args is None:
        sys.exit(0)
    return final_args

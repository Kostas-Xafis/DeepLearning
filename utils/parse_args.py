from calendar import c
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
        'model': (str, 'cnn1', 'Cnn model to use. "cnn1"|"cnn2"| "resnet50"|"resnet50-pretrained"|"cnn-basicblock"'),
        'batch_size': (int, None, 'Batch size'),
        'epochs': (int, None, 'Number of epochs'),
        'lr': (float, None, 'Learning rate'),
        'full_device_load': (int, 1, 'Load the dataset into the GPU memory: 1 (None), 2 (Training data), 3 (Training & Validation data)'),
        'dataset_size': (int, 1, 'Size of the dataset. 1 (Full), 2 (Small)'),
        'image_resize': (int, 100, 'Image resize from 1% to 100%'),
        'verbose': (bool, False, 'Print verbose output'),
        'log': (bool, False, 'Log the output to a file'),
        'save_fig': (bool, False, 'Save the confusion matrix to a file'),
        # 'help': (bool, False, 'Print this message.')
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
    
    args['dataset_size'] = 100 if args['dataset_size'] == 1 else 10
    
    pick_model_specific_params(args, exclude=[ x for x in ['batch_size', 'epochs', 'lr'] if args[x] is not None])
    print(args)
    return args

def gather_args():
    args = {}
    # Get the model argument first:
    print('Choose a model to use: ')
    print('(1.) Cnn1')
    print('(2.) Cnn2')
    print('(3.) ResNet50')
    print('(4.) ResNet50 - Pretrained')
    print('(5.) BasicBlock based CNN')
    inp = int(input('Enter a number: '))
    if inp < 1 or inp > 5:
        print('Invalid input. Exiting...')
        return None
    args['model'] = ['cnn1', 'cnn2', 'resnet50', 'resnet50-pretrained', 'cnn-basicblock'][inp - 1]
    
    # Set the batch size, epochs and learning rate based on the model
    pick_model_specific_params(args)
    
    inp = input('\nUse the full dataset or a smaller one (1:Full or 2:Small): ')
    if inp != '2' and inp != '1':
        print('Invalid input. Exiting...')
        return None
    args['dataset_size'] = 100 if inp == '1' else 10
    args['image_resize'] = 100
    
    while True:
        # Ask the user if he wants to load the full dataset into the GPU memory or not or try and resize the images
        dataset = COVID19Dataset(transform=transforms.ToTensor())
        dataset_size = len(dataset)
        data_sample = dataset[0]
        dataset_inp_size = args['dataset_size'] / 100
        img_factor = args['image_resize'] / 100
        
        training_data_size = estimate_dataset_memory(data_sample, int(dataset_size * 0.6 * dataset_inp_size * img_factor))
        validation_data_size = estimate_dataset_memory(data_sample, int(dataset_size * 0.2 * dataset_inp_size * img_factor))
        
        print(f'\nEstimated VRAM usage for the training data: {training_data_size:.2f} GB')
        print(f'Estimated VRAM usage for the training & validation data: {training_data_size + validation_data_size:.2f} GB')
        print('Load the full dataset into the GPU memory?')
        print('(1.) None')
        print('(2.) Training data')
        print('(3.) Training & Validation data')
        print('(4.) Resize the images')
        inp = input('Εnter a number: ')
        
        if inp == '4':
            inp = input('Enter the new image size (1-100)%: ')
            args['image_resize'] = int(inp)
            if args['image_resize'] < 1 or args['image_resize'] > 100:
                print('Invalid input. Exiting...')
                return None
            continue
        
        args['full_device_load'] = int(inp)
        
        if args['full_device_load'] < 1 or args['full_device_load'] > 3:
            print('Invalid input. Exiting...')
            return None
        break
        
            
    return args

def pick_model_specific_params(args: dict, exclude: list = []):
    bs, ep, lr = 64, 20, 10e-3
    if args['model'] == 'resnet50-pretrained':
        bs, ep, lr = (64, 5, 10e-4)
    elif args['model'] == 'resnet50':
        bs, ep, lr = (64, 5, 10e-4)
    elif args['model'] == 'cnn1':
        bs, ep, lr = (64, 20, 10e-3)
    elif args['model'] == 'cnn2':
        bs, ep, lr = (64, 20, 10e-3)

    if 'batch_size' not in exclude:
        args['batch_size'] = bs
    if 'epochs' not in exclude:
        args['epochs'] = ep
    if 'lr' not in exclude:
        args['lr'] = lr        

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

'''
This library contains functions which might be useful for the training process.
'''

from torch.optim import SGD, Adam
from nets import SwinT, ResNet

def create_optimizer(parameters, args):
    '''
    Function to create the optimizer instance to use during training.
    This function reads the arguments passed in the args parameter to build the output optimizer
    
    Parameters
    ----------
    parameters : Iterable[torch.nn.Parameter]
        The parameters of the model to train.
    args : Namespace
        Arguments parsed from input and describing the optimizer to create.
    
    Returns
    -------
    optimizer : torch.optim.SGD || torch.optim.Adam
        The optimizer built based on the parameters passed as input.
    '''
    match args.optimizer:
        case 'sgd':
            return SGD(parameters,
                       lr=args.learning_rate, 
                       momentum=args.sgd_momentum, 
                       weight_decay=args.sgd_weight_decay)
        case 'adam':
            return Adam(parameters,
                        lr=args.learning_rate)
        case _:
            raise ValueError("Invalid value for parameter `optimizer`.")
        
def create_model(args):
    '''
    Function to create the model instance to use during training.
    This function reads the arguments passed in the args parameter to build the model to train.
    
    Parameters
    ----------
    args : Namespace
        Arguments parsed from input and describing the model to create.
    
    Returns
    -------
    model : torch.nn.Module
        The network built based on the parameters passed as input.
    '''
    match args.network:
        case 'resnet':
            return ResNet(args.rn_arch,
                          head=args.rn_head, 
                          pretrained=args.rn_pretrained,
                          first_trainable=args.rn_first_trainable,
                          pretraining_model=args.pretraining_model)
        case 'swint':
            return SwinT(head=args.st_head, 
                         pretrained=args.st_pretrained,
                         first_trainable=args.st_first_trainable,
                         pretraining_model=args.pretraining_model)
        case _:
            raise ValueError("Invalid value for parameter `network`.")
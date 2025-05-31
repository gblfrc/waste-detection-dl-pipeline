'''
This library contains functions which might be useful at inference time.
'''

from nets import SwinT, ResNet

def create_model(args):
    '''
    Function to create the model instance to use during training.
    This function reads the arguments passed in the args parameter to build the model to use for inference.
    
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
                          pretrained=None,
                          first_trainable=4,
                          pretraining_model=args.pretraining_model)
        case 'swint':
            return SwinT(head=args.st_head, 
                         pretrained=None,
                         first_trainable=4,
                         pretraining_model=args.pretraining_model)
        case _:
            raise ValueError("Invalid value for parameter `network`.")
        
def get_target_layers(model):
    '''
    Function to obtain the list of target layers for computing CAMs given the target model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for computing CAMs.

    Returns
    -------
    target_layers : [torch.nn.Module]
        The list of layers to target when computing CAMs.
    '''
    # Select layer based on model class
    if isinstance(model, ResNet):
        return [model.layer4]
    elif isinstance(model, SwinT):
        return [model.norm]
    else:
        raise Exception("Target layer not defined for the given model.")

def get_reshape_transform(model):
    '''
    Function to obtain the list reshape transformation function for computing CAMs given the target model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for computing CAMs.

    Returns
    -------
    reshape_transform : function
        The function to apply to the the tensor at the target layer for computing CAMs.
    '''
    # Select layer based on model class
    if isinstance(model, ResNet):
        return None
    elif isinstance(model, SwinT):
        return lambda tensor:tensor.transpose(2, 3).transpose(1, 2)
    return None
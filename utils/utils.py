import torch 


def save_checkpoint(save_name, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, save_name)
    print('model saved to {}'.format(save_name))

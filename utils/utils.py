import torch 


def save_checkpoint(save_name, model, z, optimizer):
    state = {
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()
             }
    z_ts = torch.stack(z)
    torch.save(z, save_name+'_latents.pt')
    torch.save(state, save_name+'.pth')
    print('model saved to {}'.format(save_name))

def load_checkpoint(save_name, model):
    
    z_ts = torch.load(save_name+'_latents.pt')
    z_lst = []
    for i in z_ts.size()[0]:
        indices = torch.LongTensor([i])
        z_lst.append(torch.index_select(x, 0, indices))
    model_dict=model.load_state_dict(torch.load(save_name+'.pth'))
    
   
    model_CKPT = torch.load(save_name)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    
    return model, z_lst, optimizer
    

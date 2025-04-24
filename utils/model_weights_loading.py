import torch

def model_weights_loading(model, args, task_num_pre):
    model.net.encoder.load_state_dict(torch.load(args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task_num_pre)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt',
                                            map_location='cuda:0'))
    model.net.decoder.load_state_dict(torch.load(args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task_num_pre)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt',
                                            map_location='cuda:0'))
    return model
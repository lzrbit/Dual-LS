import torch
import os

def model_weights_saving(args, epoch, task, model, dataset):
    if args.model == 'clser':
        if epoch==(args.n_epochs-1):
            save_path_encoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'
            save_path_decoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'
            save_dir = os.path.dirname(save_path_encoder)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.stable_model.encoder.state_dict(), save_path_encoder)
            torch.save(model.stable_model.decoder.state_dict(), save_path_decoder) 
    else:
        if not hasattr(model, 'end_task'):
            if epoch==(args.n_epochs-1):
                save_path_encoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'
                save_path_decoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'
                save_dir = os.path.dirname(save_path_encoder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.net.encoder.state_dict(), save_path_encoder)
                torch.save(model.net.decoder.state_dict(), save_path_decoder)
        #A-GEM, GEM methods
        elif hasattr(model, 'end_task'):
            model.end_task(dataset)          
            if epoch==(args.n_epochs-1):
                save_path_encoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_encoder'+'.pt'
                save_path_decoder = args.saved_dir+'/'+args.model+'_'+'tasks_'+str(task+1)+'_'+'bf_'+str(args.buffer_size)+'_decoder'+'.pt'
                save_dir = os.path.dirname(save_path_encoder)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(model.net.encoder.state_dict(), save_path_encoder)
                torch.save(model.net.decoder.state_dict(), save_path_decoder)
    return 0

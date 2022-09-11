
sweep_config = {
    'method': 'bayes'
    }
metric = {
    'name': 'dice_coeff',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric


parameters_dict={}

model_args_dict={ 'arch': {
        'values': ['resnet18','resnet34','resnet50','resnet101','resnet152']},
        'self_attention':{'values':[True,False]}
}

loss_args_dict={
    
     'loss_fn': {
        'values': ['focal_loss','wted_crossentropy']},
    
    'small_bowel_wt': {
        'distribution': 'q_log_uniform_values',
        'q': 10,
        'min': 10,
        'max': 1000
      },
    
    'stomach_wt': {
        'distribution': 'q_log_uniform_values',
        'q': 10,
        'min': 10,
        'max': 1000
      },
    'large_bowel_wt': {
        'distribution': 'q_log_uniform_values',
        'q': 10,
        'min': 10,
        'max': 1000
      },
    'background_wt':{'values':[1.]}
    }

    
optimizer_args_dict={ 'learning_rate': {
        'distribution': 'q_log_uniform_values',
        'q': 10,
        'min': 1e-4,
        'max': 100
      }
         }

dataloader_args_dict={
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 2,
        'min': 8,
        'max': 32,
      }


}

epochs_dict={'epochs': {
        'values': [5,10,20]}}    
    
    
parameters_dict.update(model_args_dict)
parameters_dict.update(loss_args_dict)
parameters_dict.update(optimizer_args_dict)
parameters_dict.update(epochs_dict)
parameters_dict.update(dataloader_args_dict)
sweep_config['parameters'] = parameters_dict
sweep_id=wandb.sweep(sweep_config)


arch_dict={'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,'resnet101':resnet101,'resnet152':resnet152}
loss_fns_dict={'focal_loss':focal_loss,'wted_crossentropy':weighted_crossentropy_flat}

import ml_collections

Synapse_train_list=['label0006','label0007' ,'label0009', 'label0010', 'label0021' ,'label0023' ,'label0024','label0026' ,'label0027' ,'label0031', 'label0033' ,'label0034','label0039', 'label0040','label0005', 'label0028', 'label0030', 'label0037']
Synapse_val_list  =['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022','label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

Synapse13_train_list=['label0006','label0007' ,'label0009', 'label0010', 'label0021' ,'label0023' ,'label0024','label0026' ,'label0027' ,'label0031', 'label0033' ,'label0034','label0039', 'label0040','label0005', 'label0028', 'label0030', 'label0037']
Synapse13_val_list  =['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022','label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']




def EM_512():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = EM_train_list
    config.val_list = EM_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512,512]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 7e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [16,16,16,8]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 2000
    config.hyper_parameter.convolution_stem_down = 8
   
    return config

def ISIC_512():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = ISIC_train_list
    config.val_list = ISIC_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [512,512]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1.3e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [8,8,16,8]
    config.hyper_parameter.val_eval_criterion_alpha = 0
    config.hyper_parameter.epochs_num = 75
    config.hyper_parameter.convolution_stem_down = 8
   
    return config

def ACDC_224():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = False
    config.train_list = ACDC_train_list
    config.val_list = ACDC_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224,224]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Large'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [7,7,14,7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.9
    config.hyper_parameter.epochs_num = 500
    config.hyper_parameter.convolution_stem_down = 4
   
    return config

def Synapse_224():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224,224]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Base'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [7,7,14,7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 2700
    config.hyper_parameter.convolution_stem_down = 4
   
    return config
    
def Synapse_320():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = Synapse_train_list
    config.val_list = Synapse_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [320,320]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [10,10,20,10]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 1300
    config.hyper_parameter.convolution_stem_down = 4
   
    return config
    

def Synapse13_224():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = Synapse13_train_list
    config.val_list = Synapse13_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [224,224]
    config.hyper_parameter.batch_size = 16
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Base'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [7,7,14,7]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 2700
    config.hyper_parameter.convolution_stem_down = 4
   
    return config
    
    
def Synapse13_320():
    config = ml_collections.ConfigDict()
    
    config.pretrain = False
    config.deep_supervision = True
    config.train_list = Synapse13_train_list
    config.val_list = Synapse13_val_list
    
    config.hyper_parameter = ml_collections.ConfigDict()
    config.hyper_parameter.crop_size = [320,320]
    config.hyper_parameter.batch_size = 8
    config.hyper_parameter.base_learning_rate = 1e-4
    config.hyper_parameter.model_size = 'Tiny'
    config.hyper_parameter.blocks_num = [3,3,3,3]
    config.hyper_parameter.window_size = [10,10,20,10]
    config.hyper_parameter.val_eval_criterion_alpha = 0.
    config.hyper_parameter.epochs_num = 1300
    config.hyper_parameter.convolution_stem_down = 4
   
    return config



    
CONFIGS = {
    'EM_512':EM_512(),
    'ISIC_512':ISIC_512(),
    'ACDC_224':ACDC_224(),
    'Synapse_224':Synapse_224(),
    'Synapse_320':Synapse_320(),
    'Synapse13_224':Synapse13_224(),
    'Synapse13_320':Synapse13_320()
}




import ml_collections

Synapse_train_list=['label0006','label0007' ,'label0009', 'label0010', 'label0021' ,'label0023' ,'label0024','label0026' ,'label0027' ,'label0031', 'label0033' ,'label0034','label0039', 'label0040','label0005', 'label0028', 'label0030', 'label0037']
Synapse_val_list  =['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022','label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

Synapse13_train_list=['label0006','label0007' ,'label0009', 'label0010', 'label0021' ,'label0023' ,'label0024','label0026' ,'label0027' ,'label0031', 'label0033' ,'label0034','label0039', 'label0040','label0005', 'label0028', 'label0030', 'label0037']
Synapse13_val_list  =['label0001', 'label0002', 'label0003', 'label0004', 'label0008', 'label0022','label0025', 'label0029', 'label0032', 'label0035', 'label0036', 'label0038']

ACDC_train_list=['patient001_frame01', 'patient001_frame12', 'patient004_frame01','patient004_frame15', 'patient005_frame01', 'patient005_frame13','patient006_frame01', 'patient006_frame16', 'patient007_frame01','patient007_frame07', 'patient010_frame01', 'patient010_frame13','patient011_frame01', 'patient011_frame08', 'patient013_frame01','patient013_frame14', 'patient015_frame01', 'patient015_frame10','patient016_frame01', 'patient016_frame12', 'patient018_frame01','patient018_frame10', 'patient019_frame01', 'patient019_frame11','patient020_frame01', 'patient020_frame11', 'patient021_frame01','patient021_frame13', 'patient022_frame01', 'patient022_frame11','patient023_frame01', 'patient023_frame09', 'patient025_frame01','patient025_frame09', 'patient026_frame01', 'patient026_frame12','patient027_frame01', 'patient027_frame11', 'patient028_frame01','patient028_frame09', 'patient029_frame01', 'patient029_frame12','patient030_frame01', 'patient030_frame12', 'patient031_frame01','patient031_frame10', 'patient032_frame01', 'patient032_frame12','patient033_frame01', 'patient033_frame14', 'patient034_frame01','patient034_frame16', 'patient035_frame01', 'patient035_frame11','patient036_frame01', 'patient036_frame12', 'patient037_frame01','patient037_frame12', 'patient038_frame01', 'patient038_frame11','patient039_frame01', 'patient039_frame10', 'patient040_frame01','patient040_frame13', 'patient041_frame01', 'patient041_frame11','patient043_frame01', 'patient043_frame07', 'patient044_frame01','patient044_frame11', 'patient045_frame01', 'patient045_frame13','patient046_frame01', 'patient046_frame10', 'patient047_frame01','patient047_frame09', 'patient050_frame01', 'patient050_frame12','patient051_frame01', 'patient051_frame11', 'patient052_frame01','patient052_frame09', 'patient054_frame01', 'patient054_frame12','patient056_frame01', 'patient056_frame12', 'patient057_frame01','patient057_frame09', 'patient058_frame01', 'patient058_frame14','patient059_frame01', 'patient059_frame09', 'patient060_frame01','patient060_frame14', 'patient061_frame01', 'patient061_frame10','patient062_frame01', 'patient062_frame09', 'patient063_frame01','patient063_frame16', 'patient065_frame01', 'patient065_frame14','patient066_frame01', 'patient066_frame11', 'patient068_frame01','patient068_frame12', 'patient069_frame01', 'patient069_frame12','patient070_frame01', 'patient070_frame10', 'patient071_frame01','patient071_frame09', 'patient072_frame01', 'patient072_frame11','patient073_frame01', 'patient073_frame10', 'patient074_frame01','patient074_frame12', 'patient075_frame01', 'patient075_frame06','patient076_frame01', 'patient076_frame12', 'patient077_frame01','patient077_frame09', 'patient078_frame01', 'patient078_frame09','patient080_frame01', 'patient080_frame10', 'patient082_frame01','patient082_frame07', 'patient083_frame01', 'patient083_frame08','patient084_frame01', 'patient084_frame10', 'patient085_frame01','patient085_frame09', 'patient086_frame01', 'patient086_frame08','patient087_frame01', 'patient087_frame10']
ACDC_val_list  =['patient089_frame01', 'patient089_frame10', 'patient090_frame04','patient090_frame11', 'patient091_frame01', 'patient091_frame09','patient093_frame01', 'patient093_frame14', 'patient094_frame01','patient094_frame07', 'patient096_frame01', 'patient096_frame08','patient097_frame01', 'patient097_frame11', 'patient098_frame01','patient098_frame09', 'patient099_frame01', 'patient099_frame09','patient100_frame01', 'patient100_frame13']

EM_train_list=['train-labels00', 'train-labels01', 'train-labels02', 'train-labels03','train-labels04', 'train-labels05', 'train-labels06', 'train-labels07', 'train-labels08', 'train-labels10', 'train-labels12', 'train-labels14', 'train-labels15', 'train-labels16', 'train-labels17', 'train-labels19', 'train-labels20', 'train-labels23', 'train-labels24', 'train-labels25', 'train-labels26', 'train-labels27', 'train-labels28', 'train-labels29']
EM_val_list  =['train-labels09','train-labels11','train-labels13']


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
    'ACDC_224':ACDC_224(),
    'Synapse_224':Synapse_224(),
    'Synapse_320':Synapse_320(),
    'Synapse13_224':Synapse13_224(),
    'Synapse13_320':Synapse13_320()
}




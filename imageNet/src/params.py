
def getTrainingParams(version=0):
    if version == 0:
        params = {
            'batch_size'    :32,
            'optimizer'     :"SGD",
            'learning_rate' :0.001,
            'momentum'      :0.9,
            'nesterov'      :True,
            'reduce_lr'     :False
        } 
    elif version == 1:
        params = {
            'batch_size'    :32,
            'optimizer'     :"SGD",
            'learning_rate' :0.02,
            'momentum'      :0.9,
            'nesterov'      :True,
            'reduce_lr'     :True,
            'rlr_monitor'   :'val_loss',
            'rlr_factor'    :0.5,
            'rlr_patience'  :10,
            'rlr_mode'      :'auto',
            'rlr_min_delta' :0.0001,
            'rlr_cooldown'  :0,
            'rlr_min_lr'    :0.0001
        }         
    else:
        raise ValueError("param version " + str(version) + " does not exist.")
    
    return params
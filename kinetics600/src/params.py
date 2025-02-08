
def getTrainingParams(version=0):
    if version == 0:
        params = {
            'batch_size'    :32,
            'optimizer'     :"SGD",
            'learning_rate' :0.001,
            'momentum'      :0.9,
            'nesterov'      :True,
            'decay'         :0.0,
            'reduce_lr'     :False
        }     
    else:
        raise ValueError("param version " + str(version) + " does not exist.")
    
    return params
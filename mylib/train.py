def train(model='LSTM',file_name=\
    ['label.txt','skeleton.txt','motion.avi'],path='data'):
    from mylib.data_import import dataCreate
    from mylib.dataRead import data_read
    
    data_read(file_name=file_name,path=path).run()
    dataCreate(model=model,path=path).run()

    if model == 'LSTM':
        from LSTM_Train.lstm import LSTM
        return LSTM()
    elif model == 'CNN':
        from CNN_Train.keleton_based_classfication import CNN
        return CNN()

def create_data(model='LSTM',file_name=\
    ['label.txt','skeleton.txt','motion.avi'],path='data'):
    from mylib.data_import import dataCreate
    from mylib.dataRead import data_read
    
    data_read(file_name=file_name,path=path).run()
    dataCreate(model=model,path=path).run()

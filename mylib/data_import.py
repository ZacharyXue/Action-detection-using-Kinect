import numpy as np

class dataCreate:
    '''
    use this class to read data and to create test data.
    selectedLabel needs to input labels which is selected as training target;
    joints is setted as Kinect skeleton joints as default.
    '''
    def __init__(self,n_steps=30,label_num=7,path='data',model='LSTM'):
        self.n_steps = n_steps   
        self.size = label_num
        self.path = path
        self.model = model

    # create vector label
    def labelCreate(self,label):
        labelArr = np.zeros([1,self.size])
        labelArr[0,label] = 1
        return labelArr

    # standard data's size
    def add2List(self,rawSkeleton,label):
        '''
        output:Skeleton,Label
        '''
        import cv2
        allSkeleton = np.array([])
        allLabel = np.array([])
        # if data's size is more than 200, split it into several data
        if rawSkeleton.shape[0] > self.n_steps:
            num = int(rawSkeleton.shape[0] / self.n_steps) + 1
            for i in range(1,num + 1):
                start = self.n_steps * i
                end = start + self.n_steps
                if end > rawSkeleton.shape[0]:
                    end = -1
                    start = end - self.n_steps
                allSkeleton = np.append(allSkeleton,rawSkeleton[start:end,:])
                labelArr = self.labelCreate(label)
                allLabel = np.append(allLabel,labelArr)
        # if data's size is less than 200, add zeros
        elif rawSkeleton.shape[0] < self.n_steps:
            rawSkeleton = cv2.resize(rawSkeleton,(self.n_steps,rawSkeleton.shape[1]))

            allSkeleton = np.append(allSkeleton,rawSkeleton)

            labelArr = self.labelCreate(label)
            allLabel = np.append(allLabel,labelArr)
        else:
            allSkeleton = np.append(allSkeleton,rawSkeleton)

            labelArr = self.labelCreate(label)
            allLabel = np.append(allLabel,labelArr)
        
        return allSkeleton,allLabel

    # import data, and put them into an numpy array
    def dataImport(self):
        '''
        output: skeleton,label
        '''
        import os,sys 
        parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        sys.path.insert(0,parentdir) 
        from mylib.preprocess import preprocessing 

        all_skeleton = np.array([])
        all_label = np.array([])

        for i in os.listdir(self.path+'/data2'):
            if i == 'data.npz':
                continue
            for j in os.listdir('{}/data2/{}'.format(self.path,i)):
                try:
                    # ====pay attention to this address, every time move python file, remember to change it====#
                    rawData = np.loadtxt('{}/data2/{}/{}'.format(self.path,i,j))
                except:
                    pass
                else:
                    _pre = preprocessing(pos=rawData)
                    _skeleton = _pre.run()
                    # print("_data size is {}".format(_data.shape))                        
                    skeleton,label = self.add2List(_skeleton,int(i))
                    all_skeleton = np.append(all_skeleton,skeleton)
                    all_label = np.append(all_label,label)
                    # print("allData size is {}".format(allData.shape))

        all_label = np.reshape(all_label,[-1,self.size])
        all_skeleton = np.reshape(all_skeleton,[-1,self.n_steps,60])
        return all_skeleton,all_label
    
    # create the test data and traning data
    def testCreate(self):
        '''
        output: skeleton,label,test_skeleton,test_label
        '''
        skeleton,label = self.dataImport()
        # allLabel = np.reshape(allLabel,[-1,len(self.selectedLabel)])
        size = skeleton.shape[0]

        test_num = [ i for i in range(size) if i % 10 == 0]

        test_skeleton = np.array(skeleton[test_num])
        train_skeleton = np.delete(skeleton,test_num,axis=0)
        test_label = np.array(label[test_num])
        train_label = np.delete(label,test_num,axis=0)

        permutation1 = np.random.permutation(train_skeleton.shape[0] - 1)
        shuffled_skeleton = train_skeleton[permutation1, :]
        shuffled_label = train_label[permutation1, :]
        permutation2 = np.random.permutation(test_skeleton.shape[0])
        shuffled_test_skeleton = test_skeleton[permutation2, :]
        shuffled_test_label = test_label[permutation2, :]
        return shuffled_skeleton,shuffled_label,shuffled_test_skeleton,shuffled_test_label
    
    def motion_create(self):
        self.motion = self.skeleton[:,1:] - self.skeleton[:,:-1]
        self.motion = np.reshape(self.motion,[-1,29,20,3])
        # self.motion = np.reshape(self.motion,[-1,29,14,3])

        self.skeleton = self.skeleton[:,1:]
        self.skeleton = np.reshape(self.skeleton,[-1,29,20,3])
        # self.skeleton = np.reshape(self.skeleton,[-1,29,14,3])


        self.test_motion = self.test_skeleton[:,1:] - self.test_skeleton[:,:-1]
        self.test_motion = np.reshape(self.test_motion,[-1,29,20,3])
        # self.test_motion = np.reshape(self.test_motion,[-1,29,14,3])
        self.test_skeleton = self.test_skeleton[:,1:]
        self.test_skeleton = np.reshape(self.test_skeleton,[-1,29,20,3])
        # self.test_skeleton = np.reshape(self.test_skeleton,[-1,29,14,3])

    def run(self):
        '''
        create training data(with label) and test data(with label)
        tempSet: only output test data
        '''
        self.skeleton,self.label,self.test_skeleton,self.test_label = self.testCreate()
        if self.model == 'LSTM':      
            np.savez("{}/data2/data.npz".format(self.path),skeleton=self.skeleton\
                ,label=self.label,test_skeleton=self.test_skeleton,test_label=self.test_label)
        elif self.model == 'CNN':
            self.motion_create()
            np.savez("{}/data2/data.npz".format(self.path),skeleton=self.skeleton,motion=self.motion,label=self.label,\
                test_skeleton=self.test_skeleton,test_motion=self.test_motion,test_label=self.test_label)
    
    def data(self):
        '''
        input data from npz
        '''
        temp = np.load("{}/data2/data.npz".format(self.path))
        if self.model == 'LSTM':           
            self.skeleton =  temp['skeleton']
            self.label = temp['label']
            self.test_skeleton = temp['test_skeleton']
            self.test_label = temp['test_label']
        elif self.model == 'CNN':
            self.skeleton =  temp['skeleton']
            self.motion = temp['motion']
            self.label = temp['label']
            self.test_skeleton = temp['test_skeleton']
            self.test_motion = temp['test_motion']
            self.test_label = temp['test_label']


    def next_batch(self,epoch=0,batch_size=100,flag=0):
        '''
        create the batch of the data
        input epoch, dont't worry about epoch is too big;
        output is batch_data and batch_label
        '''
        if self.model == 'LSTM':
            if flag == 0:
                skeleton = self.skeleton.copy()
                label = self.label.copy()
            else:
                skeleton = self.test_skeleton.copy()
                label = self.test_label.copy()
            start = ( epoch * batch_size ) % skeleton.shape[0]
            end = start + batch_size
            if end > skeleton.shape[0]:
                end = -1

            batch_skeleton = skeleton[start:end]
            batch_label = label[start:end]
            return batch_skeleton,batch_label
        elif self.model == 'CNN':
            if flag == 0:
                skeleton = self.skeleton.copy()
                motion = self.motion.copy()
                label = self.label.copy()
            else:
                skeleton = self.test_skeleton.copy()
                motion = self.test_motion.copy()
                label = self.test_label.copy()
            start = ( epoch * batch_size ) % skeleton.shape[0]
            end = start + batch_size
            if end > skeleton.shape[0]:
                end = -1

            batch_skeleton = skeleton[start:end]
            batch_motion = motion[start:end]
            batch_label = label[start:end]
            return batch_skeleton,batch_motion,batch_label

# try to find out the class if it works or not
if __name__ == "__main__":    
    dataCreate(model='CNN').run()
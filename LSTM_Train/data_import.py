import numpy as np
import cv2

class dataCreate:
    '''
    use this class to read data and to create test data.
    selectedLabel needs to input labels which is selected as training target;
    joints is setted as Kinect skeleton joints as default.
    '''
    def __init__(self,selectedLabel,n_steps=30,joints=20):
        self.selectedLabel = selectedLabel
        self.n_steps = n_steps
        # self.joints = joints   
        self.joints = 14
        self.size = len( self.selectedLabel )

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
        from preprocess import preprocessing 

        all_skeleton = np.array([])
        all_label = np.array([])

        lastStrList = ['L','M','R']
        for num in self.selectedLabel:
            for i in range(1,27):
                for j in range(1,14):
                    for lastStr in lastStrList:
                        try:
                            # ====pay attention to this address, every time move python file, remember to change it====#
                            rawData = np.loadtxt('LSTM_Train/data/{}/A{:0>2d}N{:0>2d}-{}.txt'.format(num,i,j,lastStr))
                        except:
                            pass
                        else:
                            _pre = preprocessing(pos=rawData)
                            _skeleton = _pre.run()
                            # print("_data size is {}".format(_data.shape))
                            label = self.selectedLabel.index(num)                        
                            skeleton,label = self.add2List(_skeleton,label)
                            all_skeleton = np.append(all_skeleton,skeleton)
                            all_label = np.append(all_label,label)
                            # print("allData size is {}".format(allData.shape))

        all_label = np.reshape(all_label,[-1,self.size])
        all_skeleton = np.reshape(all_skeleton,[-1,self.n_steps,self.joints*3])
        return all_skeleton,all_label
    
    # create the test data and traning data
    def testCreate(self):
        '''
        output: skeleton,label,test_skeleton,test_label
        '''
        skeleton,label = self.dataImport()
        # print(data.shape)
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
    
    def run(self):
        '''
        create training data(with label) and test data(with label)
        tempSet: only output test data
        '''
        self.skeleton,self.label,self.test_skeleton,self.test_label = self.testCreate()
    
    def data_import(self):
        # temp = np.load("LSTM_Train/PKUMMD1.npz")
        temp = np.load("LSTM_Train/PKUMMD3.npz")
        self.skeleton =  temp['skeleton']
        self.label = temp['label']
        self.test_skeleton = temp['test_skeleton']
        self.test_label = temp['test_label']

    def next_batch(self,epoch=0,batch_size=100,flag=0):
        '''
        create the batch of the data
        input epoch, dont't worry about epoch is too big;
        output is batch_data and batch_label
        '''
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

# try to find out the class if it works or not
if __name__ == "__main__":    
    test = dataCreate([11,13,19,41])
    test.run()
    print(test.skeleton.shape)
    # np.savez("LSTM_Train/PKUMMD1.npz",skeleton=test.skeleton,label=test.label,\
    #     test_skeleton=test.test_skeleton,test_label=test.test_label)
    np.savez("LSTM_Train/PKUMMD3.npz",skeleton=test.skeleton,label=test.label,\
        test_skeleton=test.test_skeleton,test_label=test.test_label)
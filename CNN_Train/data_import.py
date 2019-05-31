import numpy as np
import cv2

class dataCreate:
    '''
    use this class to read data and to create test data.
    selectedLabel needs to input labels which is selected as training target;
    joints is setted as Kinect skeleton joints as default.
    '''
    def __init__(self,n_steps=30,joints=60):
        self.n_steps = n_steps
        self.joints = joints 

    def data_init(self):
        temp = np.load("LSTM_Train/PKUMMD1.npz")
        self.skeleton =  temp['skeleton']
        self.label = temp['label']
        self.test_skeleton = temp['test_skeleton']
        self.test_label = temp['test_label']

    def motion_create(self):
        self.motion = self.skeleton[:,1:] - self.skeleton[:,:-1]
        self.motion = np.reshape(self.motion,[-1,29,20,3])
        self.skeleton = self.skeleton[:,1:]
        self.skeleton = np.reshape(self.skeleton,[-1,29,20,3])

        self.test_motion = self.test_skeleton[:,1:] - self.test_skeleton[:,:-1]
        self.test_motion = np.reshape(self.test_motion,[-1,29,20,3])
        self.test_skeleton = self.test_skeleton[:,1:]
        self.test_skeleton = np.reshape(self.test_skeleton,[-1,29,20,3])
    
    def run(self):
        self.data_init()
        self.motion_create()

    def data_import(self):
        temp = np.load("CNN_Train/PKUMMD2.npz")
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
    test = dataCreate([11,13,19,41])
    test.run()
    # test.data_import()
    # data,dataLabel = test.next_batch()
    # data,label = test.labelCreate(data)
    print(test.skeleton.shape)
    # print(dataLabel.shape)
    # print(data[-1,-1,0])
    # print(dataLabel[-1])
    # print(test.shape)
    np.savez("CNN_Train/PKUMMD2.npz",skeleton=test.skeleton,motion=test.motion,label=test.label,\
        test_skeleton=test.test_skeleton,test_motion=test.test_motion,test_label=test.test_label)
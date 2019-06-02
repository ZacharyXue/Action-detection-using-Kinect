import numpy as np
import re
import os

'''
the label form should like that: label,start_frame,end_frame
'''
# training_file = range(0,19)
class data_read():
    '''
    Label is number and it should apppear firstly in the txt
    Every file in 'data' should have 'label.txt' and 'skeleton.txt'
    '''
    def __init__(self,file_name=['label.txt','skeleton.txt','motion.avi'],path='data'):
        '''
        'path' is the path of the data
        'file_name' is the name of [label_file, skeleton_file]
        '''
        self.file_name = file_name
        self.path = path

    def run(self):
        
        for i in os.listdir(self.path):   
            try:
                flieLabel = open('{}/{}/{}'.format(self.path,i,self.file_name[0]))
            except:
                pass
            else:
                # print('data/{}/{}'.format(i,self.file_name[0]))
                rawLabel = flieLabel.readlines()    #get all data in label txt
                flieLabel.close()
                #get the information about skeleton
                rawSkeleton = np.loadtxt('{}/{}/{}'.format(self.path,i,self.file_name[1])) 
                for num in range(len(rawLabel)):
                    rawLabel[num] = re.split(r',|\n',rawLabel[num]) #delete the ',' '\n' in strings about labels
                for k in range(len(rawLabel)):
                    num = int( rawLabel[k][0] )    #get label number
                    start = int( rawLabel[k][1] )
                    end = int( rawLabel[k][2] )
                    if not os.path.exists('{}/data2/{}'.format(self.path,num)):  #make sure there is this file
                        os.makedirs('data/data2/{}'.format(num))
                    np.savetxt('{}/data2/{}/{}.txt'.format(self.path,num,i),rawSkeleton[start:end,:75],fmt='%0.8f')

if __name__ == "__main__":
    data_read().run()
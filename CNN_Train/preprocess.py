import numpy as np

class preprocessing():
    def __init__(self,joints=25,time=200,pos=np.array([])):
        self.joints = joints
        self.pos = np.reshape(pos,[-1,25,3])
        self.time = time

    def normalize(self):
        data_norm = np.array([],dtype=float)
        data = self.local_pos

        joint_data = [[3,2],[2,20],[20,1],[1,0],\
            [0,12],[12,13],[13,14],[14,15],\
                [0,16],[16,17],[17,18],[18,19],\
                    [20,4],[4,5],[5,6],[6,7],\
                        [20,8],[8,9],[9,10],[10,11]]
        for joint in joint_data:
            temp = data[joint[0],:]-data[joint[1],:]
            data_temp = temp / np.linalg.norm(temp)
            data_norm = np.append(data_norm,data_temp)

        data_norm = np.reshape(data_norm,[-1,len(joint_data),3])
        return data_norm

    def deal_joints(self):
        norm_array = np.array([])
        for pos in self.pos:
            self.local_pos = pos
            gate = np.zeros(self.local_pos.shape)
            if self.local_pos.all() == gate.all():
                norm = np.zeros([20,3])
            else:
                norm = self.normalize()
            norm_array = np.append(norm_array,norm)
        norm_array = np.reshape(norm_array,[-1,20,3])
        return norm_array

    def run(self):
        '''
        output: skeleton, motion
        '''
        norm_array = self.deal_joints()
        motion = norm_array[1:] - norm_array[:-1]
        skeleton = norm_array[1:]
        return skeleton,motion

if __name__ == "__main__":
    pos = np.ones([1,200,75])
    pos[0,:,:]=0
    # print(pos)
    pre = preprocessing(pos=pos)
    print(pre.run()[1].shape)
    print(pre.run()[0].shape)
    # print(pre.run().all()==0)
import numpy as np

class preprocessing():
    def __init__(self,pos=np.array([])):
        self.pos = np.reshape(pos,[-1,25,3])

    def transform(self):
        o = self.local_pos[0]
        self.local_pos = self.local_pos - o

    def rotate(self,origin_joint):
        origin_z = self.local_pos[origin_joint[0],:] - self.local_pos[origin_joint[1],:]
        origin_x = self.local_pos[origin_joint[2],:] - self.local_pos[origin_joint[3],:]
        origin_y = np.cross(origin_z,origin_x)
        rotate_mat = np.zeros([3,3])
        rotate_mat[0] = origin_x
        rotate_mat[1] = origin_y
        rotate_mat[2] = origin_z
        rotate_mat = np.array(rotate_mat)
        rotate_mat = np.linalg.inv(rotate_mat)
        self.local_pos = np.matmul(self.local_pos,rotate_mat)

    def normalize(self):
        data_norm = np.array([],dtype=float)
        data = self.local_pos

        # joint_data = [[3,2],[2,20],[20,1],[1,0],\
        #     [0,12],[12,13],[13,14],[14,15],\
        #         [0,16],[16,17],[17,18],[18,19],\
        #             [20,4],[4,5],[5,6],[6,7],\
        #                 [20,8],[8,9],[9,10],[10,11]]
        joint_data = [[2,20],[20,12],[20,16],\
            [12,16],[12,13],[13,14],\
                [16,17],[17,18],\
                    [20,4],[4,5],[5,6],\
                        [20,8],[8,9],[9,10]]
        for joint in joint_data:
            temp = data[joint[0],:]-data[joint[1],:]
            data_temp = temp / np.linalg.norm(temp)
            data_norm = np.append(data_norm,data_temp)

        data_norm = np.reshape(data_norm,[-1,len(joint_data),3])
        return data_norm

    def run(self):
        norm_array = np.array([])
        for pos in self.pos:
            self.local_pos = pos
            gate = np.zeros(self.local_pos.shape)
            if self.local_pos.all() == gate.all():
                # norm = np.zeros([20,3])
                norm = np.zeros([14,3])
            else:
                self.transform()
                self.rotate([1,0,8,4])
                norm = self.normalize()
            norm_array = np.append(norm_array,norm)
        # norm_array = np.reshape(norm_array,[30,20,3])
        norm_array = np.reshape(norm_array,[30,14,3])
        skeleton = norm_array[1:]
        motion = norm_array[1:] - norm_array[:-1]
        return skeleton,motion

if __name__ == "__main__":
    pos = np.ones([1,200,75])
    pos[0,:,:]=0
    # print(pos)
    pre = preprocessing(pos=pos)
    print(pre.run()[1].shape)
    print(pre.run()[0].shape)
    # print(pre.run().all()==0)
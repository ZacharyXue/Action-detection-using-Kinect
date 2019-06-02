import sys, os

# CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir+ "/githubs/tf-pose-estimation")  
# sys.path.apspend(CURR_PATH + "githubs/tf-pose-estimation")
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.estimator import TfPoseEstimator
from tf_pose import common

# Openpose Human pose detection ==============================================================

class SkeletonDetector(object):
    # This func is copied from https://github.com/ildoonet/tf-pose-estimation
    def __init__(self, model=None, image_size=None):
        
        if model is None:
            model = "cmu"

        if image_size is None:
            image_size = "640x480" 

        models = set({"mobilenet_thin", "cmu"})
        self.model = model if model in models else "mobilenet_thin"
        self.resize_out_ratio = 4.0

        w, h = model_wh(image_size)
        if w == 0 or h == 0:
            e = TfPoseEstimator(get_graph_path(self.model),
                                target_size=(432, 368))
        else:
            e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        self.w, self.h = w, h
        self.e = e

        self.cnt_image = 0

    def detect(self, image):
        self.cnt_image += 1
        if self.cnt_image == 1:
            self.image_h = image.shape[0]
            self.image_w = image.shape[1]
            self.scale_y = 1.0 * self.image_h / self.image_w

        # Inference
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0),
                                #   upsample_size=self.args.resize_out_ratio)
                                  upsample_size=self.resize_out_ratio)


        return humans
    
    def draw(self, img_disp, humans):
        img_disp = TfPoseEstimator.draw_humans(img_disp, humans, imgcopy=False)

    def humans_to_skelsList(self, humans, scale_y = None): # get (x, y * scale_y)
        # type: humans: returned from self.detect()
        # rtype: list[list[]]
        if scale_y is None:
            scale_y = self.scale_y
        skeletons = []
        NaN = 0
        for human in humans:
            skeleton = [NaN]*(18*2)
            for i, body_part in human.body_parts.items(): # iterate dict
                idx = body_part.part_idx
                skeleton[2*idx]=body_part.x
                skeleton[2*idx+1]=body_part.y * scale_y
            skeletons.append(skeleton)
        return skeletons, scale_y

# ==============================================================
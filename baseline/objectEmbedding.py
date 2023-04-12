##########################
###  Visual Extractor  ###
##########################

# PyTorch related imports
import torch
import torch.nn as nn
import torchvision

# Baseline configs import
from baseline.configs import configs

# Pre-trained resnet import
import helpers.mytorch.vision.models.resnet as myresnet


# Helper module used in ObjectEmbedding module
class Identity(nn.Module):
    def forward(self, *inputs):
        if len(inputs) == 1:
            return inputs[0]
        return inputs


# Module for object feature extraction
class ObjectEmbedding(nn.Module):
    
    # Arguments:
    #   image_dim: (list(int))               : Dimension of the input image ([256, 384]) 
    #   visual_feature_dim: (int)            : Size of object embedding (256)
    #   object_feature_bbox_size: (list(int)): Object bounding box dimension used during RoI alignment ([4, 4])
    #   resnet_pretrained: (bool)            : Pre-trained resnet usage switch (True)
    def __init__(self, image_dim, visual_feature_dim, object_feature_bbox_size, resnet_pretrained=True):
        super().__init__()
        self.resnet = myresnet.resnet34(pretrained=resnet_pretrained, num_classes=None)
        self.resnet.layer4 = Identity()
        self.resnet_num_layers = 3

        self.roi_align = torchvision.ops.roi_align
        self.object_feature_bbox_size = object_feature_bbox_size
        self.visual_feature_dim = visual_feature_dim

        self.extract_context_features = nn.Conv2d(2**(self.resnet_num_layers+5), 2**(self.resnet_num_layers+5), 1)
        self.object_feature_fc = nn.Linear(2**(self.resnet_num_layers+5)*(object_feature_bbox_size[0]*object_feature_bbox_size[1]), self.visual_feature_dim) 

    # Arguments:
    #   image: (tensor(B X 3 X 256 X 384)): Batch of input images
    #   bboxes: (list(tensor(N X 4)))     : Batch of bounding boxes of objects in image (xywh format)
    #   object_lengths: (tensor(B,))      : Batch of count of objects in a given image
    # Return Value:
    #   object_features: list(tensor(N X 256)): Batch of objects embedding for each image
    # Here, B = batch size and N = number of objects in a given image
    def forward(self, image, bboxes, object_lengths):
        if configs.model.fixed_resnet:
            with torch.no_grad():
                f_image = self.resnet(image)
        else:
            f_image = self.resnet(image)
        f_context = self.extract_context_features(f_image)
        f_object = self.roi_align(f_context, bboxes, output_size = self.object_feature_bbox_size, spatial_scale = 1/2**(self.resnet_num_layers+1))
        object_features, start_index = [], 0
        for i in range(image.size(0)):
            f_obj_curr_scene = f_object[start_index:start_index+object_lengths[i]]
            start_index += object_lengths[i].item()
            this_obj_feature = self.object_feature_fc(f_obj_curr_scene.view(object_lengths[i], -1))
            object_features.append(this_obj_feature)
        return object_features

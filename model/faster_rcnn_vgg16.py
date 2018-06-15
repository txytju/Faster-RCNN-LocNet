import torch as t
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3

    # use either caffe or pytorch pretrained model
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)  # use pretrained torchvision vgg net

    features = list(model.features)[:30]
    
    # get the classification layer and drop some of them, leave the rest for use

    # classifier defined in pytorch source code.
    #     self.classifier = nn.Sequential(
    # 0    nn.Linear(512 * 7 * 7, 4096),
    # 1    nn.ReLU(True),
    # 2    nn.Dropout(),
    # 3    nn.Linear(4096, 4096),
    # 4    nn.ReLU(True),
    # 5    nn.Dropout(),
    # 6    nn.Linear(4096, num_classes),
    # )
    # only two linear and two ReLU layers are kept for classifier.

    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """


    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        # extractor is for base net of faster rcnn and classifier is for the final ROIHead part.
        # These are just some layers, not values.         
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            M=28,
            classifier=classifier
        )   

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale, M,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.M = M


        # branch_1
        self.roi_1 = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)     # roi shape of (N, C, outh, outw)
        self.classifier = classifier
        self.score = nn.Linear(4096, n_class)
  
        # branch_2
        self.roi_2 = RoIPooling2D(self.roi_size*2, self.roi_size*2, self.spatial_scale) # roi shape of (N, C, outh*2, outw*2)
        self.conv_21 = nn.ConV2d(512, 512, (3,3))
        self.conv_22 = nn.ConV2d(512, 512, (3,3))  # output shape (1, 512, 14, 14)
        self.max_x = nn.MaxPool2d((14,1))         # output shape (1, 512, 1, 14)
        self.max_y = nn.MaxPool2d((1,14))         # output shape (1, 512, 14, 1)
        self.fc_x = nn.Linear(7168, M)
        self.fc_y = nn.Linear(7168, M)


        normal_init(self.score, 0, 0.01)
        normal_init(self.conv_21, 0, 0.001)
        normal_init(self.conv_22, 0, 0.001)
        normal_init(self.fc_x, 0, 0.001)
        normal_init(self.fc_y, 0, 0.001)


    def forward(self, x, rois, seach_regions, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable. (batch_size, channels, width, height)
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        
        roi_indices = at.totensor(roi_indices).float()
        
        rois = at.totensor(rois).float()        
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous()) # [index, x1, y1, x2, y2] now

        seach_regions = at.totensor(seach_regions).float()  
        indices_and_search_regions = t.cat([roi_indices[:, None], seach_regions], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_search_regions = indices_and_search_regions[:, [0, 2, 1, 4, 3]]
        indices_and_search_regions = t.autograd.Variable(xy_indices_and_search_regions.contiguous()) # [index, x1, y1, x2, y2] now

        # branch_1
        pool_1 = self.roi_1(x, indices_and_rois) # get all the ROI pooling, shape of (N, C, outh, outw)
        pool_1 = pool.view(pool_1.size(0), -1)   # shape of shape of (N, C * outh * outw) where C=512    
        fc7 = self.classifier(pool_1)
        roi_scores = self.score(fc7)

        # branch_2
        pool_2 = self.roi_2(x, indices_and_search_regions)
        conv_1 = self.conv21(pool_2)
        conv_2 = self.conv22(conv_1)

        max_x_ = self.max_x(conv_2)
        max_y_ = self.max_y(conv_2)
        max_x_ = max_x_.view(max_x_.size(0), -1)
        max_y_ = max_y_.view(max_y_.size(0), -1)
        px = F.sigmoid(self.fc_x(max_x_))
        py = F.sigmoid(self.fc_y(max_y_))

        return (px, py), roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

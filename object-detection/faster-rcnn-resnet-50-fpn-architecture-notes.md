# The Torchvision Faster R-CNN ResNet50 FPN model {ignore=true}

This document describes the PyTorch Torchvision implementation of the Faster R-CNN ResNet50 FPN object detection neural network model. This model is referred to as being a Faster R-CNN model with a ResNet50-FPN backbone.

The description consists of the following components:
* a summary listing of the main source code packages, modules, classes and functions involved in the implementation of the model
* a visualisation of the main components and structure of the model
* a detailed description of the instantiation of the model
* a detailed description of the forward pass of the model, covering both training mode and test (inference/evaluation) mode contexts.

**Table of Contents**
[TOC]

# TODO

RoIHeads
* post-processing function; what does it do, how and why
* function select_training_samples() in the forward() method; what does it do, how and why

find and remove a certain textual comment
* at one point, somewhere I make a comment that RoI Heads component only needs two kinds of info from the images: the common HxW size and the number of images; this is wrong and should be removed or amended; I think it uses the individual size of each image, after transformation, but prior to zero-padding to the common size
* last time I looked I couldn't find this comment; so maybe it's already removed; just make sure it's removed

class BoxCoder in models.detection._utils.py
* we don't yet understand its role, what it does, how and why sufficiently
* review everywhere it's used, both for encode() & decode()
* explain its role fully and properly
* it may be the case that most references to bboxes and proposals in the model's internal variables are misleading and that, in fact, at all times the bboxes and proposals are 'bbox deltas'
* if the neuron activations are 'real numbers', then they can be any positive or negative real number; but bbox coords are strictly positive; so this suggests neurons can only learn bbox deltas, not proper bbox coords
* even the outputs of the bbox_regression output layer in RoIHeads may be bbox_deltas, rather than proper bboxes; see the postprocessing() function in RoIHeads for evidence of this
* if it turns out that supposed bboxes are in fact bbox-deltas, then we may also need to adjust what we say in the RPN section: How is it that RPN learns bbox deltas

model parameters 'rpn_batch_size_per_image' and 'box_batch_size_per_image'
* confirm what these are for and how they're used and how they affect things

loss functions
* confirm how the two RPN losses are calculated
* confirm how the two RoI losses are calculated



# Torchvision packages, modules, classes & functions for the Faster R-CNN ResNet50 FPN model

on my iMac, in my Anaconda installation, and within my conda 'ai' environment, the `torchvision` package lives at:
`/Users/dave/opt/anaconda3/envs/ai/lib/python3.8/site-packages/torchvision`

the main classes and functions involved in (used to construct) the Torchvision implementation of Faster R-CNN ResNet50-FPN model are distributed across the following packages, subpackages and modules:

```Python
torchvision                           # top-level pkg
  models                              # models sub-pkg
    _utils.py
      class IntermediateLayerGetter(nn.ModuleDict):
    resnet.py
      class Bottleneck(nn.Module):
      class ResNet(nn.Module):
      def resnet50():
    detection                      # detection sub-pkg
      _utils.py
        class BalancedPositiveNegativeSampler(object):
        class BoxCoder(object):
        class Matcher(object):
      anchor_utils.py
        class AnchorGenerator(nn.Module):
      backbone_utils.py
        class BackboneWithFPN(nn.Module):
        def resnet_fpn_backbone():
        def _validate_trainable_layers():
      faster_rcnn.py
        class FasterRCNN(GeneralizedRCNN):
        class TwoMLPHead(nn.Module):
        class FastRCNNPredictor(nn.Module):
        def fasterrcnn_resnet50_fpn():
      generalized_rcnn.py
        class GeneralizedRCNN(nn.Module):
      image_list.py
        class ImageList(object):
      roi_heads.py
        class RoIHeads(nn.Module):
      rpn.py
        class RPNHead(nn.Module):
        class RegionProposalNetwork(torch.nn.Module):
      transform.py
        class GeneralizedRCNNTransform(nn.Module):
        def resize_boxes():
  ops                                  # ops sub-pkg
    boxes.py
      def box_iou():
    feature_pyramid_network.py
      class ExtraFPNBlock(nn.Module):
      class FeaturePyramidNetwork(nn.Module):
      class LastLevelMaxPool(ExtraFPNBlock):
    poolers.py
      class LevelMapper(object):
      class MultiScaleRoIAlign(nn.Module):
    roi_align.py
      def roi_align():
```

# Components and structure of the Faster R-CNN ResNet50 FPN model

A *Faster R-CNN ResNet50 FPN* neural network model consists of multiple PyTorch `nn.Module` components (most of which are proper subnetworks) arranged sequentially.

The largest single component, called the `backbone`, is the subnetwork responsible for learning the spatial and scale invariant features that are ultimately fundamental to accurate  object detection. In our case, the `backbone` is a *ResNet-50 FPN* subnetwork. This subnetwork is itself a combination of 2 independent subnetwork models: 1) a *ResNet-50* network model, followed by 2) a *Feature Pyramid Network* (FPN) network model.

The outputs of the `backbone` subnetwork are then passed into a `rpn` (region proposal network) subnetwork, and the outputs of the `rpn` subnetwork are then passed into an `roi_heads` subnetwork which, ultimately (in test mode, rather than training mode), outputs  the predictions of the overall Faster R-CNN model for each input image: bounding boxes (for the detected objects), class labels (for the predicted bounding boxes) and confidence scores (for the class labels).

```Python
FasterRCNN(
  (transform): GeneralizedRCNNTransform(
    Normalize()
    Resize()
  )
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(  # ResNet50 layers in OrderedDict
      (conv1) (Conv2d, bn, relu, maxpool)
      (layer1) Sequential(3 ResNet 'Bottleneck' residual blocks)
      (layer2) Sequential(4 ResNet 'Bottleneck' residual blocks)
      (layer3) Sequential(6 ResNet 'Bottleneck' residual blocks)
      (layer4) Sequential(3 ResNet 'Bottleneck' residual blocks)
    )
    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(4 Conv2d layers)
      (layer_blocks): ModuleList(4 Conv2d layers)
      (extra_blocks): LastLevelMaxPool()
    )
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead()
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign()
    (box_head): TwoMLPHead()
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91)   # num_classes = 91
      (bbox_pred): Linear(in_features=1024, out_features=364)  # 364 = 91 * 4 (xmin,ymin,xmax,ymax)
    )
  )
)
```

# Instantiating a Faster R-CNN ResNet50 FPN model

## The `fasterrcnn_resnet50_fpn()` function drives everything

The function `torchvision.models.detection.fasterrcnn_resnet50_fpn()` drives the instantiation of the Faster RCNN ResNet50 FPN model. It's defined in module `models.detection.faster_rcnn.py`.

See PyTorch documentation page https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection, and refer to the section on *Faster R-CNN*, for information on calling this function.

Here is a simplified version of the function that captures its key elements:
```Python
def fasterrcnn_resnet50_fpn(pretrained, num_classes, **kwargs):
    backbone = resnet_fpn_backbone('resnet50', pretrained, trainable_layers=3)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model
```
First the resnet50-fpn backbone is instantiated. Then the rest of the Faster RCNN ResNet50 FPN model is instantiated around that backbone by the initialiser of class `FasterRCNN`. In the sections which follow, we examine the lower-level steps involved in these two macro instantiation steps.

The number of *trainable* (not frozen) resnet layers in the backbone defaults to 3. The number applies starting from the final layer of the resnet model and working backward. Per the structure of the ResNet50 model, above, it is considered to have 5 layers, consisting of (starting from the end and working backward): layer4, layer3, layer2, layer1, conv1.  So, since the default number of trainable layers is 3, this means that layer4, layer3 and layer2 will be trainable, whereas layer1 and conv1 will be set to `requires_grad=False`. All of this assumes we are using a pretrained model. So freezing conv1 and layer1 means that any training we do on the pretrained resnet model will apply only to the parameters of the last 3 layers: layer2, layer3, layer4.

There are **many** *keyword arguments* that can be passed to function `fasterrcnn_resnet50_fpn()` for onward passage when instantiating the `FasterRCNN` class. These keyword arguments allow one to customise the architecture, components and behaviours of a Faster R-CNN model.  We mention just a few of these keyword arguments here:
* `rpn_batch_size_per_image` - nr of anchors sampled during training of the RPN (re RPN loss)
* `rpn_anchor_generator` - a custom AnchorGenerator module for the RPN
* `box_score_thresh` - only return proposals with score > thresh (for inference only)
* `box_batch_size_per_image` - nr of proposals sampled during training of the RoI classification head
* `box_roi_pool` - a custom MultiScaleRoIAlign module for the RoI classification head


## Instantiating the ResNet50 FPN backbone subnetwork

The function `resnet_fpn_backbone()` defined in module `backbone_utils.py` drives the construction of a specified ResNet model (subnetwork) together with a Feature Pyramid Network  (`fpn`) subnetwork.  Together, the two components are referred to as the `backbone` subnetwork  of the Faster R-CNN model whose construction we are analysing.  During a *forward pass*, the outputs of the layers of the ResNet model are passed as inputs into the `forward()` method of the `fpn` subnetwork.

The function `resnet_fpn_backbone()` freezes a specified number of layers in the ResNet model so that no training occurs on these layers.

Here is a simplified version of the function that captures its key elements. The first two code blocks of the function construct the ResNet subnetwork. The remaining code blocks relate to the construction of the FPN subnetwork and the combining of the ResNet and FPN subnetworks into a single subnetwork called the `backbone` subnetwork of the Faster R-CNN network model.
```Python
from torchvision.models import resnet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection._utils import IntermediateLayerGetter

def resnet_fpn_backbone(backbone_name, pretrained, trainable_layers=3):
    # construct resnet portion of the backbone
    backbone = resnet.resnet50(backbone_name, pretrained)   # backbone is an nn.Module

    # freeze all parameters associated with layers that are NOT trainable
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    # initialise components and parameters to construct an FPN
    extra_blocks = LastLevelMaxPool()  # an nn.Module
    returned_layers = [1, 2, 3, 4]
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # nb: return_layers is {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_stage2 = backbone.inplanes // 8  # 2048 // 8 = 256
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # nb: in_channels_list is [256, 512, 1024, 2048]
    out_channels = 256 # nr of feature maps (channels) output by all levels of the FPN

    # construct an Feature Pyramid Network (FPN) model (subnetwork) and have it sequentially
    # follow the ResNet-50 model so that, in the forward pass of the overall model, the
    # correct outputs from the ResNet-50 model (subnetwork) are passed as inputs into the
    # FPN model (subnetwork); together, these two subnetworks will constitute the 'backbone'
    # of the Faster R-CNN model
    model = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels,
                            extra_blocks=extra_blocks)  # model is an nn.Module

    return model
```

The class (nn.Module) `LastLevelMaxPool` is defined in module `torchvision.ops.feature_pyramid_network.py`. It's a component (final/extra block) of a Feature Pyramid Network (FPN) model and is needed to construct the FPN model (subnetwork). It's the last block (layer) of the FPN.  It has no `__init__()` method and maintains no state (ie no instance attributes).  It has a tiny `forward()` method that simply applies a *max_pool2d* operation.

### Instantiating the `BackboneWithFPN` module `backbone`

The class `BackboneWithFPN` is also defined in module `backbone_utils.py`, along with function `resnet_fpn_backbone()`.

Here is the `__init__()` method of class `BackboneWithFPN`:
```Python
class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list,
                 out_channels, extra_blocks):
        self.body = IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels, extra_blocks)
```

### Instantiating the `IntermediateLayerGetter` module `body`

The class `IntermediateLayerGetter` instantiates an object that maintains an Ordered Dictionary containing all of the layers of the resnet-50 model except for its final fully connected (fc) layer.  Being an ordered dictionary, it maintains the order in which elements are added to the dictionary and hence it recognises the proper order of the layers of the resnet-50 model. The class inherits from `nn.ModuleDict`, so it instantiates an nn.Module object which has a `forward()` method.

Here is the `__init__()` method of class `IntermediateLayerGetter` which populates an OrderedDict with the desired layers of the ResNet-50 model.
```Python
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        """
        Args:
          model - a ResNet-50 model from which specified layers are to be
                  extracted
          return_layers - Dict[str, str] - the layers of the ResNet-50 model
                  whose outputs we wish to capture and return within the
                  forward() method of this module
                  {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        """
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # extract the layers of the ResNet-50 model and store them in an
        # OrderedDict so that we can later step thru them, one by one, in
        # the forward() method of this module; but don't extract the final
        # fully-connected (fc) layer that does classification; we won't be
        # needing that one
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        # save the OrderedDict containing the layers of the ResNet-50 model
        # in attribute self.items of the parent nn.ModuleDict class for use
        # in the forward() method
        super(IntermediateLayerGetter, self).__init__(layers)

        # save the names of the ResNet-50 layers whose outputs we will want to
        # capture during the forward() method, and return from the forward()
        # method; {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        # note: the layer names are keys in a dictionary because the names
        # will later be swapped to the values of these keys
        self.return_layers = orig_return_layers
```
Note that when the final layer to be extracted has in fact been extracted, the `return_layers` variable is empty and the `break` statement is executed which escapes the `for` loop and prevents extraction of any further layers. Thus, the final fully connected (fc) layer of the ResNet-50 model is NOT extracted. So the final layer in our 'version' of the ResNet-50 model is the 'layer4' convolution layer whose outputs will, in our Faster R-CNN model, be fed as inputs into the FPN (feature pyramid network) component of our `backbone`.  


### Instantiating the `FeaturePyramidNetwork` module `fpn`

The class `FeaturePyramidNetwork` from module `feature_pyramid_network.py` in the `torchvision.ops` package instantiates the `fpn` subnetwork (component) of the `backbone` subnetwork of the Faster R-CNN model.

Here is a simplified version of the `__init__()` method of class `FeaturePyramidNetwork`:
```Python
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks):
      """
      Args:
        in_channels_list - [256, 512, 1024, 2048]
        out_channels - 256
        extra_blocks - LastLevelMaxPool() (a max_pool2d operation only)
      """
      self.inner_blocks = nn.ModuleList()
      self.layer_blocks = nn.ModuleList()
      for in_channels in in_channels_list:
          inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
          layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
          self.inner_blocks.append(inner_block_module)
          self.layer_blocks.append(layer_block_module)
      self.extra_blocks = extra_blocks
```

We end up with:
```Python
(inner_blocks): ModuleList(
  (0): Conv2d(256, 256, kernel_size=(1,1), stride=(1,1))
  (1): Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
  (2): Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1))
  (3): Conv2d(2048, 256, kernel_size=(1,1), stride=(1,1))
)
```
and
```Python
(layer_blocks): ModuleList(
  (0): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (1): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (2): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (3): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
)
```
Note that the configurations of the Conv2d layers in both `inner_blocks` and `layer_blocks` (with the specified kernel sizes, strides and padding) represents `same` convolution, meaning that the HxW size of the output feature maps will be the *same* as the HxW size of the input feature maps.

To get `same` convolution: if stride=1, then, to get `same` convolution we need the following relationship to hold: $p = \frac{k-1}{2}$. So, if $p=0$ (as in the `inner_blocks`), we need $k=1$; if $p=1$ (as in the `layer_blocks`), we need $k=3$.


## Instantiating the overall FasterRCNN model

The `FasterRCNN` class that's called within the `fasterrcnn_resnet50_fpn()` driver function is defined in module `faster_rcnn.py`. Given a backbone network (which, in our case, is a ResNet-50-FPN backbone), it drives the construction of a Faster R-CNN model around the backbone.  It only has an `__init__()` method, which drives construction of the other required components (subnetworks) of a Faster R-CNN model: 1) a region proposal network (RPN), 2) a region of interest network (RoIHeads), and 3) a preliminary model input (image) transformation network (RCNN Transform).

The `FasterRCNN` class inherits from class `GeneralizedRCNN`. Having driven the construction of the 3 remaining required components (subnetworks) of a Faster R-CNN model (as just described), at the end of its `__init__()` method it calls the `__init__()` method of class `GeneralizedRCNN`, passing it the 4 components (all of which are `nn.Module` instances) for a full Faster R-CNN model: *transform*, *backbone*, *rpn*, *roi_heads*.  The `__init__()` method of class `GeneralizedRCNN` simply stores these 4 components as object attributes. That concludes the construction of a full Faster R-CNN model.

Here is a simplified version of the FasterRCNN `__init__()` method:
```Python
class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone, num_classes,
        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None, box_head=None, box_predictor=None,
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None):

        #
        # construct RPN subnetwork
        #

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                    out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
                                 testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                                  testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
                                    rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                                    rpn_batch_size_per_image,
                                    rpn_positive_fraction,
                                    rpn_pre_nms_top_n, rpn_post_nms_top_n,
                                    rpn_nms_thresh,
                                    score_thresh=rpn_score_thresh)

        #
        # construct RoI Heads subnetwork
        #

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                              output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2,representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size,num_classes)

        roi_heads = RoIHeads(box_roi_pool, box_head, box_predictor, ...)

        #
        # construct transform component
        #

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        #
        # assemble the overall Faster R-CNN network model
        #

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)  
```
Note that the only method defined in class (subclass) `FasterRCNN` is the `__init__()` method above.  The `forward()` method for Faster R-CNN resides in the parent class `GeneralizedRCNN`.

Since class `FasterRCNN` inherits from class `GeneralizedRCNN`, function `super().__init__()` calls the `__init__()` method of class `GeneralizedRCNN`, which is shown here. It simply stores the 4 large components of the Faster R-CNN model in instance attributes.
```Python
class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
```

### Instantiating the `GeneralizedRCNNTransform` module `transform`

The *transform* module is the first component in an overall Faster R-CNN network model. It performs transformations (preprocessing) on the *images* input to the Faster R-CNN network model (during both training and test mode), and to the *target* ground-truth bounding boxes (during training).  Note that while the `transform` component inherits from `nn.Module`, and hence has a `forward()` method, we should not think of it as being a subnetwork because it is NOT a neural network.  It has no layers of neurons, or activation functions. It simply takes in images and targets as input and outputs transformed images and targets which feed into the next component of the Faster R-CNN model (the `backbone` subnetwork).

Constructing the *transform* module is simple: the `__init__()` method simply stores the arguments as instance attributes for later use with the `forward()` method.  

Here is a (very slightly) simplied version of the `__init__()` method:
```Python
class GeneralizedRCNNTransform(nn.Module):
    def __init__(self, min_size, max_size, image_mean, image_std):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size       # min image size (default: (800,))
        self.max_size = max_size       # max image size (default: 1333)
        self.image_mean = image_mean   # RGB channel means
        self.image_std = image_std     # RGB channel standard deviations
```

### Instantiating the `RegionProposalNetwork` module `rpn`

As we saw in the `__init__()` method of class `FasterRCNN`, above, constructing an RPN (region proposal network) is a 3-step process: 1) first we construct an `AnchorGenerator`, 2) then we construct an `RPNHead`, and 3) finally we construct a `RegionProposalNetwork`, passing it the `rpn_anchor_generator` and `rpn_head` components that it needs.  We discuss each of these steps in turn.

#### Instantiating the `AnchorGenerator` module `anchor_generator`

Here is a simplified `__init__()` method of class `AnchorGenerator`:
```Python
class AnchorGenerator(nn.Module):
    def __init__(self, sizes, aspect_ratios):
        """
        Args:
          sizes - the sizes (areas) for the anchor boxes to be generated
                  # ( (32,), (64,), (128,), (256,), (512,) )
          aspect_ratios - the aspect ratios for the anchor boxes to be generated
          # ( (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0),
              (0.5, 1.0, 2.0), (0.5, 1.0, 2.0) )
        """
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}
```

#### Instantiating the `RPNHead` module `head`

Here is a simplified `__init__()` method of class `RPNHead`:
```Python
class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
    """
    Args:
      in_channels (int) - number of channels of the input
      num_anchors (int) - number of anchors to be predicted
    """
    # the 3x3 convolution layer of the RPN head
    self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    # the first of the two sibling 1x1 convolutions, this one for the
    # object/non-object binary classification
    self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

    # the 2nd of the two sibling 1x1 convolutions, this one for the
    # bounding box regression
    self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

    # initialise the parameters of the 3 layers of the RPN Head we just defined
    # (the children() method of class nn.Module returns an iterator over
    #  immediate children modules)
    for layer in self.children():
        torch.nn.init.normal_(layer.weight, std=0.01)
        torch.nn.init.constant_(layer.bias, 0)
```

#### Instantiating the overall `RegionProposalNetwork` module `rpn`

Here is the `__init__()` method of class `RegionProposalNetwork`, with only very minor omissions and some comments added:
```Python
class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):

        self.anchor_generator = anchor_generator
        self.head = head

        # A BoxCoder encodes and decodes a set of bounding boxes to/from
        # the representation used for training the box regressors.
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        #
        # tools used during training
        #

        # box_iou() is a function that calculates the intersection-over-union
        # (IoU) for all pairs of boxes in two sets of boxes: boxes1 (size M)
        # and boxes2 (size N); it returns an MxN matrix, often called a
        # match_quality_matrix, where each cell holds the IoU for the
        # corresponding pair of boxes; all boxes in boxes1 and boxes2 are
        # expected to have format [xmin, ymin, xmax, ymax]
        self.box_similarity = box_ops.box_iou

        # A Matcher assigns to each predicted box the best-matching ground-truth
        # box (if there is one).
        # Each predicted box will end up having exactly one or zero matches.
        # Each ground-truth box may be assigned to zero or more predicted boxes.
        # Matching is based on an MxN match_quality_matrix (see above).
        # There are M ground-truth boxes and N predicted boxes (where M < N).
        # For a given image, a Matcher returns a tensor of size N
        # containing the index (a positive integer) of the ground-truth box (m)
        # that matches to predicted box (n) (if there is a match).
        # If there is no match for predicted box (n), a negative value is
        # returned.
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        # A BalancedPositiveNegativeSampler randomly samples 'batches' of
        # predicted boxes that have been labelled as being either 'positive'
        # or 'negative', whilst respecting a specified proportion of
        # positives. It samples a batch for each image from amongst the
        # labelled predicted boxes for that image.
        # For each image, it creates two binary masks, one indicating the
        # positive predicted boxes that were sampled for that image, the other
        # indicating the negative predicted boxes that were sampled for that
        # image. It packages these binary masks into two lists, one for
        # positive box masks for all images, one for negative box masks.
        # It returns these two lists of binary masks.
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        #
        # parameters used during testing
        #

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3
```

### Instantiating the `RoIHeads` module `roi_heads`

The RoI (region of interest) subnetwork (called `roi_heads`) is the final component of the Faster R-CNN ResNet50-FPN model. It produces the predictions (the predicted bounding boxes for detected objects, the object class predictions for these bboxes, and the confidence scores of the object class predictions) output by the model during inference.

The RoI subnetwork (and hence the Faster R-CNN ResNet50-FPN model as a whole) has dual output layers: 1) the layer that produces the bbox predictions performs 'regression', whereas 2) the layer that produces the object class predictions performs 'classification'.

As we saw in the `__init__()` method of class `FasterRCNN`, above, an RoI (region of interest) subnetwork has 3 components, so constructing an RoI subnetwork is a 4-step process:
1. first we construct a `box_roi_pool` component using a `MultiScaleRoIAlign` module
2. then we construct a `box_head` component using a `TwoMLPHead` module
3. then we construct a `box_predictor` component using a `FastRCNNPredictor` module, and
4. finally, we construct the overall RoI subnetwork by instantiating an `RoIHeads` module which assembles the 3 components (`box_roi_pool`, `box_head` and `box_predictor`).

We discuss each of these steps in turn.

#### Instantiating the `MultiScaleRoIAlign` module `box_roi_pool`

Here is the `__init__()` method of class `MultiScaleRoIAlign`
```Python
class MultiScaleRoIAlign(nn.Module):
    def __init__(
        self,
        featmap_names,   # ['0', '1', '2', '3']
        output_size,     # 7
        sampling_ratio,  # 2
        *,
        canonical_scale=224,
        canonical_level=4
    ):
        super(MultiScaleRoIAlign, self).__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)   # 7x7
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
```
Objects of this class perform three tasks: scaling, alignment and pooling. Multi-scale RoI alignment involves both scaling (re-sizing) region proposals and the assignment of the scaled (re-sized) region proposals to the appropriate (best-matching) layer of the FPN (feature pyradmid network). Pooling involves max-pooling: the scaled (re-sized) region proposals are then used to perform max-pooling against the feature map volume of the FPN with which they have been assigned (aligned). As with all max-pooling operations, the output is a feature map volume with the same number of channels as the input feature map volume, but where the output feature maps are of smaller dimension. In this case, the output feature maps are all of a fixed size: 7x7.  See the `forward()` method for more information multi-scale RoI alignment and pooling.

#### Instantiating the `TwoMLPHead` module `box_head`

Here is the `__init__()` method of class `TwoMLPHead`
```Python
class TwoMLPHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
```
As we can see, this component is simple: two fully-connected linear layers of the same size.  The role of the TwoMLPHead component is to take as input the 7x7 feature map volumes produced by the multi-scale alignment and pooling component and learn flat, intermediate representations of these features that are appropriate for feeding the final regression/classification prediction (output) layers of the overall model.

#### Instantiating the `FastRCNNPredictor` module `box_predictor`

Here is the `__init__()` method of class `FastRCNNPredictor`
```Python
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
```
The two layers initialised here are the standard, dual output layers of a Fast R-CNN model.  One layer performs object class classification, the other performs bounding box regression.

#### Instantiating the overall `RoIHeads` module `roi_heads`

Here is a simplified version of the `__init__()` method of class `RoIHeads`
```Python
class RoIHeads(nn.Module):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img
                 ):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
```
The first 3 parameters to the initialisation method are the 3 components of the RoI subnetwork: the `MultiScaleRoIAlign` module, the `TwoMLPHead` module and the `FastRCNNPredictor` module.


# The forward pass of a Faster R-CNN ResNet50 FPN model

## The forward pass of the `FasterRCNN` module drives the process

As mentioned earlier, class `FasterRCNN` does not have its own `forward()` method; instead, it inherits the `forward()` method of its parent, class `GeneralizedRCNN`.

The `forward()` method of class `GeneralizedRCNN` contains the code that drives the overall *forward pass* of a full Faster R-CNN model.

### Model inputs and outputs

The inputs to the *forward pass*, the operations performed by the *forward pass*, and the outputs of the *forward pass* of a Faster R-CNN model vary significantly between *training* and *inference* (test/evaluation) modes.

**Training mode**
Inputs:
* images
  - a list of 3D images (CxHxW = 3xHxW) with pixel values in [0,1]
  - the images can be of different sizes; they do not have to be resized to a standard size prior to input to the Faster R-CNN model
* targets
  - a list of dictionarys, one per image
  - each dictionary defines the ground-truth bounding boxes
    (tensor of shape (N,4)) and the ground-truth class labels of each bounding boxes

Outputs:
* losses (2 RPN proposal losses, 2 RoI detection losses)

**Test mode**
Inputs:
* images
  - same as for training mode

Outputs:
* detections (predictions): for each image, a set of bounding boxes (one for each detected object), object class labels, and object class prediction confidence scores

### The FP of the `GeneralizedRCNN` module drives everything

Here is a simplified (and heavily commented) version of the `GeneralizedRCNN` `forward()` method, which is the overall *master* `forward()` method for the overall Faster R-CNN model:
```Python
class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Args:
          images - a list of tensors; the original input images to be processed,
                   of shape CxHxW, where each image can have a different size
                   (ie H and W can vary per image); the image pixel values need
                   to be values in [0,1]
          targets - a list of dictionaries; each dictionary has two keys:
                    1) a key 'boxes' whose value is a tensor of shape (N,4)
                    containing the ground-truth bounding boxes of objects
                    detected in the image; 2) a key 'labels' whose value is a
                    tensor of shape (N,) containing the class labels
                    corresponding to the bounding boxes   
        """
        # capture the original sizes of the input images
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the images and targets using the 'transform' component
        # - see the 'forward pass' of 'transform' for details of the processing
        # - the transformed images are returned in an ImageList object
        # - the transformed targets are still packaged as a list of dictionaries
        images, targets = self.transform(images, targets)

        # run the transformed images through the 'backbone' subnetwork
        # (ie through the resnet-50-fpn subnetwork)
        # - see the 'forward pass' of 'backbone' for details of the processing
        # - the images are stored in the 'tensors' attribute of the ImageList
        #   object returned by the 'transform' component; they are now of
        #   identical size and occupy a common tensor
        # - returns: an OrderedDict of pyramid feature maps
        #   constructed by the `fpn` (feature pyramid network)
        #   component of the 'backbone', based on feature maps from the  
        #   successive layers of the ResNet-50 model, which were carefully
        #   captured from the ResNet-50 model, during its forward pass, for
        #   the purpose of having the 'fpn' construct the pyramid
        #   feature maps from them
        features = self.backbone(images.tensors)
        # - the OrderedDict in 'features' looks like this:
        #   {'0':P2, '1':P3, '2':P4, '3':P5, 'pool':P6},
        #   where the lower levels of the pyramid are at the start and the
        #   top level of the pyramid at the end; the dimensions of the
        #   pyramid feature maps are widest at the bottom and smallest at the
        #   top (hence the pyramid metaphor)
        # - arranged as a pyramid, the shapes of the pyramid feature maps are:
        #     pool P6 CxHxW (256, Hp, Wp)  
        #        3 P5 CxHxW (256, H3, W3)
        #        2 P4 CxHxW (256, H2, W2)
        #        1 P3 CxHxW (256, H1, W1)
        #        0 P2 CxHxW (256, H0, W0)
        # - the heights and widths are widest at the bottom of the pyramid
        #   and get progressively smaller as we go up the pyramid levels;
        #   the HxW shapes of the levels of the pyramid will vary across
        #   successive mini-batches as the original sizes of the images in
        #   each mini-batch vary  
        # - notice that the number of channels used in the pyramid feature
        #   maps at each level of the pyramid is standardised at 256

        # run the features output by the 'backbone' subnetwork through
        # the RPN subnetwork together with the transformed images and
        # transformed targets returned by the 'transform' component
        # - see the 'forward pass' of 'rpn' for details of the processing
        # - in training mode, proposals and RPN losses are returned
        #     - two RPN losses are returned in a dictionary with 2 keys:
        #       'loss_objectness', 'loss_rpn_box_reg'
        # - in test mode, only proposals are returned
        # - the variable 'proposals' contains a list of predicted bboxes,
        #   one set of predicted bboxes for each image in the current
        #   mini-batch; by default, in inference mode, there are 1000
        #   proposals (predicted boxes) per image; in training mode there
        #   are 2000 per image
        proposals, proposal_losses = self.rpn(images, features, targets)

        # run the 'features' output by the 'backbone' (ie the feature pyramid
        # output by the feature pyramid network component of the backbone)
        # and the 'proposals' (predicted bboxes) output by the RPN
        # through the 'roi_heads' subnetwork, together with the 'targets'
        # and 'image sizes'
        # - see the 'forward pass' of 'roi_heads' for details of the processing
        # - the 'image_sizes' attribute contains the size of each resized
        #   image, after transformation but prior to zero-padding to a common size   
        # - in training mode, only losses are returned, in a dictionary with
        #   2 keys: 'loss_classifier', 'loss_box_reg'
        # - in test mode, only detections are returned, in a list containing
        #   1 dictionary with 3 keys: 'boxes', 'labels', 'scores'
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes, targets)

        # resize the predicted bboxes so they correspond to the original image
        # sizes
        # - this requires the sizes of the original images and the
        #   sizes of the resized images (prior to zero-padding) in order to
        #   calculate the correct ratios for performing the resizing
        # - this resizing only happens in test mode; in training mode, there
        #   are no detections, so the call just returns immediately
        detections = self.transform.postprocess(detections, images.image_sizes,               
                                                original_image_sizes)

        # assemble the 2 RPN proposal losses and the 2 RoIHeads detection losses
        # into a single losses dictionary
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training
            return losses

        return detections
```

## The forward pass of the `GeneralizedRCNNTransform` module `transform`

The `transform` nn.Module preprocesses the images and targets input to the Faster R-CNN model prior to these inputs being fed into the chain of subnetwork components of the model.

Here is a simplified and modified (and heavily commented) version of the `forward()` method of the `transform` nn.Module, which is the first component of the overall Faster R-CNN model.  The code, and the comments in the code, explain the preprocessing that's performed.
```Python
class GeneralizedRCNNTransform(nn.Module):
    def forward(self, images, targets=None):
        """
        Args:
          images - the original input images; a list of tensors of shape CxHxW,
                   where each tensor can have a different size; ie H and W can
                   be different for each image, which is why they are
                   packaged in a list at this point, rather than a tensor
          targets - only required during training mode
                  - a list of dictionaries, one per image; each image has 1) a
                    key 'boxes' whose value is a list of ground-truth bounding
                    boxes, and 2) a key 'labels' whose value is a list of
                    ground-truth class labels corresponding to each of the
                    bounding boxes
        """

        # 1) normalise each input image (subtract the mean and divide by the  
        #    standard deviation, on a per-channel basis)
        # 2) resize each input image and resize the corresponding target
        #    (ground-truth) bounding boxes
        #    - the images are resized using bilinear interpolation according
        #      to a scaling_factor computed given the relationship between the
        #      original image dimensions and min/max config parameters  
        #    - the ground-truth bounding boxes are resized according to
        #      height and width ratios calculated from the original image
        #      size and the resized image size
        for i in range(len(images)):
            image = images[i]
            target_i = targets[i] if targets is not None else None
            image = self.normalize(image)
            image, target_i = self.resize(image, target_i)
            images[i] = image
            if targets is not None and target_i is not None:
                targets[i] = target_i

        # 1) capture the sizes of the resized images and package them into  
        #    a list of 2-tuples: [(H,W), (H,W), ...]
        # 2) 'batch' the resized images, meaning: a) find the largest height
        #    (H) and width (W) amongst the resized images; b) scale these
        #    sizes up to next nearest integer divisible by 32, giving
        #    H' and W'; c) create a tensor of zeros of shape (N,C,H',W'),
        #    where N = nr of images, C = nr of channels (3); d) copy each
        #    resized image into the tensor of zeros of shape (N,C,H',W');
        #    e) the end result is that the batched images are now all of
        #    identical size due to each image being padded with zeros along
        #    its bottom and right edges
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))
        # variable 'images' is now a tensor of size (N,C,H',W'); these are the  
        # images that will ultimately be input into the resnet-50-fpn
        # backbone of the Faster R-CNN model

        # store 1) the batched images (resized and zero-padded so they all have
        # identical size H'xW' so they can occupy a common tensor) and 2) the
        # companion list of image sizes (ie the sizes of the resized images   
        # BEFORE they were individually zero-padded along the bottom and right
        # edges to make them all the same size) together in an ImageList object
        image_list = ImageList(images, image_sizes_list)

        # return to the caller two things:
        # 1) an ImageList object with two attributes:
        #    - a tensor of resized, zero-padded images of identical size
        #      (N,C,H',W')
        #    - a list of image sizes (the individual sizes of the resized
        #      images prior to them all being zero-padded to give them
        #      identical sizes)
        # 2) the transformed targets
        #    - the targets are only needed during training mode, so in test
        #      mode none will have been passed in, so we'll return none
        #    - in training mode, the transformed targets will still be
        #      formatted as a list of dictionaries, one dictionary per image,
        #      containing a key 'boxes' whose value is a list of resized
        #      ground-truth bounding boxes, and a key 'labels' whose value
        #      is a list of integer class labels, one per ground-truth
        #      bounding box
        #
        # Observe: the resized ground-truth bounding boxes were resized BEFORE
        # the resized images were zero-padded to give them identical size; but
        # because the zero-padding is applied only along the bottom and right
        # edges of the images, the zero-padding will NOT invalidate any of
        # the resized ground-truth bounding boxes because they are all
        # constructed relative to the top-left corner of the image. So the
        # zero padding added to the resized images will always be beyond the
        # dimensions of any ground-truth bounding box.
        return image_list, targets
```

## The forward pass of the `BackboneWithFPN` module `backbone`

Here is the `forward()` method of class `BackboneWithFPN`, which is the nn.Module representing the `backbone` subnetwork of the Faster R-CNN model.
```Python
class BackboneWithFPN(nn.Module):
    def forward(self, x):
        """
        Args:
          x - the transformed images output by the 'transform' component of the
              model; the images now have identical size and occupy a single
              tensor of shape BxCxHxW
        """
        # call the 'forward()' method of the 'resnet-50' component of the
        # 'backbone' subnetwork; but attribute self.body() holds an object
        # of type IntermediateLayerGetter, an nn.ModuleDict, that holds
        # the layers of the ResNet-50 model in an OrderedDict
        # - see the 'forward pass' of 'resnet-50' for details of the processing        
        # - inputs (x): - the transformed images
        x = self.body(x)
        # the 'x' returned by self.body() is an OrderedDict containing the
        # convolutional feature maps output by layers
        # [layer1, layer2, layer3, layer4] of the resnet-50 subnetwork;
        # but the layer names (keys) in the OrderedDict are now
        # {'0':Tensor, '1':Tensor, '2':Tensor, '3':Tensor}

        # call the 'forward()' method of the 'fpn' (feature pyramid network)
        # component of the 'backbone' subnetwork
        # - see the 'forward pass' of 'fpn' for details of the processing
        # - returns: an OrderedDict containing the feature maps created by
        #            the feature pyramid network
        # - the 'x' returned by 'fpn' looks like this:
        #   {'0':P2, '1':P3, '2':P4, '3':P5, 'pool':P6}
        #   where the lower levels of the pyramid are at the start and the
        #   top level of the pyramid at the end; the dimensions of the
        #   pyramid feature maps are widest at the bottom and smallest at the
        #   top (hence the pyramid metaphor)
        # - arranged as a pyramid, the shapes of the feature maps are:
        #     pool P6 CxHxW (256, Hp, Wp)  
        #        3 P5 CxHxW (256, H3, W3)
        #        2 P4 CxHxW (256, H2, W2)
        #        1 P3 CxHxW (256, H1, W1)
        #        0 P2 CxHxW (256, H0, W0)
        x = self.fpn(x)

        # note: what is returned from the forward pass of the `backbone` is
        # just the feature pyramid feature maps; the original ResNet-50
        # feature maps were used to construct the pyramid feature maps and
        # their information is therefore contained within the pyramid feature
        # maps; but those original ResNet-50 feature maps are NOT themselves
        # returned by the `backbone`; the onward subnetworks of the Faster R-CNN
        # model will only have the pyramid feature maps to work with!
        # fascinating!!
        return x
```


### The FP of the `IntermediateLayerGetter` module `body`

Here we describe what happens within the `forward()` method of the `resnet-50` component of the `backbone` subnetwork.

Recall (from above) that the `resnet-50` component is stored in the `self.body` attribute of the `backbone`. The `self.body` attribute of the `backbone` contains on object of class `IntermediateLayerGetter`, which is a PyTorch `nn.ModuleDict` module that holds submodules in an ordered dictionary. In our case, it holds all of the submodules (layers) of the ResNet-50 model except for the final fully-connected (fc) layer that performs the final classification.

Here is the `forward()` method of `resnet-50` model:
```Python
class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(model, orig_return_layers):
        self.return_layers = orig_return_layers
               # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    def forward(self, x):
        """
        Args:
          x - the transformed images output by the 'transform' component of the
              model; the images now have identical size and occupy a single
              tensor of shape BxCxHxW
        """
        out = OrderedDict()

        # self.items() holds an OrderedDict containing all of the layers of
        # the ResNet-50 model, except for the final fully-connected (fc)
        # classification layer; iterate thru these layers, in order, and pass
        # the data in and out of each one to perform the equivalent of a
        # normal forward() method; but, as we do so, capture the outputs
        # from those layers whose outputs we wish to return from this method,
        # which is the convolutional feature maps output by layers: layer1,
        # layer2, layer3 and layer4
        #
        # store the captured outputs from layer1, layer2, layer3 and layer4
        # in an OrderedDict; but, whilst constructing the OrderedDict of
        # feature maps, change the names of the layers by assigning new keys
        # so that the keys (new layer names) become '0', '1', '2', '3'    
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x

        return out
```
The OrderedDict of convolutional feature maps output by layers 1,2,3,4 of the ResNet-50 model will be passed as inputs to the FPN to use them to construct a feature pyramid.


### The FP of the `FeaturePyramidNetwork` module `fpn`

Here we describe what happens within the `forward()` method of the `fpn` subnetwork component of the `backbone` subnetwork.

Recall (from above) that the `fpn` component is stored in the `self.fpn` attribute of the `backbone`. The `self.fpn` attribute of the `backbone` contains an object of class `FeaturePyramidNetwork`.

The `forward()` method of class `FeaturePyramidNetwork` computes a FPN from sets of convolutional feature maps.

Feature Pyramid Networks are designed to learn scale-invariant features, which makes CNNs that include FPNs better at detecting objects at different scales, including, in particular, very small objects. The job of the ResNet-50 component is primarily to learn spatial features (or spatially-invariant features) in the input images. The job of the FPN is primarily to learn scale features (or scale-invariant features) from within the convolutional feature maps (at different resolutions/sizes) learned by the different layers of the ResNet-50 model.

When reading the `forward()` method code which follows shortly, recall that attribute `self.inner_blocks` looks like this:
```Python
(inner_blocks): ModuleList(
  (0): Conv2d(256, 256, kernel_size=(1,1), stride=(1,1))
  (1): Conv2d(512, 256, kernel_size=(1,1), stride=(1,1))
  (2): Conv2d(1024, 256, kernel_size=(1,1), stride=(1,1))
  (3): Conv2d(2048, 256, kernel_size=(1,1), stride=(1,1))
)
```
and that attribute `self.layer_blocks` looks like this:
```Python
(layer_blocks): ModuleList(
  (0): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (1): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (2): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
  (3): Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
)
```

Here is (a modified and heavily commented version of) the `forward()` method of the `fpn` component of our `backbone`. The modifications simply replace function calls in the code with equivalent code, making the function calls unnecessary. The PyTorch authors were forced to implement certain lines of code as equivalent functions in order for their code to be supported (compilable) by TorchScript.  The original lines of code have been retained but are commented-out.
```Python
class FeaturePyramidNetwork(nn.Module):
    def forward(self, x):
        """
        Args:
          x - an OrderedDict of tensors of convolutional feature maps output  
              from layers [layer1, layer2, layer3, layer4] of the ResNet-50
              model, but now with layer names (keys) '0', '1', '2', '3'
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())  # ['0', '1', '2', '3']
        x = list(x.values())
        # x = [
        #   0 feature maps: CxHxW = 256xH0xW0,  C2
        #   1 feature maps: CxHxW = 512xH1xW1,  C3
        #   2 feature maps: CxHxW = 1024xH2xW2, C4
        #   3 feature maps: CxHxW = 2048xH3xW3  C5
        # ]

        # convolve ResNet-50 feature maps from highest (last) layer, 3;
        # C5 (2048xH3xW3) -> Conv2d(2048, 256, k=1, s=1) -> (256xH3xW3) (C5a)
        last_inner = self.inner_blocks[-1](x[-1])
        #last_inner = self.get_result_from_inner_blocks(x[-1], -1)

        # initialise place to store the feature maps constructed for the pyramid
        results = []

        # convolve C5a to create feature maps at top level of pyramid
        # C5a (256xH3xW3) -> Conv2d(256, 256, k=3, s=1, p=1) -> (256xH3xW3) (P5)
        # then store pyramid feature maps P5 in results
        results.append(self.layer_blocks[-1](last_inner))
        #results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # work down the pyramid from the top, constructing the pyramid
        # feature maps for the next-lower level of the pyramid at each iteration  
        for idx in range(len(x) - 2, -1, -1):  # idx in [2,1,0]

            # convolve ResNet-50 feature maps from the next lower layer (level N)
            # CN (CxHxW) -> Conv2d(C, 256, k=1, s=1) -> (256xHxW) (CNa)
            inner_lateral = self.inner_blocks[idx](x[idx])
            #inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)

            # upsample (make bigger) the pyramid feature maps from level N+1,
            # PN+1, to create a temporary pyramid feature map that matches the
            # size of the ResNet-50 feature maps at level N; call it PN+1u
            # e.g.: P5 (256,H3,W3) --> (256,H2,W2) P5u
            # e.g.: P4 (256,H2,W2) --> (256,H2,W2) P4u
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")

            # lateral connection: merge (add, element-wise) the upsampled
            # pyramid feature map, PN+1u, with the convolved ResNet-50 feature map
            # at level N
            # i.e.: last_inner = CNa + PN+1u
            last_inner = inner_lateral + inner_top_down

            # convolve last_inner to create feature maps at level N of the pyramid
            # last_inner (256xHxW) -> Conv2d(256, 256, k=3, s=1, p=1) -> (256xHxW) (PN)
            # then store pyramid feature maps PN at the beginning of the results list
            results.insert(0, self.layer_blocks[idx](last_inner))
            #results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # once the 'for' loop finishes, we have:  results = [P2, P3, P4, P5]

        # take the P5 feature maps that are currently at the top of the pyramid
        # and apply a max_pool2d operation to them that is designed to down-sample
        # by a factor of 2, creating new P6 feature maps at the top of the
        # pyramid
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        # results = [P2,P3,P4,P5,P6], where the P6 feature maps have size
        #           HxW = HpxWp;
        # names = ['0', '1', '2', '3', 'pool']

        # convert the names and pyramid feature maps into an ordered dictionary
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        # out = {'0':P2, '1':P3, '2':P4, '3':P5, 'pool':P6}

        return out
```

Feature Pyramid Networks are defined in the paper `Feature Pyramid Networks for Object Detections`, (Lin et al., 2017). Section 3 describes the construction process for a feature pyramid. This construction process is precisely what the PyTorch authors have implemented in the `forward()` method code presented above (as well as a nuanced bit from Section 4.1 of the paper, including a footnote).

The construction of the pyramid involves a *bottom-up pathway*, a *top-down pathway*, and *lateral connections*.

The *bottom-up pathway* is the feed-forward computation of the ResNet-50 model. They use the feature activations output by each stage's last residual block. They denote these outputs as C2, C3, C4 and C5. These outputs are what we have in the unpacked variable `x`, above: the feature maps output by layers layer1 (C2), layer2 (C3), layer3 (C4) and layer4 (C5), respectively.  The dimensions of the feature maps decrease as we move up the *bottom-up* pathway (ie with each successive ResNet layer). So we have: layer1 (C2) H0xW0, layer2 (C3) H1xW1, layer3 (C4) H2xW2, layer4 (C5) H3xW3.

The *top-down* pathway upsamples (ie increases the size of) smaller pyramid feature maps from a higher level so that they match the size ResNet-50 feature maps from the next lower level. The upsampling is down with nearest-neighbours interpolation. This is what we see in the lines of code  
```Python
feat_shape = inner_lateral.shape[-2:]
inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
```

The *lateral connections* merge (add, element-wise) feature maps of the same spatial size from the *bottom-up* and *top-down* pathways. These are designed to *enhance* the pyramid features with features from the *bottom-up* pathway (ie from the ResNet-50 layers). This is what we see in the line of code
```Python
last_inner = inner_lateral + inner_top_down
```

As we move down the *top-down* pathway, we construct successively lower levels of the pyramid feature maps. At each level a `kernel_size=(1,1)` convolution is applied to the ResNet-50 layer feature maps at that level. The purpose here is dimensionality reduction: to reduce the number of channels in the ResNet-50 feature maps to the number of channels used in the pyramid feature maps so that the element-wise addition applied in the lateral connections can take place.

Once the lateral connections are applied (ie the element-wise addition takes place), a final `kernel_size=(3,3)` convolution is applied to the result of the element-wise addition. The feature maps that result from this 3x3 convolution are the pyramid feature maps for the current level of the pyramid.

The last layer of the `fpn` subnetwork is stored in attribute `self.extra_blocks`.  It is an instance of class (nn.Module) `LastLevelMaxPool`. This module has a tiny `forward()` method whose role is to simply apply a *max_pool2d* operation to the feature maps at the top of the pyramid (P5) and *extend* the pyramid with one more level on top (P6, lets call it). The kernel size is 1x1, stride=2 and padding=0, so the *max_pool2d* operation effectively down-samples P5 by a factor of 2, meaning P6 has feature maps half the size of 25x25, which becomes feature maps of size 13x13.  The P6 feature maps are *appended* to the top of the pyramid and given the name 'pool'.
```Python
class LastLevelMaxPool(ExtraFPNBlock):
    def forward(self,x,y,names):
        """
        Args:
          x - the 'results' of the feature pyramid construction, so far; a list
              of tensors of pyramid features maps [P2, P3, P4, P5], ordered from
              the bottom of the pyramid to the top
          y - the original ResNet-50 feature maps for layers 1,2,3,4;
              but these are NOT used; so why are they passed in????
          names - a list of the names of the current layers of the pyramid;
                  ['0', '1', '2', '3']
        """
        names.append("pool")

        # x[-1]=P5; with k=1, s=2, p=0, the max pooling down-samples
        x.append(F.max_pool2d(x[-1], 1, 2, 0))

        # The pyramid results, in x, now look like this: [P2,P3,P4,P5,P6]
        # The names of the levels are now
        # ['0', '1', '2', '3', 'pool']
        return x, names
```

Arranged as a pyramid, the shapes of the pyramid feature maps in the OrderedDict returned by the `forward()` method of class `FeaturePyramidNetwork` look like this:
```
     pool P6 CxHxW (256, Hp, Wp)  
        3 P5 CxHxW (256, H3, W3)
        2 P4 CxHxW (256, H2, W2)
        1 P3 CxHxW (256, H1, W1)
        0 P2 CxHxW (256, H0, W0)
```

Note that the HxW dimensions of the feature maps are greatest at the bottom of the pyramid (ie H0 x W0) and smallest at the top of the pyramid (Hp x Wp).

Observe that the only inputs to the FPN's `forward()` method (ie the only inputs to the construction of the feature pyramid) were the 3D volumes of convolutional feature maps output from the layers of the ResNet-50 model: the layer1 (C2), layer2, layer3, layer4 (C5) outputs. Here they are, arranged with layer1 (C2) on the bottom and layer4 (C5) at the top (ie in an order that corresponds to the arrangement of the layers of the FPN):
```
  L4 3 C5 CxHxW  (2048, H3, W3)
  L3 2 C4 CxHxW  (1024, H2, W2)
  L2 1 C3 CxHxW  ( 512, H1, W1)
  L1 0 C2 CxHxW  ( 256, H0, W0)
```
So all of the information (features) in the feature maps of the feature pyramid originated from the convolutional feature maps output by the layers of the ResNet-50 model. So an FPN creates a re-expression of the features and feature maps learned by the layers of the ResNet-50 model and standardises the numbers of channels in those re-expressed feature maps at 256. Another important aspect of the features in the levels of an FPN feature pyramid is that they are constructed *top-down* whereas the ResNet-50 feature maps were created *bottom-up*. And, crucially, in the *top-down* construction process, convolved versions of the ResNet-50 feature maps are *merged* into (added to, element-wise) the (upsampled) feature pyramid feature maps from the level above. So the coarser but semantically richer features from the upper levels of the pyramid are transmitted downward to the finer but semantically weaker features at lower levels of the pyramid. This results in a more consistent distribution of semantic content across the levels of the feature pyramid. It also results in information at different scales being distributed across the levels of the feature pyramid. So the features of the FPN give us a blending of spatial information (spatially-invariant features) and scale information (scale-invariant features).

Here is simplified view of the FPN construction process which helps us to visualise not only the construction process but the blending of spatial and scale feature information just discussed:  
```
      ResNet-50 feature maps                   FPN feature maps

                                          pool P6 CxHxW (256, Hp, Wp)
                                                           ^
                                                           |
L4   3 C5 CxHxW  (2048, H3, W3)    --->      3 P5 CxHxW (256, H3, W3)
                     ^                                     |
                     |                                     V
L3   2 C4 CxHxW  (1024, H2, W2)    --->      2 P4 CxHxW (256, H2, W2)
                     ^                                     |
                     |                                     V
L2   1 C3 CxHxW  ( 512, H1, W1)    --->      1 P3 CxHxW (256, H1, W1)
                     ^                                     |
                     |                                     V
L1   0 C2 CxHxW  ( 256, H0, W0)    --->      0 P2 CxHxW (256, H0, W0)
```
What's missing from (implicit in) this diagram are 1) the convolutions that are applied, 2) the upsampling of pyramid features from one level to level below, and 3) the *merging* (element-wise addition) of the ResNet-50 convolutional feature maps and the FPN pyramid feature maps.  For example, let's review how the feature maps of pyramid level 2 (ie P4) are created. First, a 1x1 convolution is applied to ResNet-50 C4 to create a (256,H2,W2) volume (C4a). Then, pyramid level 3 (P5) is upsampled from (256,H3,W3) to create (256,H2,W2) (P5u). Then we merge (add) these two volumes, C4a + P5u, giving a volume of size (256,H2,W2).  Then we do a final 3x3 (`same`) convolution on this summed volume, giving P4 of size (256,H2,W2).

Pyramid level 3 (P5) is created first, to start the *top-down* construction process. P5 is created by applying a 1x1 convolution to C5, giving a volume of size (256,H3,W3). The 3x3 (`same`) convolution is then applied, giving P5 with size (256,H3,W3).

To finish the construction of the pyramid, a max_pool2d operation is applied to P5 to down-sample it by a factor of 2, to yield P6 at the top of the pyramid with size (256,Hp,Wp).



## The forward pass of the `RegionProposalNetwork` subnetwork

### Preliminary overview and discussion of RPN

source paper: Feature Pyramid Networks for Object Detection, Lin et al., 2017

DH: most of what follows in this section is a *very near verbatim copy* of the text from the Lin (2017) paper; where I interject with my own comments, I prefix them with 'DH:'.
* the original paper on RPNs is: Faster R-CNN: Towards Real-Time Object Detection with RPNs, Ren et al., 2015; but I found I didn't need to draw on this paper to understand RPNs; because the PyTorch Faster R-CNN model uses and FPN in its backbone, the adapted RPN described by Lin (2017) feels more relevant to understanding the PyTorch Faster R-CNN implementation

DH: what is described below, from Lin (2017), appears to be what has been implemented by the PyTorch authors of the PyTorch implementation of the Faster R-CNN model!!

RPN is a sliding-window class-agnostic object detector.

An RPN is a small subnetwork.

The *original RPN design* operates on a single-scale convolutional feature map. It performs object/non-object binary classification and bounding box regression.  This is realised by *a 3x3 convolutional layer followed by two sibling 1x1 convolutions for classification and regression* which we refer to as a network *head*.

The object/non-object (ie binary classification) criterion and bounding box regression target are defined with respect to a set of *reference boxes* called *anchors*. The anchors are of multiple pre-defined scales and aspect ratios in order to cover objects of different shapes.
DH: of different shapes and sizes, presumably

*We adapt RPN for use with FPN* by replacing the single-scale (ie single-level) feature map with the multi-scale (ie multi-level) feature pyramid output by FPN. *We attach a head of the same design (3x3 conv and two sibling 1x1 convs) to each level of our feature pyramid*.
DH: Our pyramid (in Faster R-CNN) has 5 levels, so the RPN Head gets used 5 times, implying a total of 5 * 3 = 15 individual convolution operations. Note also: there is a single instance of the RPN head, and that single instance of the RPN head is used on all 5 levels of the pyramid. This implies that there is a single set of RPN head parameters and that these, therefore, must be shared across all 5 levels of the pyramid.  There are NOT multiple RPN heads (plural). There is a single RPN head that gets used (applied) against all 5 levels of the feature pyramid.  See below for statements directly from the paper that confirm this point.

Because the RPN Head slides densely (ie stride=1 in all 3 convolutions) over all locations in all pyramid levels, it is not necessary to have multi-scale anchors on a specific level of the pyramid. Instead, *we assign anchors of a single scale to each level*.
DH: to each level of the feature pyramid.

Formally, *we define the anchors to have areas of 32x32, 64x64, 128x128, 256x256, 512x512 pixels on pyramid levels P2, P3, P4, P5, P6, respectively*. As in Ren (2015), *we also use anchors of multiple aspect ratios (1:2, 1:1, 2:1) at each level*. So, in total, there are 15 anchors over the pyramid.
DH: 5 levels of pyramid, with each level having anchors of a given area but with 3 different aspect ratios, implies 5 * 3 = 15 anchors

DH: Don't be confused by the relation between the size of the anchors and size of the feature maps in the pyramide levels.  For example, anchors of size 512x512 are used with the top level of the feature pyramid where the feature maps have size 256x13x13.  But we have to remember that the anchors aren't applied to the feature maps, they are applied to the *images*, and the images will generally be much larger in area than 512x512. However, it is interesting to observe that the RPN design uses larger anchors on pyramid levels where the feature maps are smaller, and smaller feature maps on the pyramid levels where the feature maps are larger.

*We assign training labels to the anchors based on their Intersection-over-Union (IoU) ratios with ground-truth bounding boxes, as in Ren (2015)*. Formally, an anchor is assigned a *positive label* if it has the highest IoU for a given ground-truth bbox or an IoU over 0.7 with any ground-truth bbox, and a *negative label* if it has IoU lower than 0.3 for all ground-truth bboxes.
DH: what actually happens in the PyTorch implementation of RPN differs somewhat from what the Lin (2017) paper just said. An

Note that scales of ground-truth bboxes are not explicitly used to assign them to the levels of the pyramid; instead, ground-truth bboxes are associated with anchors, which have been assigned to pyramid levels.

We note that *the parameters of the heads are shared* across all feature pyramid levels.
DH: that is, there is actually only one, single RPN head, but it gets applied to each of the 5 levels of the feature pyramid; so the paper's use of the plural here (`heads`) is a bit misleading; there is a single RPN head that's reused against each of the 5 levels of the feature pyramid.


### The FP of the `RegionProposalNetwork` module `rpn`

When called, the RPN is passed 3 inputs: images, features and targets. The `RPNHead` is run on the features to produce 2 outputs: objectness and pred_bbox_deltas.  The `AnchorGenerator` is then run on the features to produce anchor boxes.

Interestingly, the `RPNHead` and `AnchorGenerator` modules are *independent* of one another. The outputs of one are NOT fed into the other, so the order in which they are run does not matter.  In the PyTorch implementation, the `RPNHead` is called before the `AnchorGenerator`, but this is an arbitrary choice.

As we will see below, the outputs of the components `RPNHead` (objectness and pred_bbox_deltas) and `AnchorGenerator` (anchor boxes) are merely inputs to substantial subsequent processing that is managed by the remainder of the `forward()` method of the `RegionProposalNetwork`. It is this subsequent processing which transforms these inputs into region proposals and TODO

Note that the images themselves are NOT processed (or used) by the RPN in any way. In fact, they are barely needed by the RPN at all. They are only needed by the AnchorGenerator to help it generate and package anchor boxes correctly. The only image information needed by the AnchorGenerator is: 1) the common HxW size of the transformed (zero-padded) images, and 2) the number of images.    

Here is a heavily commented version of the `forward()` method of class `RegionProposalNetwork`:
```Python
def forward(self, images, features, targets):
    """
    Args:
      images - an ImageList object contained the transformed images
      features - the feature pyramid output by the Feature Pyramid Network
                 component of the 'backbone' subnetwork; an OrderedDict with
                 five levels of feature maps that looks like this:
                 ```
                      pool P6 CxHxW (256, Hp, Wp)  
                         3 P5 CxHxW (256, H3, W3)
                         2 P4 CxHxW (256, H2, W2)
                         1 P3 CxHxW (256, H1, W1)
                         0 P2 CxHxW (256, H0, W0)
                 ```
      targets - a list of dictionaries, one dictionary per image, containing
                the ground-truth bounding boxes and corresponding class labels   
    """
    # get the features of the feature pyramid (in the correct pyramid order)
    features = list(features.values())

    # run the RPNHead on the features at each level of the feature pyramid
    # to generate the RPN's predictions:
    # 1) the object/non-object binary classification prediction of whether
    #    or not each anchor box contains an object (of some kind) or not; this
    #    is equivalent to predicting whether the anchor box contains foreground
    #    or background; foreground corresponds to an object (of some kind),
    #    and background corresponds to 'no object'
    # 2) the predicted bbox regression deltas (predicted differences between
    #    the dimensions of anchor boxes and predicted boxes)
    objectness, pred_bbox_deltas = self.head(features)
    # 'objectness' is a list of 5 tensors (volumes) with shapes
    # (ignoring the Batch dimension) of
    # [(3, H0, W0), (3, H1, W1), (3, H2, W2), (3, H3, W3), (3, Hp, Wp)]
    # containing the cls_logits produced for object/non-object binary
    # classification for each anchor box;
    # the shapes of these volumes corresponds exactly to the anchor boxes that
    # are about to be generated in the next line of code; there is a 1-to-1
    # mapping between each cell in these volumes and a unique anchor box;
    # the cls_logit value in a given cell expresses a binary classification
    # prediction as to whether the corresponding anchor box contains an object
    # or not; these are class-agnostic binary predictions of object/non-object    
    #
    # 'pred_bbox_deltas' is a list of 5 tensors (volumes) with shapes
    # (ignoring the Batch dimension) of
    # [(12, H0, W0), (12, H1, W1), (12, H2, W2), (12, H3, W3), (12, Hp, Wp)]
    # containing numbers that are regarded as predicted bbox 'deltas';
    # here, there is a 4-to-1 mapping between each group of 4 cells representing
    # a predicted bbox (delta) and a unique anchor box

    # Generate the anchor boxes
    #
    # call the `forward()` method of class AnchorGenerator in module
    # anchor_utils.py in package torchvision.models.detection
    #
    # Return: 'anchors' - a list of tensors, one tensor per image, where each
    # tensor is IDENTICAL and holds all of the anchor boxes generated for all
    # levels of the feature pyramid; each identical tensor in 'anchors' holds
    # 159,882 anchor boxes (anchors_per_image)
    #
    # (nb: the only image information the anchor generator uses is the
    #  common HxW 'size' of the images and the 'number' of images)
    anchors = self.anchor_generator(images, features)  

    # derive the number of images
    num_images = len(anchors)

    # extract the shapes of the tensors in 'objectness' in order to then
    # calculate the number of anchors that were generated with respect to each
    # level of the feature pyramid; and then package these per-level counts of
    # anchors into a list
    # (note: we can't retrieve the number of anchors per feature pyramid
    # level from the 'anchors' variable because each element of that list
    # is a fully concatenated tensor of ALL anchors, and so does not
    # convey information about which anchors pertain to which level of
    # the feature pyramid  
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = \
        [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    # we get: num_anchors_per_level = [120000, 30000, 7500, 1875, 507]

    # for both objectness and pred_bbox_deltas: rearrange the dimensions of
    # each volume, then flatten each volume, then concatenate the flattened
    # volumes; that is, change the representation of the anchor box
    # classification and bbox regression (delta) predictions
    objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)

    # The next function constructs initial proposal bboxes (initial
    # candidate bbox predictions).
    #
    # Given 1) the pred_box_deltas (relative bbox offsets) for each image,
    # and 2) the original anchor (reference) boxes (one full set of them)
    # for each image:
    # - compute the widths, heights and centres (ctr_x, ctr_y) of every
    #   anchor
    #   (reference) box (for every image)
    # - using the pred_box_deltas (relative box offsets) for each image,
    #   together with the widths, heights and centres of the anchor
    #   (reference) boxes for each image, construct initial proposal
    #   bboxes (ie a large set of initial candidate bbox predictions)
    #
    # The algorithm constructs as many proposals as there are anchor
    # (reference) boxes (a large number depending on image sizes, etc.).
    # It does this for each image in the current mini-batch.
    # Clearly, the number of initial proposals (initial candidate
    # predicted bboxes) per image is far, far greater than the number of
    # objects likely to appear in any given image. So most of the initial
    # proposals will be poor ones.
    #
    # (note: the pred_bbox_deltas are detached because Faster R-CNN does not
    # backprop through the proposals)
    proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)

    # Prepare a reshaped view of the initial proposals so that the  
    # filter_proposals() function (see next) can easily iterate through
    # the initial proposals associated with individual images.
    proposals = proposals.view(num_images, -1, 4)

    # Filter the large set of initial proposals (initial candidate bbox
    # predictions) and objectness scores and return only the top-n 'best'
    # proposals and their corresponding objectness scores; do this for
    # each object detected in each image; this is a multi-step process.
    #
    # Here is a summary of the multi-step process involved in filtering
    # the initial proposals and objectness scores to find only the top-n
    # 'best' proposals that are returned in variable 'boxes' (see below):
    #
    # 1) create a large mask tensor containing only indices of feature
    #    pyramid levels, (0,1,2,3,4). Construct long sequences of identical
    #    indices so that the mask tensor spans across the full set of
    #    proposals and objectness scores. This long mask of indices is
    #    used to map proposals and objectness elements to their associated
    #    feature pyramid level. The number of proposals and objectness
    #    elements is equal to the number of anchor boxes (a large number),
    #    so this mask is long.
    #    - the mask variable here is called 'levels'
    #    - 'levels' is like [0,0,...0,1,1...1,2,2...2,3,3...3,4,4..4]
    #
    # 2) get the indices of the top-n largest objectness scores
    #    (cls_logits), selected independently per feature pyramid level
    #    (using mask 'levels'); these indices also point to the top_n
    #    proposals, since the structures of the objectness scores and
    #    proposals are synchronised
    #    - at this stage, 'top-n' is specified by an RPN model parameter
    #      called 'pre_nms_top_n'; (for trainging mode, the default
    #      value is 2000; for test mode, the default value is 1000)
    #    - the variable holding the top_n indices is called 'top_n_idx'
    #
    # 3) arrange the objectness scores, proposals and 'levels' into per-
    #    image batches so we can process them on a per-image basis
    #    - having done so, for each image, extract the top_n objectness
    #      scores and corresponding proposals and levels indices
    #
    # For each image:
    #
    # 4) apply a Sigmoid activation function to the extracted top_n
    #    objectness scores (cls_logits), turning them into probability
    #    scores
    #    - store these in variable 'objectness_prob'
    #
    # 5) prepare and find the final top-n proposals, with corresponding
    #    objectness_probs, for each image:
    #    - clip the top_n proposals so they lie inside the corresponding
    #      image's size
    #    - remove small proposals (those with at least one side < min_size)
    #      and the corresponding objectness_probs and levels; (but
    #      min_size = 1e-3 (0.001) via hard-coded value, so it looks like
    #      no proposal will ever be removed for being too small)
    #    - remove low-scoring proposals (those with objectness_prob <
    #      score_thresh)
    #    - apply non-maximum suppression (nms), independently per feature
    #      pyramid level; ie select the best proposal for each object and
    #      reject (or suppress) all the other highly-overlapping proposals;
    #      this uses objectness_probs and the IoU with other proposals;
    #      select the proposal with the highest objectness_prob and remove
    #      all the other proposals that overlap highly with that
    #      proposal (ie where IoU exceeds the nms_thresh, which defaults
    #      to 0.7); iterate in this way until no more proposals are removed
    #      (ie suppressed); for the proposals and scores that survive nms,
    #      order these in descending order by the scores
    #    - of the proposals (and scores) that survive nms, keep only the
    #      post_nms_top_n of these; (in training mode this defaults to )
    #      2000 per image; in test mode it defaults to 1000 per image)
    #    - the final set of filtered (top-n) proposals) and scores for an
    #      image are kept in variables called 'boxes' and 'scores';
    #      append these to lists called 'final_boxes' and 'final_scores'
    #
    # When we're finished processing all the images in the mini-batch,
    # return the variables (lists) 'final_boxes' and 'final_scores'.
    #
    boxes, scores = self.filter_proposals(
           proposals, objectness, images.image_sizes, num_anchors_per_level
    )
    # notes:
    # - variable 'boxes' (the top-n 'best' region proposals for each
    #   image) is returned by the RPN (see below), both in training mode
    #   and test mode; remember: by default, in training mode there will
    #   be 2000 top-n region proposals per image, and in test mode 1000
    #   top-n region proposals per image
    # - variable 'scores' (the objectness scores for the top-n 'best'
    #   region proposals per image) is NOT returned by the RPN; nor are
    #   these scores used in any way after this point; IE they do NOT play
    #   a role in the calculation of the RPN losses when in training mode);
    #   the objectness scores simply provided the basis for selecting the
    #   top-n 'best' region proposals; now that that task is finished,
    #   the actual scores of the top-n 'best' region proposals per image
    #   are not of interest (except, perhaps, during model design,
    #   development and testing)

    # calculate the RPN losses (if in training mode)
    losses = {}
    if self.training:
        assert targets is not None

        # The next function that's called, assign_targets_to_anchors(), is
        # an important one for understanding what an RPN does and how it
        # works. It's a complex function that calls complex sub-functions,
        # so we try to provide a detailed explanation of what it does.
        # The outputs of this function are essential components for
        # computing the RPN losses.
        #
        # FOR EACH IMAGE:
        # 1) measure the similarity (overlap) between the ground-truth (gt)
        #    boxes and the anchor boxes; doing so involves constructing
        #    an MxN match_quality_matrix where each cell contains the IoU
        #    (intersection-over-union) for a unique pair of gt and anchor
        #    boxes; there are M gt boxes and N anchor boxes (where M << N)
        # 2) pass the match_quality_matrix to a 'matcher';
        #    (see the __call__ method of class Matcher in module _utils.py
        #    in package torchvision.models.detection); the matcher
        #    builds a tensor of size N that initially assigns to each anchor
        #    box the index (in [0, M-1]) of the gt box with which it has the
        #    highest IoU; the initial matches to gt boxes are then analysed
        #    to verify if certain IoU conditions are satisified and, if not,
        #    the initial matches to gt boxes are adjusted and refined;
        #    a) if the IoU for an initially matched gt box is < bg_iou_thresh,
        #    the index of that gt box is overwritten with -1; this means
        #    the anchor box is judged not to contain any 'foreground'
        #    object and to contain 'background' only; in other words, the
        #    bg_iou_thresh parameter represents the maximum IoU with a gt box  
        #    for an anchor box to be judged to contain background only,
        #    ie to be considered a 'negative' anchor box example
        #    b) if the IoU for an initially matched gt box is >= bg_iou_thresh
        #    AND < fg_iou_thresh (ie it's between these two thresholds),
        #    the index of that gt box is overwritten with -2; this means
        #    that the anchor box fails to satisfy both the criterion to be
        #    judged to contain background only and the criterion to be judged
        #    to contain a foreground object; in other words, the fg_iou_thresh
        #    parameter is the minimum IoU with a gt box for an anchor box
        #    to be judged to contain a foreground object, ie to be
        #    considered a 'positive' anchor box example; anchor boxes with
        #    low-quality matches such as these, where the IoU falls between
        #    the bg_iou_thresh (lower threshold) and fg_iou_thresh (upper
        #    threshold) fail to quality as being either 'negative' or
        #    'positive' examples; they will ultimately be discarded from
        #    consideration during onward processing by the RPN;
        #    c) a model parameter exists to 'allow low quality matches'
        #    which, if True, then leads to matching some anchors with -1 or
        #    -2 values with gt boxes where the IoUs are low but the best
        #    available; this turns some unmatched anchors into matched
        #    anchors, but the algorithm doesn't eliminate all -1 and -2 values.
        #    Note:  
        #    The references to foreground and background help explain the  
        #    nature of the object/non-object binary classification predictions
        #    that the RPN Head is responsible for making. The
        #    object/non-object distinction (classification decision) is
        #    essentially a foreground/background classification decision: ie
        #    the part of the RPN Head that predicts the cls_logits is trying
        #    to predict whether each anchor box contains foreground or
        #    background, which is the same as predicting whether it contains an
        #    object or not.
        #    Note:
        #    This stage of the processing returns a tensor of size N
        #    containing, for each anchor box, the index of a matched gt box
        #    (in [0, M-1]) or a negative number (-1 or -2).
        #    We call this tensor: 'matched_idxs'
        # 3) Using tensor matched_idxs (from step 2), get the corresponding
        #    actual gt boxes from the targets for the current image; store
        #    these in 'matched_gt_boxes_per_image'; note: this step is done
        #    by 'clamping' the values in matched_idxs with a min=0, meaning
        #    that, during this 'get' operation, all of the -1 and -2 values
        #    in matched_idxs are converted to 0, which is a valid gt box index,
        #    since the valid gt box indices run from 0...M-1; this has two
        #    consequences:
        #    a) the tensor 'matched_gt_boxes_per_image' will still have size
        #    N, which is extremely useful and is exploited in the code that
        #    implements subsequent processing; but, more worryingly,
        #    b) unwanted instances of the gt box for index 0 are extracted
        #    and included in 'matched_gt_boxes_per_image' when, in fact,
        #    they are bogus matches.
        #    However, consequence (b) is NOT A CONCERN because these bogus
        #    matches with the gt box with index 0 won't introduce any unwanted
        #    noise into proceedings by corrupting (making noisy) the RPN
        #    loss computations.
        #    Here is why they are not a concern:
        #    a) For cases where values of -2 get clamped to index 0, the
        #    corresponding anchor boxes will get labeled in such a way
        #    (see step 4) that they get discarded and are NOT sampled for RPN
        #    loss calculation. They will redundantly be involved in the
        #    calculation of box regression targets (see box_coder.encode
        #    function, further down), but those targets will never get used in
        #    the computation of the box regression loss because only anchors
        #    labelled as 'positive' are involved in the computation of the
        #    box regression loss.
        #    b) For cases where values of -1 get clamped to index 0, the
        #    corresponding anchor boxes are labelled as 'negatives' (see step
        #    4), meaning they are judged to be 'background' and to not
        #    contain an object, so they don't need a bbox predictions. So,
        #    'negative' anchors, even if sampled for RPPN loss computation,
        #    are never involved in the computation of box regression loss. Once
        #    again, the bogus matched gt boxes will be redundantly involved
        #    in the computation of box regression targets, but those targets
        #    will never be used in RPN classification loss computation.
        #    And even though 'negative' anchors are sampled for the  
        #    computation of RPN object/non-object binary classification loss,
        #    that loss computation does not involve bboxes (either gt or
        #    predicted) in any way, so the bogus matched gt boxes won't cause
        #    a problem.
        # 4) Label the anchor boxes (for the current image).
        #    Using tensor matched_idxs again, create a corresponding tensor of
        #    size N that will be used to 'label' each anchor box.
        #    Assign a 'label' of 1.0 to each anchor box that is matched to a
        #    gt box (ie has a value >= 0 in matched_idxs). Label 1.0 signals
        #    that the corresponding anchor box is judged to contain
        #    'foreground' (ie an object, of some kind) and will be regarded
        #    as a 'positive' anchor during anchor box sampling (later).
        #    Then find the indices in matched_idxs with values of -1
        #    (meaning the highest IoU with a gt box was below the
        #    bg_iou_thresh). Assign a 'label' of 0.0 to these
        #    anchor boxes. A label 0.0 signals that the corresponding anchor
        #    box is judged to contain 'background' (ie no object) and will
        #    be regarded as a 'negative' anchor during anchor box sampling
        #    (later).
        #    Then find the indices in matched_idxs with values of -2 (meaning
        #    the highest IoI with a gt box was 'between' the low and
        #    high thresholds). Assign a 'label' of -1.0 to these anchor boxes.
        #    A label -1.0 signals that the corresponding anchor box will be
        #    ignored (discarded) during anchor box sampling (later) because
        #    they are judged to neither be 'positive' (foreground) nor
        #    'negative' (background) examples.
        # 5) Append the tensor of anchor labels for the current image to a
        #    list called 'labels'. Append the tensor of matched gt boxes in
        #    variable 'matched_gt_boxes_per_image' to a list
        #    called 'matched_gt_boxes'.
        #
        # Once we have finished processing the anchor boxes and gt boxes for
        # each image, per steps 1 thru 5, return the two lists:
        # 'labels' - a list of tensors of anchor labels, one tensor per image
        # and each tensor of size N
        # 'matched_gt_boxes' - a list of tensors of gt bboxes that have been
        # matched to anchors, one tensor per image, and each tensor of size N;
        #
        # Observe: the sizes of variables 'labels' and 'matched_gt_boxes'
        # both match that of variable 'anchors' in the sense that:
        # a) each is a list of tensors having the same length, since each
        #    of these lists contains one element per image in the current
        #    mini-batch, and
        # b) each tensor in each list has length N, which is the number of
        #    anchor boxes used in the model
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        # note: the positive (1.0) and negative (0.0) labels assigned to anchor
        # boxes play a critical role in computing the RPN losses, shortly. They
        # are used in two ways: a) for sampling batches of 'positive' and
        # 'negative' anchors to be used in computing both of the two RPN losses,
        # and b) as 'targets' when computing RPN objectness classification loss.

        # The next method is part of class BoxCoder in module _utils.py of
        # package torchvision.models.detection.
        #
        # Processing steps:
        # 1) calculate the widths, heights and centres of each anchor box,
        #    for each image  
        # 2) calculate the widths, heights and centres of each matched gt box,
        #    for each image
        # 3) for corresponding pairs of anchor boxes and matched gt boxes,
        #    compute box deltas, (dx, dy, dw, dh), where dx and dy represent
        #    coordinate deltas for box centres, and dw and dh represent
        #    coordinate deltas for box widths and heights, respectively;
        #    [note: there will always be corresponding pairs because the
        #     inputs to the method, 'matched_gt_boxes' and 'anchors', are  
        #     both lists with the same number of tensors (one per image), and
        #     each tensor has length N (the number of anchor boxes used in
        #     the model)]
        #
        # The box deltas represent RPN 'regression targets' in the sense that
        # the box_regression part of the RPN Head learns to predict bbox
        # deltas. So the box deltas computed here represent the 'targets'
        # against which to compute the loss on the predictions output by the
        # box regression part of the RPN Head and stored in variable
        # 'pred_bbox_deltas'
        #
        # the 'regression_targets' variable returned by the delta encoder is a
        # tuple of tensors, where each tensor holds the regression targets
        # (anchor/gt box deltas) for one image (and, hence, has length N,
        # once again)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

        # Finally, we are ready to compute the losses of the RPN. The RPN Head
        # makes two categories of prediction: a) anchor binary classification
        # predictions and b) bbox delta regression predictions. So two losses  
        # are computed using two separate loss functions.
        #
        # The inputs to the compute_loss() method are:
        # 'objectness' - the RPN's object/non-object (foreground/background)
        #                binary classification predictions for each anchor,
        #                per image
        # 'pred_bbox_deltas' - the RPN's bbox delta regression predictions
        #                      for the differences between anchor boxes and
        #                      predicted boxes, per image
        # 'labels' - the labelling of anchors as 'positive' (foreground) (1.0),
        #            'negative' (background) (0.0) and 'discard' (ignore) (-1.0),
        #            per image
        # 'regression_targets' - box deltas with respect to pairs of anchor boxes
        #                        and matched gt boxes, per image
        #
        # Processing:
        # 1) First, an object of class BalancedPositiveNegativeSampler from
        #    module _utils.py in package torchvision.models.detection is
        #    used to randomly sample 'batches' of predicted anchor boxes that
        #    have been labelled as being either 'positive' or 'negative',
        #    whilst respecting a specified proportion of positive examples.
        #    It samples a 'batch' of anchor boxes for each image from  
        #    amongst the anchor boxes labelled as positive or negative for
        #    that image.
        #    For each image, it creates two binary masks, one indicating the
        #    positive predicted anchor boxes that were sampled for that image,
        #    one indicating the negative predicted anchor boxes that were
        #    sampled for that image. It packages these binary masks into two
        #    lists, one for masks for positive boxes (for all images), one for
        #    masks for negative boxes (for all images). It returns these
        #    two lists of binary masks.
        # 2) A 'loss_rpn_box_reg' is calculated using a 'smooth_L1_loss' function
        #    that compares the pred_bbox_deltas for all sampled 'positive'
        #    anchor predictions to the regression_targets for all sampled
        #    'positive' anchor predictions.
        # 3) A 'loss_objectness' (anchor binary classification prediction loss)
        #    is calculated using a 'binary_cross_entropy_with_logits' loss
        #    function that compares the objectness cls_logits for all
        #    sampled anchor predictions (both positive and negative) with the
        #    labels (now acting as 'targets' for loss computation purposes)
        #    assigned to all sampled anchors (both positive and negative).
        # The two RPN losses for the current mini-batch are then returned.
        loss_objectness, loss_rpn_box_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets)
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

    return boxes, losses
```


### The FP of the `RPNHead` module `head`

Here is the full `forward()` method of class `RPNHead` (with comments added):
```Python
def forward(self, x):
    """
    Args:
      x - the features for each level of the feature pyramid produced by the
          Feature Pyramid Network component of the 'backbone' subnetwork,
          packaged as a list of Tensors
          ```
               pool P6 CxHxW (256,  13,  13)  
                  3 P5 CxHxW (256,  25,  25)
                  2 P4 CxHxW (256,  50,  50)
                  1 P3 CxHxW (256, 100, 100)
                  0 P2 CxHxW (256, 200, 200)
          ```
    """
    logits = []
    bbox_reg = []

    # run the RPN Head on the features at each level of the feature pyramid
    for feature in x:
        # apply the initial 'same' 3x3 convolution on a level of features,
        # followed by RELU activation; the volumes output will have the same
        # shape as in the feature pyramid depicted in the comments above    
        t = F.relu(self.conv(feature))

        # now run two sibling 'same' 1x1 convolutions on the outputs of the
        # initial 3x3 convolution

        # run a 'same' 1x1 convolution (with 3 out_channels) to perform
        # object/non-object binary classification
        logits.append(self.cls_logits(t))

        # run a 'same' 1x1 convolution (with 12 out_channels) to perform
        # bbox regression
        bbox_reg.append(self.bbox_pred(t))

    return logits, bbox_reg
```
The `logits` variable that's returned is a list of 5 tensors (volumes) with shapes:
```
[(3, H0, W0), (3, H1, W1), (3, H2, W2), (3, H3, W3), (3, Hp, Wp)]
```
The `bbox_reg` variable that's returned is a list of 5 tensors (volumes) with shapes:
```
[(12, H0, W0), (12, H1, W1), (12, H2, W2), (12, H3, W3), (12, Hp, Wp)]
```


### The FP of the `AnchorGenerator` module `anchor_generator`

Here is a simplified (and heavily commented) version of the `forward()` method of class `AnchorGenerator`:
```Python
def forward(self, image_list, feature_maps):
    """
    Args:
      image_list - an ImageList object containing: 1) in attribute 'tensors',
                   a tensor of images; 2) in attribute 'image_sizes', a list
                   of tuples giving the size of each image in 'tensors' prior
                   to the images being zero-padded so they all have a common
                   size so they can occupy the same tensor
      feature_maps - a list of tensors, where each tensor contains the
                     feature maps for a given level of feature pyramid created
                     by the FPN (feature pyramid network); these levels are:
                     [ layer1 (P2) 256x200x200, layer2 (P3) 256x100x100,
                     layer3 (P4) 256x50x50, layer4 (P5) 256x25x25, pool (P6)
                     256x13x13 ]
    """
    # get the grid size (HxW) of the feature maps in each level of the pyramid
    # [[H0,W0], [H1,W1], [H2,W2], [H3,W3], [Hp,Wp]]: List[List[int]]
    grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

    # get the common (shared) HxW size of all of the images
    image_size = image_list.tensors.shape[-2:]

    # calculate strides; eg if image_size=(800,1200), we get strides
    # [[4,6],[8,12],[16,24],[32,48],[61,92]], but where every integer
    # is actually a scalar tensor: List[List[torch.Tensor]]
    strides = [ [torch.tensor(image_size[0] // g[0]),
                 torch.tensor(image_size[1] // g[1])] for g in grid_sizes ]

    # Generate zero-centered cell anchors (base anchors) with
    # format [xmin, ymin, xmax, ymax] for each of the sizes (areas)
    # specified in self.sizes ((32,), (64,), (128,), (256,), (512,))  
    # and, for each size, having each of the aspect ratios specified in
    # self.aspect_ratios (0.5, 1.0, 2.0) and then store the generated base
    # anchors in a list in instance attribute self.cell_anchors
    #
    # for example, for size=(32,) we get 3 zero-centred base anchors with
    # area = 32x32, and each having one of the 3 aspect ratios:
    # tensor([[-23., -11.,  23.,  11.],    # short and wide    AR=0.5
    #         [-16., -16.,  16.,  16.],    # square            AR=1
    #         [-11., -23.,  11.,  23.]])   # tall and narrow   AR=2
    #
    self.set_cell_anchors()
    # note: the number of aspect_ratios (3) is an important model parameter;
    # its role is significant and ties-in with other parameters configuring the
    # RPN model; together with the number of sizes specified, it determines the
    # number of base_anchors generated: 5 sizes * 3 aspect_ratios = 15
    # zero-centred base anchors; the reason we have 5 sets of 3 base anchors
    # is because each set of 3 will be used in relation to one of the 5
    # levels of the feature pyramid to generate real anchor boxes for that
    # level of the pyramid
    #
    # further, the number of base anchors determines the number of real anchor
    # boxes generated for each level of the feature pyramid
    #
    # the number of aspect_ratios (3) also appears to dove-tail with why the
    # cls_logits 1x1 convolution in the RPNHead has 3 out_channels, and,
    # hence, also with why the bbox_pred 1x1 convolution of the RPNHead as
    # 12 out_channels (3 * 4 = 12)

    # Generate the real anchor boxes.
    #
    # using the combinations (triples) of HxW grid_size, stride and the 3 base
    # anchors for each level of the feature pyramid (generated by the FPN),
    # generate a large set of real anchor boxes.
    #
    # this is done by first generating box centres for the anchor boxes
    # and then using these as offsets with respect to the zero-centered
    # base anchors; each anchor box has format [xmin, ymin, xmax, ymax]
    #
    # the grid sizes (HxW) of the feature maps in each level of the
    # feature pyramid, together with the number of base anchors for each
    # level of the pyramid (3), is what determines the number of anchors
    # generated with respect to each level of the feature pyramid;
    # specifically, the numbers of anchors generated per level of the pyramid
    # is as follows:
    #
    #   HpxWp x 3 =     507
    #   H3xW3 x 3 =   1,875
    #   H2xW2 x 3 =   7,500  
    #   H1xW1 x 3 =  30,000
    #   H0xW0 x 3 = 120,000
    #               -------
    #               159,882
    #
    # So, very deliberately, the anchor boxes have a strong relationship with
    # the grid cells of the feature maps of the levels of the feature pyramid.
    # Each grid cell is (effectively) related to 3 unique anchor boxes, each
    # with the same area but a different aspect ratio. Further, and even
    # more importantly, a similar but even tighter relationship exists between
    # the grid cells of the volumes output by the RPN Head and the anchor boxes.
    # The convolutions of the RPN Head, both for anchor box classification and
    # bbox delta regression, are designed to output volumes for each level of
    # the feature pyramid which preserve the HxW grid size of the feature maps
    # at the different levels of the feature pyramid. The volumes output
    # by the RPN Head for object/non-object (foreground/background classification
    # per anchor box have shapes that are IDENTICAL to the ones depicted above   
    # for the numbers of anchors generated.  This means there is a 1-to-1
    # mapping between the grid cells of the volumes output by the RPN Head
    # for anchor box classification and the anchor boxes themselves. The same
    # applies with respect to the bbox delta regression predictions output by
    # the RPN Head. Those volumes have the same shapes as depicted just above
    # with 3*4=12 channels instead of just 3, because each bbox delta
    # prediction needs 4 delta coordinates.  Each set of 4 delta coordinates
    # maps 1-to-1 to a unique anchor box and is used to calculate a
    # predicted bbox by applying the deltas to the corresponding anchor box.
    #
    # The generated anchor boxes are stored in a list of 5 tensors, where
    # each tensor holds the (different number of) anchor boxes generated for
    # the corresponding level of the feature pyramid.  List[torch.Tensor]
    anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
    # note: essentially, there are 3 anchor boxes associated with each cell
    # in the HxW grid of the feature maps at each level of the feature
    # pyramid; each of these 3 has the same area but a different aspect ratio   

    # Replicate the full set of anchors for as many times as there are images in
    # the current mini-batch so that we'll end up with one complete set of all
    # the anchors for each image.
    anchors = []  # List[List[torch.Tensor]]
    for i in range(len(image_list.image_sizes)):
        # build a fresh list of 5 tensors, each holding the anchors for the
        # feature maps of one level of the feature pyramid
        anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        anchors.append(anchors_in_image)

    # Now, for each image's list of anchor box tensors (one anchor tensor per
    # level of the feature pyramid), concatenate all the anchor boxes for the
    # different pyramid levels into one large tensor of anchor boxes, and
    # put this large concatenated tensor into a new list.  That is, create
    # a List[torch.Tensor], where each tensor in the list contains all of the
    # anchor boxes (for all levels of the feature pyramid). The list will have
    # as many elements as there are images, and all the elements are duplicates
    # of one another: a full set of all anchor boxes for each image.
    anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

    return anchors
```

### Reflection: How it is that the RPN learns bbox deltas

While learning RPN it was a mystery to me how it was that we could speak of the bbox regression portion of the RPN Head as learning and predicting bbox coordinate deltas (dx, dy, dw, dh). What is it that makes this so?  It felt like the answer was a bit like a "build it and they will come" type of explanation, with some magic somewhere that I couldn't yet grasp.  But, upon reflection, I've come to feel like the correct explanation is pretty much a "build it and they will come" type argument.

More concretely, I think the explanation lies in the combination of the following elements:
1. the model's outputs fit our prediction needs *structurally*
2. the *semantics* of the model's outputs (what turns the outputs into predictions, and of the precise type we want) is imposed upon the model by us, by how we use the model's outputs in our computations and, crucially, in our computation of loss  
3. the magic (awesome power) of *mathematical optimisation*, specifically, in our case, of *gradient descent-based loss function minimisation*

In other words, the only reason we can claim that the RPN predicts bbox coordinate deltas (dx, dy, dw, dh) in relation to specific anchor boxes, as desired, is because: 1) we've designed the RPN model to output what we need, structurally, 2) we use those outputs, from the outset, as if they were, in fact, what we want them to be (ie we impose our semantics upon the outputs), and 3) we then utilise mathematical optimisation to *make* those outputs *become* what we want them to be.

Clearly, these reflections pertain beyond RPN to connectionist AI generally. We are reflecting here on how it is that neural networks do what they do. We strongly suspect that these reflections represent a general recipe.


## The forward pass of the `RoIHeads` subnetwork (IP)

### Preliminary overview of RoI

The PyTorch torchvision implementation of the Faster R-CNN model appears to follow Lin (2017) for RoI pooling and final predictions, just as it did for the RPN component.
* see paper "Feature Pyramid Networks for Object Detection", Lin et al., 2017, in particular section 4.2 for RoI pooling

**RoI pooling**
Region-of-Interest (RoI) pooling is used to extract features. To use it with FPN, RoIs of different scales (sizes) are assigned to the different levels of a feature pyramid, since the feature maps at different levels have different sizes (resolutions). We assign an RoI of width $w$ and height $h$ to the level $P_k$ of the feature pyramid, where $k$ is defined per the following equation:
$$k = \lfloor k_0 + \log_2 (\sqrt{wh}/224) \rfloor$$
Lin (2017) uses $k_0 = 4$. So, if $wh = 224^2$, we have $\log_2(1) = 0$, and so $k=4$.  If the RoI's scale (size) becomes smaller (eg, say, 1/2 of 224), it should be mapped into a finer-resolution (lower) pyramid level (where the feature maps are larger). For example, if $wh = 112^2$, we have $\log_2(0.5) = -1$, and hence we get $k=3$.

NOTE: the equation above is Eq.(1) in the Lin (2017) FPN paper; it is implemented in class `LevelMapper` in module `poolers.py` in package  `torchvision.ops`.

Lin (2017) adapts RoI pooling to extract 7x7 feature maps.

**The RoI Head**
Multiple RoIs may be assigned to each level of the feature pyramid. An *RoI head* is applied to all RoIs of all pyramid levels. As in the RPN, the *RoI head* has just one set of (shared) parameters regardless of the pyramid levels of the RoIs.  

Lin's *RoI head* consists of 2 hidden 1024-dimension fully-connected (fc) layers (each followed by ReLU).  These layers are randomly initialised.  Lin et al. call their *RoI head* a "2-fc MLP head".  Hence the name of the class `TwoMLPHead` used in the PyTorch implementation of Faster R-CNN.

Lin (2017) applies the *RoI head* (`TwoMLPHead`) to the outputs of the RoI pooling to produce intermediate representations of the RoI-specific, pooled feature pyramid features prior to performing the final bbox regression and bbox classification steps which generate the final predictions.

**Bbox and object class prediction**
The final predictions are produced by sibling (ie dual) fully-connected bbox regression (`bbox_pred`) and bbox classification (`cls_score`) and layers. Each of these layers takes as input the intermediate representations output by the *RoI head* (`TwoMLPHead`).


### The FP of the `RoIHeads` module `roi_heads` (IP)

The forward pass of the RoI component of the Faster R-CNN ResNet50 FPN model is controlled by the `forward()` method of module (class) `RoIHeads`.

Here is a simplified and heavily commented version of the `forward()` method of class `RoIHeads`:
```Python
class RoIHeads(nn.Module):
  ...
  def forward(self,
              features,      # type: Dict[str, Tensor]
              proposals,     # type: List[Tensor]
              image_shapes,  # type: List[Tuple[int, int]]
              targets=None   # type: Optional[List[Dict[str, Tensor]]]
              ):
      # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
      """
      Args:
          features (List[Tensor])
            - the feature map volumes of the feature pyramid network
          proposals (List[Tensor[N, 4]])
            - the top-n best region proposals per image, as output by
            the RPN module; by default, in training mode, this is 2000
            per image; in test mode, it is 1000 per image; so, in test
            mode, N = 1000; the length of the List is num_images
          image_shapes (List[Tuple[H, W]])
          targets (List[Dict])
      """

      # TODO - explain what select_training_samples() does and why
      if self.training:
          proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
      else:
          labels = None
          regression_targets = None
          matched_idxs = None

      #
      # call the 'forward()' methods of the sequence of 3 component
      # modules that make up the RoIHeads module
      #

      # call the MultiScaleRoIAlign module to perform RoI
      # alignment and max pooling to produce a 7x7 feature map volume
      # of size [C,7,7] for each individual RoI, for each image;
      # (where, in our case, C=256);
      # - each proposal (candidate bbox prediction) output by the RPN
      #   module and received as input to this forward() method
      #   becomes an RoI
      # - so variable 'box_features' has shape [num_rois, C, 7, 7];
      #   by default, in test mode, num_rois = num_images * 1000
      box_features = self.box_roi_pool(features, proposals, image_shapes)

      # call the TwoMLPHead module to learn intermediate representations
      # of the bbox features from the [C,7,7] feature map volumes for
      # each RoI, for each image, using two fully-connected layers;
      # - the variable 'box_features' that is returned now has shape
      #   [num_rois, 1024]
      box_features = self.box_head(box_features)

      # call the FastRCNNPredictor module to make the final bbox
      # (regression) predictions and the final object class
      # (classification) predictions for each RoI, for each image
      # - variable class_logits has shape [num_rois, num_classes]
      # - variable box_regression has shape [num_rois, num_classes * 4]
      class_logits, box_regression = self.box_predictor(box_features)

      # perform post-processing:
      # - if in training mode, calculate the two RoI losses
      # - if in test (inference) mode, select the final 'best' set
      #   of bbox predictions and their associated object class
      #   predictions and object class prediction confidence scores
      #
      # TODO post-processing of the detections:
      # -
      result: List[Dict[str, torch.Tensor]] = []
      losses = {}
      if self.training:
          assert labels is not None and regression_targets is not None
          loss_classifier, loss_box_reg = fastrcnn_loss(
              class_logits, box_regression, labels, regression_targets)
          losses = {
              "loss_classifier": loss_classifier,
              "loss_box_reg": loss_box_reg
          }
      else:
          # class_logits shape: Tensor[num_rois, num_classes]
          # box_regression shape: Tensor[num_rois, num_classes * 4]
          # proposals: List[Tensor[num_rois_per_img, 4]], len=num_images
          boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
          # assemble the post-processed results (final predictions per
          # image) into a list of dictionaries
          num_images = len(boxes)
          for i in range(num_images):
              result.append(
                  {
                      "boxes": boxes[i],
                      "labels": labels[i],
                      "scores": scores[i],
                  }
              )

      # in training mode, return the losses;
      # in test (inference) mode, return the predictions
      return result, losses
```


### The FP of the `MultiScaleRoIAlign` module `box_roi_pool`

Here is a heavily commented version of the `forward()` method of class `MultiScaleRoIAlign`:
```Python
class MultiScaleRoIAlign(nn.Module):
    def forward(self, x, boxes, image_shapes):
        """
        Args:
          x - OrderedDict[Tensor] - the feature maps for all levels of the
              feature pyramid (all with 256 channels but different HxW)
              ```
                   pool P6 CxHxW (256, Hp, Wp)  
                      3 P5 CxHxW (256, H3, W3)
                      2 P4 CxHxW (256, H2, W2)
                      1 P3 CxHxW (256, H1, W1)
                      0 P2 CxHxW (256, H0, W0)
              ```
          boxes - List[Tensor[N, 4]] - proposal bbox predictions; (by default, in test mode, 1000 per image; in training mode, 2000 per image)
          image_shapes - List[Tuple[H, W]] - the size of each image after
                         being transformed, but prior to being zero-padded
                         to a common size
        """
        # extract the feature map tensors (P2, P3, P4, P5) for the
        # first 4 levels of the feature pyramid: levels '0','1','2','3';
        # note that we ignore the top-most 'pool' level of the feature
        # pyramid here for some reason
        x_filtered = []
        for k, v in x.items():
            if k in self.featmap_names: # ['0','1','2','3']
                x_filtered.append(v)

        num_levels = len(x_filtered)    # num_levels = 4

        # convert the proposal bboxes to RoI format:
        # - concatenate all bboxes for all images into a single tensor, but
        #   prefix each box with a new, additional coordinate that holds a
        #   sequential index that maps each box to its associated image
        # - in other words, bbox [x1, y1, x2, y2] for image k
        #   becomes RoI [k, x1, y1, x2, y2] in one long [N, 5] tensor of RoIs
        # IMPORTANT: the RoIs are simply the bboxes (the proposals
        # output by the RPN subnetwork) with a minor format adjustment;
        # there's no change in the number of proposal bboxes or in the bbox
        # coordinates; we're simply changing a list of tensors holding boxes
        # (one tensor per image) into one tensor holding all boxes for
        # all images
        rois = self.convert_to_roi_format(boxes)
        # len(rois) = 1000 per image (in test/inference mode)

        # call method `setup_scales()` to:
        # 1) calculate scaling factors and set these to attribute
        #    self.scales
        # 2) instantiate a LevelMapper and set this to attribute
        #    self.map_levels
        #
        # the term 'scales' means 'scaling factors', to be used
        # for scaling (resizing) proposal bboxes (RoIs) for specific
        # levels of the feature pyramid (whose feature maps have
        # different sizes/resolutions)
        #
        # steps for calculating the scaling factors:
        # - find largest H and largest W amongst the sizes of all the images;
        #   call these Hi and Wi
        # - get the H and W of feature maps at each level of the feature
        #   pyramid; call these Hf and Wf
        # - using the Hf and Wf of the feature maps at each level of the feature
        #   pyramid, calculate:
        #     - Hratio = Hf / Hi; Wratio = Wf / Wi
        #     - scale1 = 2 ** (torch.tensor(Hratio).log2().round())
        #     - scale2 = 2 ** (torch.tensor(Wratio).log2().round())
        #     - assert(scale1 == scale2)
        #     - return scale1
        if self.scales is None:
            self.setup_scales(x_filtered, image_shapes)

        scales = self.scales
        assert scales is not None
        # for a single image with original shape 485x1000, and size
        # 646 x 1333 after resizing in transform() but prior to
        # zero-padding, we get scales (ie scaling factors):
        # [0.25, 0.125, 0.0625, 0.03125]
        # - each scaling factor maps to one level of the feature pyramid
        # - scale 0.25 maps to level 0, 0.125 to level 1, etc.
        # - the scales will be used to scale (resize) RoIs (proposed
        #   bboxes) to better match the dimensions of the feature maps
        #   at the different levels of the feature pyramid

        if num_levels == 1:
            return roi_align(
                x_filtered[0], rois,
                output_size=self.output_size,
                spatial_scale=scales[0],
                sampling_ratio=self.sampling_ratio
            )

        mapper = self.map_levels
        assert mapper is not None

        # call the LevelMapper to map each proposal bbox to a
        # particular level of the feature pyramid (0,1,2,3):
        # - a LevelMapper is simply an implementation of Eq. (1), from
        #   the Lin (2017) FPN paper (see above), for computing 'k',
        #   the most appropriate feature pyramid level for that bbox
        #   given its area (HxW)
        levels = mapper(boxes)  
        # note:
        # - 'levels' is tensor the same length as the total number of boxes,
        #   and the same length as the number of RoIs in 'rois'
        # - in test mode, for one image we have 1000 RoIs; these get
        #   mapped to the 4 levels of the feature pyramid,
        # - 'levels' is just a squence of feature pyramid level numbers, 0,1,2,3
        # - because there is a 1-to-1 mapping between boxes and RoIs, the
        #   pyramid level indices in 'levels' also map RoIs to levels of the
        #   feature pyramid; and this is what it will be used for, below

        # num_rois is the same as len(levels) and the total number of boxes
        num_rois = len(rois)
        # for a single image input to Faster R-CNN, I got 1000 RoIs

        num_channels = x_filtered[0].shape[1] # 256

        # initialise a 4D tensor to hold the results of the RoI
        # max pooling that will happen shortly
        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = torch.zeros(
            (num_rois, num_channels,) + self.output_size,
            dtype=dtype,
            device=device,
        )
        # self.output_size is (7, 7), so tensor 'result' has shape
        # [num_rois, num_channels, 7, 7] = [num_rois, 256, 7, 7];
        # where, by default, in test model, there are 1000 RoIs per
        # image; so num_rois is 1000 * the number of images in the
        # mini-batch

        # iterate through the levels of the feature pyramid and, for
        # each level, perform the following:
        # - get the RoIs mapped to that level
        # - rescale the RoIs for feature map dimensions at that level
        # - resize, crop and position each RoI relative to the feature
        #   map volume at that level, and partition the RoI into a
        #   7x7 non-overlapping grid;
        #   - note: it is this step that motivates the use of the
        #     term 'align'; the rescaled and adjusted RoI is 'aligned'
        #     (ie correctly positioned) relative to the HxW dimensions
        #     of the feature map volume of a certain level of the  
        #     feature pyramid
        # - for each cell of the 7x7 partitioned RoI, perform max pooling  
        #   against the feature map volume, producing an output volume
        #   with shape Cx7x7, where C is the number of channels in the
        #   feature map volume (ie 256)
        #
        tracing_results = []
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):

            # get the RoIs mapped to the current level of the feature pyramid
            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]

            # perform RoI alignment & pooling for the RoIs that map
            # to the current level of the feature pyramid
            #
            # IMPORTANT: the actual RoI alignment & pooling algorithm is
            # NOT implemented in Python inside function roi_align();
            # instead, function roi_align() calls
            # torch.ops.torchvision.roi_align() which causes a C++ function
            # to be bound into Python and then called; so we can't see the
            # actual algorithm to figure out exactly what it's doing
            # and how it's doing it; if one looks in the 'torch' package
            # directory for an 'ops' package directory, there is none;
            # that's because the 'ops' namespace is a virtual kind of thing
            # that's created dynamically; for info on all this magic stuff,
            # see module '_ops.py' in package 'torch' which explains how
            # the magic 'ops' namespace works and how a call such as
            # torch.ops.torchvision.roi_align() really works
            #
            # Per INM705 lecture slides for Week 4, RoI alignment & pooling
            # involves the following steps:
            # - rescale all of the RoIs (proposal bboxes, which are of
            #   different sizes) for a given level of the feature pyramid
            #   using the scaling factor calculated for that level of the
            #   feature pyramid
            # - for each rescaled RoI, resize it, crop it (if required)
            #   and position it relative to the HxW dimensions of the
            #   feature map volume for that level of the feature pyramid
            # - split the RoI into a 7x7 non-overlapping grid  
            # - perform max pooling against the feature map volume for
            #   each cell in the 7x7 cell grid of the RoI, to produce
            #   a 'batch' of 7x7 feature maps with the same number
            #   channels as in the feature map volume of the feature
            #   pyramid (ie 256)
            result_idx_in_level = roi_align(
                per_level_feature, rois_per_level,
                output_size=self.output_size,
                spatial_scale=scale, sampling_ratio=self.sampling_ratio)
            # variable 'result_idx_in_level' has shape
            # [len(rois_per_level), 256, 7, 7];
            # this means we get back one feature map volumn of size
            # [256,7,7] for each RoI

            if torchvision._is_tracing():
                tracing_results.append(result_idx_in_level.to(dtype))
            else:
                # copy the results of the RoI alignment and pooling
                # for the current level of the feature pyramid into the
                # master results tensor, 'result'
                #
                # note: the dtype of 'result' is based on x_filtered[0];
                # but result_idx_in_level's dtype is based on all levels  
                # of x_filtered, 0,1,2,3, which are based on tensors originally
                # output by different layers of the backbone;
                # When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, the result's dtype
                # is defensively cast before copying elements from
                # result_idx_in_level, in the following line of code.
                # (The casting is manual (can't rely on autocast to cast
                # for us) because our line of code acts on 'result'
                # in-place, and autocast only affects out-of-place
                # operations.)
                result[idx_in_level] = result_idx_in_level.to(result.dtype)

        if torchvision._is_tracing():
            result = _onnx_merge_levels(levels, tracing_results)

        return result  # tensor of shape [num_rois, 256, 7, 7]
```


### The FP of the `TwoMLPHead` module `box_head`

Here is a heavily commented version of the `forward()` method of class `TwoMLPHead`:
```Python
class TwoMLPHead(nn.Module):
    ...
    def forward(self, x):
        # input x has shape [num_rois, C, 7, 7], where, in our
        # case, C=256; in inference mode, by default, there
        # are 1000 RoIs per image in the mini-batch;
        # so num_rois = num_images*1000

        # flatten each of the [C,7,7] feature map volumes so we can
        # pass them into the first of the two flat, fully-connected
        # layers; 256x7x7 = 12544
        x = x.flatten(start_dim=1)
        # so x now has shape [num_rois, 12544]

        # learn flat, 1024-dimension intermediate representations of
        # the [C,7,7] feature map volumes produced by the preceding
        # RoI alignment and max-pooling operation, which now have
        # shape [num_rois, 12544]
        x = F.relu(self.fc6(x))  # output: x.shape = [num_rois, 1024]
        x = F.relu(self.fc7(x))  # output: x.shape = [num_rois, 1024]

        # return the flat, intermediate representation of each RoI
        # feature map volume; [num_rois, 1024]
        return x    
```


### The FP of the `FastRCNNPredictor` module `box_predictor`

Here is a simplified and heavily commented version of the `forward()` method of class `FastRCNNPredictor`:
```Python
class FastRCNNPredictor(nn.Module):
    ...
    def forward(self, x):
        # input x has shape [num_rois, 1024];
        # where, by default, in test (inference) mode,
        # num_rois = num_images * 1000

        # observe that:
        # - the two layers here, self.cls_score and self.bbox_pred,
        #   are the output layers of the  Faster R-CNN model
        # - these two layers are 'dual' outputs; ie they are
        #   siblings; they both receive the same input and they
        #   are independent of each other, each producing their
        #   own predictions

        scores = self.cls_score(x)
        # scores.shape [num_rois, num_classes];
        # these are the object class classification logits (scores)
        # corresponding to the bbox predictions output by the sibling
        # layer

        bbox_deltas = self.bbox_pred(x)
        # bbox_deltas.shape [num_rois, num_classes * 4]
        # note 1: it's not at all clear why the variable 'bbox_deltas'
        # has such a name, ie one that refers to 'deltas'; at this point
        # these are final bbox predictions; they are not offsets to be
        # applied against something else in order to produce derived   
        # results  
        # note 2: the vectors of dimension [num_classes * 4] that are
        # output here will have values (activations) in every element;
        # but it's only the values (activations) in the 4 elements that
        # correspond to the element in the corresponding output vector
        # in 'scores' that has the highest positive value (largest logit,
        # or largest score) that will be recognised as being 'the
        # predicted bbox'

        # note: the variables 'scores' and 'bbox_deltas' contain the  
        # outputs from the dual, final layers of the Faster R-CNN
        # model; but there are still far too many bbox predictions
        # per image (in test mode, by default, 1000 per image) for these
        # to be the final outputs that are actually returned to the caller
        # of the model;
        #
        # there is still some post-processing to be done to select, from
        # amongst the 1000 bbox predictions per image, only the very
        # best predictions --- ie the ones with confidence scores (not
        # simply score logits) that exceed the user-defined threshold
        # specified in model parameter 'box_score_thresh'
        #
        # refer back to the 'forward()' method of module RoIHeads
        # (presented above) to learn about the post-processing
        # that is applied to the outputs that are being returned here

        return scores, bbox_deltas    
```


asdf

'''Makes available the Txt2Mask class, which assists in the automatic
assignment of masks via text prompt using clipseg.

Here is typical usage:
    
    from ldm.invoke.txt2mask import Txt2Mask, SegmentedGrayscale
    from PIL import Image

    txt2mask = Txt2Mask(self.device)
    segmented = txt2mask.segment(Image.open('/path/to/img.png'),'a bagel')
    
    # this will return a grayscale Image of the segmented data
    grayscale = segmented.to_grayscale()

    # this will return a semi-transparent image in which the
    # selected object(s) are opaque and the rest is at various
    # levels of transparency
    transparent = segmented.to_transparent()

    # this will return a masked image suitable for use in inpainting:
    mask = segmented.to_mask(threshold=0.5)

The threshold used in the call to to_mask() selects pixels for use in
the mask that exceed the indicated confidence threshold. Values range
from 0.0 to 1.0. The higher the threshold, the more confident the
algorithm is. In limited testing, I have found that values around 0.5
work fine.
'''

import torch
import numpy as  np
from models.clipseg import CLIPDensePredT
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms

CLIP_VERSION = 'ViT-B/16'
CLIPSEG_WEIGHTS = 'src/clipseg/weights/rd64-uni.pth'

class SegmentedGrayscale(object):
    def __init__(self, image:Image, heatmap:torch.Tensor):
        self.heatmap = heatmap
        self.image = image
        
    def to_grayscale(self)->Image:
        return Image.fromarray(np.uint8(self.heatmap*255))

    def to_mask(self,threshold:float=0.5)->Image:
        discrete_heatmap = self.heatmap.lt(threshold).int()
        return Image.fromarray(np.uint8(discrete_heatmap*255),mode='L')

    def to_transparent(self)->Image:
        transparent_image = self.image.copy()
        transparent_image.putalpha(self.to_image)
        return transparent_image

class Txt2Mask(object):
    '''
    Create new Txt2Mask object. The optional device argument can be one of
    'cuda', 'mps' or 'cpu'.
    '''
    def __init__(self,device='cpu'):
        print('>> Initializing clipseg model')
        self.model = CLIPDensePredT(version=CLIP_VERSION, reduce_dim=64, )
        self.model.eval()
        self.model.to(device)
        self.model.load_state_dict(torch.load(CLIPSEG_WEIGHTS, map_location=torch.device(device)), strict=False)

    @torch.no_grad()
    def segment(self, image:Image, prompt:str) -> SegmentedGrayscale:
        '''
        Given a prompt string such as "a bagel", tries to identify the object in the
        provided image and returns a SegmentedGrayscale object in which the brighter
        pixels indicate where the object is inferred to be.
        '''
        prompts = [prompt]   # right now we operate on just a single prompt at a time

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((image.width, image.height)), # must be multiple of 64...
        ])
        img = transform(image).unsqueeze(0)
        preds = self.model(img.repeat(len(prompts),1,1,1), prompts)[0]
        heatmap = torch.sigmoid(preds[0][0]).cpu()
        return SegmentedGrayscale(image, heatmap)



        

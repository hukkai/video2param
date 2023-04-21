from models.blip2_model import ImageCaptioning
from models.grit_model import DenseCaptioning
from models.gpt_model import ImageToText
from models.region_semantic import RegionSemantic
from mmengine import load, dump
from utils.util import read_image_width_height, display_images_and_text, resize_long_edge
import argparse
from PIL import Image
import os
import os.path as osp
from tqdm import tqdm
try:
    from mmcv import load, dump
except:
    from mmengine import load, dump

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', dest='verbose', default=False)
parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo', 'gpt4'], default='gpt-3.5-turbo')
parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP2 Image Caption')
parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
parser.add_argument('--semantic_segment', action='store_true', dest='semantic_segment', default=True, help='Set this flag to True if you want to use semantic segmentation')
parser.add_argument('--sam_arch', choices=['vit_b', 'vit_l', 'vit_h'], dest='sam_arch', default='vit_h', help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models')
parser.add_argument('--captioner_base_model', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip2', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
parser.add_argument('--region_classify_model', choices=['ssa', 'edit_anything'], dest='region_classify_model', default='edit_anything', help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.')
parser.add_argument('--device', default=None, help="if set, will override all other device arguments")
parser.add_argument('--image_caption_device', default='cuda:2', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
parser.add_argument('--dense_caption_device', default='cuda:2', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
parser.add_argument('--semantic_segment_device', default='cuda:2', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended. Make sue this model and image_caption model on same device.')

os.environ['OPENAI_KEY'] = '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'

class ImageTextTransformation:
    def __init__(self, args):
        # Load your big model here
        self.args = args
        self.verbose = args.verbose
        if args.device is not None:
            assert 'cuda' in args.device
            args.image_caption_device = args.device
            args.dense_caption_device = args.device
            args.semantic_segment_device = args.device
        self.init_models()
    
    def init_models(self):
        openai_key = os.environ['OPENAI_KEY']
        print(self.args)
        print('\033[1;34m' + "Welcome to the Image2Paragraph toolbox...".center(50, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(50, '-') + '\033[0m')
        print('\033[1;31m' + "This is time-consuming, please wait...".center(50, '-') + '\033[0m')
        self.image_caption_model = ImageCaptioning(device=self.args.image_caption_device, captioner_base_model=self.args.captioner_base_model, verbose=self.verbose)
        self.dense_caption_model = DenseCaptioning(device=self.args.dense_caption_device, verbose=self.verbose)
        self.gpt_model = ImageToText(openai_key)
        self.region_semantic_model = RegionSemantic(
            device=self.args.semantic_segment_device, 
            image_caption_model=self.image_caption_model, 
            region_classify_model=self.args.region_classify_model, 
            sam_arch=self.args.sam_arch, 
            verbose=self.verbose)
        print('\033[1;32m' + "Model initialization finished!".center(50, '-') + '\033[0m')
        
    def boxstr_to_box(self, boxstr, old_shape, old_format='xyxy'):
        boxstr = boxstr[1:-1]
        box = [float(x) for x in boxstr.split(", ")]
        if old_format == 'xywh':
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        # Since Dense caption & Region semantic model re-scale long edge to 384, we map it back here
        ratio = max(old_shape) / 384
        box = [int(ratio * x) for x in box]
        return box
        
    def image_to_info(self, img_src):
        width, height = read_image_width_height(img_src)
        shape = (width, height)
        # print(self.args)
        if self.args.image_caption:
            image_caption = self.image_caption_model.image_caption(img_src)
        else:
            image_caption = " "
        if self.args.dense_caption:
            dense_caption_str = self.dense_caption_model.image_dense_caption(img_src)
            dense_caption = dense_caption_str.split("; ") 
            if ":" not in dense_caption[-1]:
                dense_caption = dense_caption[:-1]
            dense_caption = [x.split(": ") for x in dense_caption]
            dense_caption = {x[0]: self.boxstr_to_box(x[1], shape) for x in dense_caption}
        else:
            dense_caption_str = " "
            dense_caption = {}
        if self.args.semantic_segment:
            region_semantic_str = self.region_semantic_model.region_semantic(img_src)
            region_semantic = region_semantic_str.split("; ") 
            if ":" not in region_semantic[-1]:
                region_semantic = region_semantic[:-1]
            region_semantic = [x.split(": ") for x in region_semantic]
            region_semantic = {x[0]: self.boxstr_to_box(x[1], shape, 'xywh') for x in region_semantic}
        else:
            region_semantic_str = " "
            region_semantic = {}
        return dict(
            shape=(width, height), caption=image_caption, dense_caption=dense_caption, region_semantic=region_semantic, 
            note="For dense caption and region_semantic, xyxy bounding box is returned")
    
    def image_to_text(self, img_src):
        img_info = self.image_to_info(img_src)
        image_caption = img_info['caption']
        dense_caption = img_info['dense_caption_str']
        region_semantic = img_info['region_semantic_str']
        width, height = img_info['shape']
        generated_text = self.gpt_model.paragraph_summary_with_gpt(image_caption, dense_caption, region_semantic, width, height)
        return generated_text
    
    # Predict the img_info for a list of image, and save the result in 'dest'
    def forward_img_list(self, img_list, dest='result.pkl', img_root=None):
        print('\033[1;34m' + f"Will process {len(img_list)} images. ".center(50, '-') + '\033[0m')
        info_list = []
        for im in tqdm(img_list):
            img_path = osp.join(img_root, im) if img_root else im
            info = self.image_to_info(img_path)
            info['file_name'] = im
            info_list.append(info)
        dump(info_list, dest)

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = 'cuda:2'
    model = ImageTextTransformation(args)
    img_list = ['1.jpg', '2.png', '3.jpg', '4.jpg', '5.jpg']
    img_root = 'examples'
    dest = 'result.pkl'
    model.forward_img_list(img_list, img_root=img_root, dest=dest)

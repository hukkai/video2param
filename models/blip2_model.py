from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration
import torch
from utils.util import resize_long_edge


class ImageCaptioning:
    def __init__(self, device, captioner_base_model='blip', verbose=False):
        self.verbose = verbose
        self.device = device
        self.captioner_base_model = captioner_base_model
        self.processor, self.model = self.initialize_model()

    def initialize_model(self,):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        if self.captioner_base_model == 'blip2':
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=self.data_type
            )
        # for gpu with small memory
        elif self.captioner_base_model == 'blip':
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=self.data_type)
        else:
            raise ValueError('arch not supported')
        model.to(self.device)
        return processor, model

    def image_caption(self, image_list):
        if type(image_list) == 'str':
            image_list = [image_list]
            single_input = True
        else:
            single_input = False
        images = [Image.open(image) for image in image_list]
        images = [resize_long_edge(image, 384) for image in images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = [text.strip() for text in generated_texts]
        if self.verbose:
            print('\033[1;35m' + '*' * 100 + '\033[0m')
            print('\nStep1, BLIP2 caption:')
            for idx, text in generated_text:
                print(f'Frame {idx} caption: {text}')
            print('\033[1;35m' + '*' * 100 + '\033[0m')
        if single_input:
            return generated_texts[0]
        return generated_texts

    def video_caption(self, video):
        images = [Image.fromarray(frame.permute(1,2,0).numpy()) for frame in video]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = [text.strip() for text in generated_texts]
        if self.verbose:
            print('\033[1;35m' + '*' * 100 + '\033[0m')
            print('\nStep1, BLIP2 caption:')
            for idx, text in generated_text:
                print(f'Frame {idx} caption: {text}')
            print('\033[1;35m' + '*' * 100 + '\033[0m')
        return generated_texts
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."
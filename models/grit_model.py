import os
from models.grit_src.image_dense_captions import image_caption_api, video_caption_api

class DenseCaptioning():
    def __init__(self, device, verbose=False):
        self.verbose = verbose
        self.device = device


    def initialize_model(self):
        pass

    def image_dense_caption_debug(self, image_src):
        dense_caption = """
        1. the broccoli is green, [0, 0, 333, 325]; 
        2. a piece of broccoli, [0, 147, 143, 324]; 
        3. silver fork on plate, [4, 547, 252, 612];
        """
        return dense_caption
    
    def image_dense_caption(self, image_src):
        dense_caption = image_caption_api(image_src, self.device)
        if self.verbose:
            print('\033[1;35m' + '*' * 100 + '\033[0m')
            print("Step2, Dense Caption:\n")
            print(dense_caption)
            print('\033[1;35m' + '*' * 100 + '\033[0m')
        return dense_caption

    def video_dense_caption(self, video):
        # video (torch.tensor): T, C, H, W
        dense_caption = video_caption_api(video, self.device)
        if self.verbose:
            print('\033[1;35m' + '*' * 100 + '\033[0m')
            print("Step2, Dense Caption:\n")
            print(dense_caption)
            print('\033[1;35m' + '*' * 100 + '\033[0m')
        return dense_caption
    
    
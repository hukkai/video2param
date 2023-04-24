import decord
import torch

from models.blip2_model import ImageCaptioning
from models.grit_model import DenseCaptioning


def read_video(video_path, sampled_frames=10, longer_side=384):
    reader = decord.VideoReader(video_path)
    num_frames = len(reader)
    assert sampled_frames <= num_frames
    offset = num_frames / sampled_frames
    index = torch.arange(sampled_frames).mul(offset)
    index = index.add(offset / 2).to(torch.int)
    index = index.clamp(0, num_frames - 1).numpy()
    video = reader.get_batch(index).asnumpy()
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    T, C, H, W = video.shape
    scale_factor = longer_side / max(H, W)
    video = torch.nn.functional.interpolate(
        video.float(), 
        scale_factor=scale_factor,
        mode='bilinear')
    video = video.to(torch.uint8)
    return video


image_caption_model = ImageCaptioning(device='cuda:1', captioner_base_model='blip2')
dense_caption_model = DenseCaptioning(device='cuda:2')

image_caption = image_caption_model.video_caption(video)
dense_caption = dense_caption_model.video_dense_caption(video)
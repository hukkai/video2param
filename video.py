import decord
import torch

from models.blip2_model import ImageCaptioning
from models.grit_model import DenseCaptioning
from models.gpt_model_video import ImageToText


openai_key = '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'

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
    return video, (W, H), index, num_frames

def boxstr_to_box(boxstr, old_shape, old_format='xyxy'):
    boxstr = boxstr[1:-1]
    box = [float(x) for x in boxstr.split(", ")]
    if old_format == 'xywh':
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
    # Since Dense caption & Region semantic model re-scale long edge to 384, we map it back here
    ratio = max(old_shape) / 384
    box = [int(ratio * x) for x in box]
    return box

def parse_dense_caption(dense_caption):
    for idx, caption in enumerate(dense_caption):
        caption = caption.split("; ") 
        if ":" not in caption[-1]:
            caption = caption[:-1]
        caption = [x.split(": ") for x in caption]
        caption = {x[0]: boxstr_to_box(x[1], (W, H)) for x in caption}
        dense_caption[idx] = caption
    return dense_caption



image_caption_model = ImageCaptioning(device='cuda:1', captioner_base_model='blip2')
dense_caption_model = DenseCaptioning(device='cuda:2')
gpt_model = ImageToText(openai_key)

video, (W, H), index, num_frames = read_video('cxk.mp4', sampled_frames=5)
image_caption = image_caption_model.video_caption(video)
dense_caption = dense_caption_model.video_dense_caption(video)
dense_caption = parse_dense_caption(dense_caption)
generated_text = gpt_model.paragraph_summary_with_gpt(num_frames, W, H, index, image_caption, dense_caption, True)
print(generated_text)


import openai

def get_question(num_frames, width, height, index, captions, dense_captions, one_person=True):
    head = f"There is a {num_frames / 30: .1f}-second video and the resolution is {width}X{height}\n"
    head += 'I have used a computer vision model to extract information at some timestamps from the video:\n\n'

    for idx, caption, dense_caption in zip(index, captions, dense_captions):
        summary = f"At around {idx/30:.1f} seconds, the image caption is '{caption}' and the dense_caption is {dense_caption}.\n"
        head += summary

    if one_person:
        head += "\nThe individual appearing in various frames is the same person.\n"

    head += """\n Generate only an informative and nature paragraph based on the given information. There are some rules:
    Use nouns rather than coordinates to show position information of each object.
    Use sequencing words or transition words rather than frame indexes to show temporal information in the video.
    No more than 7 sentences.
    Only use one paragraph.
    Do not appear any number.
    Do not appear any specific time.
    Most importantly, pretend you watched the video instead of someone giving you the information.
    """
    return head

class ImageToText:
    def __init__(self, api_key, gpt_version="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.gpt_version = gpt_version
    
    def paragraph_summary_with_gpt(self, num_frames, width, height, index, captions, dense_captions, one_person=False):
        question = get_question(num_frames, width, height, index, captions, dense_captions, one_person)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep4, Paragraph Summary with GPT-3:')
        print('\033[1;34m' + "Question:".ljust(10) + '\033[1;36m' + question + '\033[0m')
        completion = openai.ChatCompletion.create(
            model=self.gpt_version, 
            messages = [
            {"role": "user", "content" : question}]
        )

        print('\033[1;34m' + "ChatGPT Response:".ljust(18) + '\033[1;32m' + completion['choices'][0]['message']['content'] + '\033[0m')
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return completion['choices'][0]['message']['content']

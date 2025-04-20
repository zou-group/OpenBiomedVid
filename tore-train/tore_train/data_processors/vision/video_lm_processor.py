from dataclasses import dataclass, field
from datasets import Dataset
import numpy as np
import json
from transformers import AutoProcessor
import numpy as np


@dataclass
class VideoLMArguments:
    processor_name_or_path: str = field(
        default=None,
        metadata={"help": "The name or path of the processor to use."},
    )
    video_token: str = field(
        default=None,
        metadata={"help": "The video token to use."},
    )
    apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to apply chat template."},
    )
    to_numpy: bool = field(
        default=True,
        metadata={"help": "Whether to convert the output to numpy array."},
    )
    num_proc: int = field(
        default=16,
        metadata={"help": "The number of processes to use."},
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."},
    )


class VideoLMProcessor:
    input_columns = ["text", "video_metas"]
    output_columns = ["input_ids"]
    arguments = VideoLMArguments

    def __init__(self, config: VideoLMArguments):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.processor_name_or_path)
        self.video_token = config.video_token

    def precheck(self, dataset: Dataset):
        """
        Precheck the dataset to ensure it is valid for VideoLMProcessor
        """
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        if not self.config.apply_chat_template and "text" not in dataset.column_names:
            raise ValueError(
                f"Invalid dataset: `text` field not found with apply_chat_template=False"
            )
        elif self.config.apply_chat_template and "messages" not in dataset.column_names:
            raise ValueError(
                f"Invalid dataset: `messages` field not found with apply_chat_template=True"
            )

        if "video_metas" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `video_metas` field not found")

    def process_example(self, example: dict) -> dict:
        """Process a single example from the dataset.

        This method processes text and images from an example to create model inputs:
        1. If apply_chat_template is True, applies chat template to messages
        2. Otherwise uses raw text
        3. Replaces image/video tokens with processor-specific tokens
        4. Creates fake videos based on provided video metas
        5. Validates number of video tokens matches number of videos
        6. Processes text and fake videos through the model processor
        7. Returns processed input IDs

        Args:
            example (dict): Dictionary containing example data with fields:
                - messages (list): If apply_chat_template=True, list of chat messages
                - text (str): If apply_chat_template=False, raw text
                - video_metas (str): JSON string of video metas. Each video meta is a dict with keys:
                    - num_frames (int): Number of frames in the video
                    - resolution (list or dict): Resolution of the video. If list, [width, height]. If dict, {"width": width, "height": height}
                    - fps (int): Sampling frequency of the video

        Returns:
            dict: Processed example with added 'input_ids' field
        """
        if self.config.apply_chat_template:
            assert "messages" in example.keys(), "messages field not found in example"
            text = self.processor.apply_chat_template(example["messages"])

        else:
            text = example["text"]

        if self.video_token is not None:
            text = text.replace(self.video_token, self.processor.video_token)

        example["text"] = text

        video_metas = json.loads(example["video_metas"])

        # Create fake images based on the sizes
        fake_videos = []
        for meta in video_metas:
            num_frames, resolution, fps = (
                meta["num_frames"],
                meta["resolution"],
                meta.get("fps", 1),
            )
            # Load resolution
            if isinstance(resolution, (list, tuple)):
                width, height = resolution
            elif isinstance(resolution, dict):
                width, height = resolution["width"], resolution["height"]
            else:
                raise ValueError(f"Unknown resolution format: {resolution}")
            # Create a random array with the specified dimensions
            fake_video = np.zeros((num_frames, 3, height, width))  # 3 channels for RGB
            fake_videos.append(fake_video)

        # Count the number of image tokens in the text
        num_video_tokens = text.count(self.processor.video_token)
        # Count the number of fake images
        num_fake_videos = len(fake_videos)
        # Assert they are equal
        assert (
            num_video_tokens == num_fake_videos
        ), f"Number of video tokens ({num_video_tokens}) does not match number of fake videos ({num_fake_videos})"

    
        if len(fake_videos) == 0:
            fake_videos = None

        # Call the processor
        output = self.processor(
            images=None,
            videos=fake_videos,
            text=text,
            do_normalize=False,
            do_rescale=False,
        )

        # Convert to numpy array if to_numpy is True
        if self.config.to_numpy:
            example["input_ids"] = np.array(output["input_ids"][0], dtype=np.int32)
        else:
            example["input_ids"] = output["input_ids"][0]

        return example

    def __call__(self, dataset: Dataset):
        self.precheck(dataset)
        dataset = dataset.map(
            self.process_example,
            num_proc=self.config.num_proc,
            keep_in_memory=self.config.keep_in_memory,
        )
        return dataset

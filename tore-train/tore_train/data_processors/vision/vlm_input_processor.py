from dataclasses import dataclass, field
from datasets import Dataset
import numpy as np
import json
from transformers import AutoProcessor
import numpy as np


@dataclass
class VLMInputArguments:
    processor_name_or_path: str = field(
        default=None,
        metadata={"help": "The name or path of the processor to use."},
    )
    image_token: str = field(
        default=None,
        metadata={"help": "The image token to use."},
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


class VLMInputProcessor:
    input_columns = ["text", "images_sizes"]
    output_columns = ["input_ids"]
    arguments = VLMInputArguments

    def __init__(self, config: VLMInputArguments):
        self.config = config
        self.processor = AutoProcessor.from_pretrained(config.processor_name_or_path)
        self.image_token = config.image_token
        self.video_token = config.video_token

    def precheck(self, dataset: Dataset):
        """
        Precheck the dataset to ensure it is valid for VLMInputProcessor
        """
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        if not self.config.apply_chat_template and "text" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `text` field not found with apply_chat_template=False")
        elif self.config.apply_chat_template and "messages" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `messages` field not found with apply_chat_template=True")

        if "images_sizes" not in dataset.column_names:
            raise ValueError(f"Invalid dataset: `images_sizes` field not found")

    def process_example(self, example: dict) -> dict:
        """Process a single example from the dataset.

        This method processes text and images from an example to create model inputs:
        1. If apply_chat_template is True, applies chat template to messages
        2. Otherwise uses raw text
        3. Replaces image/video tokens with processor-specific tokens
        4. Creates fake images based on provided image sizes
        5. Validates number of image tokens matches number of images
        6. Processes text and fake images through the model processor
        7. Returns processed input IDs

        Args:
            example (dict): Dictionary containing example data with fields:
                - messages (list): If apply_chat_template=True, list of chat messages
                - text (str): If apply_chat_template=False, raw text
                - images_sizes (list): List of [width, height] for each image

        Returns:
            dict: Processed example with added 'input_ids' field
        """
        if self.config.apply_chat_template:
            assert "messages" in example.keys(), "messages field not found in example"
            text = self.processor.apply_chat_template(example["messages"])
            
        else:
            text = example["text"]

        if self.image_token is not None:
            text = text.replace(self.image_token, self.processor.image_token)

        if self.video_token is not None:
            text = text.replace(self.video_token, self.processor.video_token)

        example["text"] = text

        images_sizes = example["images_sizes"]

        # Convert images_sizes to list if it is a string
        if isinstance(images_sizes, str):
            images_sizes = json.loads(images_sizes)
        assert isinstance(images_sizes, list), "images_sizes must be a list"

        # Create fake images based on the sizes
        fake_images = []
        for size in images_sizes:
            width, height = size
            # Create a random array with the specified dimensions
            fake_image = np.zeros((3, height, width))  # 3 channels for RGB
            fake_images.append(fake_image)

        # Count the number of image tokens in the text
        num_image_tokens = text.count(self.processor.image_token)
        # Count the number of fake images
        num_fake_images = len(fake_images)
        # Assert they are equal
        assert (
            num_image_tokens == num_fake_images
        ), f"Number of image tokens ({num_image_tokens}) does not match number of fake images ({num_fake_images})"

        # Call the processor
        output = self.processor(fake_images, text, do_normalize=False, do_rescale=False)
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

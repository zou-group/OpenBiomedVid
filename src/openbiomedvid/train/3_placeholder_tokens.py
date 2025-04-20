from dataclasses import dataclass

from tore_train.argument_parser import ToreTrainArgumentParser, combine_arguments
from tore_train.data_processors.file.load_datasets_processor import LoadDatasetsProcessor
from tore_train.data_processors.file.save_to_disk_processor import SaveToDiskProcessor
from tore_train.data_processors.chat_processor import ChatProcessor
from tore_train.data_processors.labels_masking_processor import LabelsMaskingProcessor
from tore_train.data_processors.vision.video_lm_processor import VideoLMProcessor

import json

def main():
    arguments_dataclasses = [
        VideoLMProcessor.arguments,
        LabelsMaskingProcessor.arguments,
        ChatProcessor.arguments,
        SaveToDiskProcessor.arguments,
        LoadDatasetsProcessor.arguments,
    ]

    arguments_dataclasses = combine_arguments(arguments_dataclasses)
    parser = ToreTrainArgumentParser(arguments_dataclasses)
    config = parser.parse()

    # Step 1: Load dataset
    print("Loading Dataset")
    dataset = LoadDatasetsProcessor(config)(config.datasets)

    # Step 2: Apply Chat Template
    print("Applying Chat Template")
    dataset = ChatProcessor(config)(dataset)

    # Step 4: Tokenize VLM
    print("Tokenizing Dataset")
    dataset = VideoLMProcessor(config)(dataset)
    
    # Save dataset
    print("Saving data")
    dataset = SaveToDiskProcessor(config)(dataset)

if __name__ == "__main__":
    main()
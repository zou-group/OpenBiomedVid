from typing import List, Optional, Dict, Any
import os
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import av
import numpy as np


def load_video_in_memory(video_data: bytes):
    frames = []
    fps = None
    with av.open(BytesIO(video_data)) as container:
        # Get the video stream's FPS
        stream = container.streams.video[0]
        fps = float(stream.average_rate)

        for frame in container.decode(video=0):
            # Convert frame to a numpy array in RGB format
            frames.append(frame.to_rgb().to_ndarray())
    # Concatenate all frames into a single numpy array
    frames = np.stack(frames, axis=0)
    return {"frames": frames, "fps": fps, "resolution": (stream.width, stream.height)}


class VideosLoader:
    def __init__(self, endpoint_url: str, max_retries: int = 3, max_workers: int = 8):
        """Initializes the ImagesLoader.

        Args:
            endpoint_url: S3-compatible service endpoint URL.
            max_retries: Maximum number of retry attempts for S3 operations.
            max_workers: Maximum number of threads for concurrent image loading.
        """
        self.endpoint_url = endpoint_url
        self.max_retries = max_retries
        self.max_workers = max_workers

        if self.endpoint_url:
            self.s3_client = boto3.client("s3", endpoint_url=self.endpoint_url)
        else:
            self.s3_client = boto3.client("s3")

    def get_local_video(self, video_path: str) -> Image.Image:
        """Loads a local image synchronously.

        Args:
            image_path: Path to the local image file.

        Returns:
            PIL Image object.
        """
        assert os.path.exists(video_path), f"Video file does not exist: {video_path}"
        try:
            with open(video_path, "rb") as file:
                video_data = file.read()
            return load_video_in_memory(video_data)
        except Exception as e:
            raise IOError(f"Failed to load local video '{video_path}': {e}")

    def get_s3_video(self, video_path: str) -> Image.Image:
        """Loads an image from S3 synchronously with retry logic.

        Args:
            video_path: S3 URI (e.g., "s3://bucket-name/path/to/video.mp4").

        Returns:
            PIL Image object.
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((BotoCoreError, ClientError, IOError)),
            reraise=True,
        )
        def _get_s3_video():
            try:
                # Parse the S3 URI
                if not video_path.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {video_path}")
                parts = video_path[5:].split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid S3 URI: {video_path}")
                bucket, key = parts

                # Fetch the object from S3
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                video_data = response["Body"].read()
                return load_video_in_memory(video_data)
            except Exception as e:
                raise IOError(f"Failed to load S3 video '{video_path}': {e}")

        return _get_s3_video()

    def get_video(self, video_path: str) -> Dict[str, Any]:
        """Determines whether to load a local or S3 video and loads it.

        Args:
            image_path: Path to the image (local path or S3 URI).

        Returns:
            Dict[str, Any] with keys "video" and "audio".
        """
        if video_path.startswith("s3://"):
            return self.get_s3_video(video_path)
        else:
            return self.get_local_video(video_path)

    def exist_local_video(self, video_path: str) -> bool:
        """Checks if a local video exists.

        Args:
            video_path: Local path to the video.

        Returns:
            True if the video exists, False otherwise.
        """
        return os.path.exists(video_path)

    def exist_s3_video(self, video_path: str) -> bool:
        """Checks if a video exists in S3.

        Args:
            video_path: S3 URI (e.g., "s3://bucket-name/path/to/video.mp4").

        Returns:
            True if the video exists, False otherwise.
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((BotoCoreError, ClientError, IOError)),
            reraise=True,
        )
        def _exist_s3_video():
            try:
                # Parse the S3 URI
                if not video_path.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {video_path}")
                parts = video_path[5:].split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid S3 URI: {video_path}")
                bucket, key = parts

                # Check if object exists in S3
                try:
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                    return True
                except ClientError as e:
                    if e.response["Error"]["Code"] == "404":
                        return False
                    raise  # Re-raise other ClientError exceptions

            except Exception as e:
                raise IOError(f"Failed to check S3 video existence '{video_path}': {e}")

        return _exist_s3_video()

    def exist_video(self, video_path: str) -> bool:
        """Checks if a video exists in S3.

        Args:
            video_path: S3 URI (e.g., "s3://bucket-name/path/to/video.mp4").

        Returns:
            True if the video exists, False otherwise.
        """
        if not video_path.startswith("s3://"):
            return self.exist_local_video(video_path)
        else:
            return self.exist_s3_video(video_path)

    def get_videos(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """Loads multiple videos synchronously. Can handle both local and S3 videos.

        Args:
            video_paths: List of video paths (local or S3 URIs).

        Returns:
            List of video data.
        """
        videos = [None] * len(video_paths)  # Pre-allocate list with correct size

        if self.max_workers == 1:
            for i, path in enumerate(video_paths):
                videos[i] = self.get_video(path)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all video loading tasks
                future_to_index = {
                    executor.submit(self.get_video, path): i
                    for i, path in enumerate(video_paths)
                }

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        video = future.result()
                        videos[index] = video  # Place video in correct position
                    except Exception as e:
                        raise Exception(
                            f"Error loading video '{video_paths[index]}': {e}"
                        )

        return videos

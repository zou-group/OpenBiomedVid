from typing import List, Optional
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


class ImagesLoader:
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

    def get_local_image(self, image_path: str) -> Image.Image:
        """Loads a local image synchronously.

        Args:
            image_path: Path to the local image file.

        Returns:
            PIL Image object.
        """
        try:
            return Image.open(image_path)
        except Exception as e:
            raise IOError(f"Failed to load local image '{image_path}': {e}")

    def get_s3_image(self, image_path: str) -> Image.Image:
        """Loads an image from S3 synchronously with retry logic.

        Args:
            image_path: S3 URI (e.g., "s3://bucket-name/path/to/image.jpg").

        Returns:
            PIL Image object.
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((BotoCoreError, ClientError, IOError)),
            reraise=True,
        )
        def _get_s3_image():
            try:
                # Parse the S3 URI
                if not image_path.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {image_path}")
                parts = image_path[5:].split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid S3 URI: {image_path}")
                bucket, key = parts

                # Fetch the object from S3
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                image_data = response["Body"].read()
                return Image.open(BytesIO(image_data))
            except Exception as e:
                raise IOError(f"Failed to load S3 image '{image_path}': {e}")

        return _get_s3_image()

    def get_image(self, image_path: str) -> Image.Image:
        """Determines whether to load a local or S3 image and loads it.

        Args:
            image_path: Path to the image (local path or S3 URI).

        Returns:
            PIL Image object.
        """
        if image_path.startswith("s3://"):
            return self.get_s3_image(image_path)
        else:
            return self.get_local_image(image_path)

    def exist_local_image(self, image_path: str) -> bool:
        """Checks if a local image exists.

        Args:
            image_path: Local path to the image.

        Returns:
            True if the image exists, False otherwise.
        """
        return os.path.exists(image_path)

    def exist_s3_image(self, image_path: str) -> bool:
        """Loads an image from S3 synchronously with retry logic.

        Args:
            image_path: S3 URI (e.g., "s3://bucket-name/path/to/image.jpg").

        Returns:
            True if the image exists, False otherwise.
        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=4),
            retry=retry_if_exception_type((BotoCoreError, ClientError, IOError)),
            reraise=True,
        )
        def _exist_s3_image():
            try:
                # Parse the S3 URI
                if not image_path.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {image_path}")
                parts = image_path[5:].split("/", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid S3 URI: {image_path}")
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
                raise IOError(f"Failed to check S3 image existence '{image_path}': {e}")

        return _exist_s3_image()

    def exist_image(self, image_path: str) -> bool:
        """Checks if an image exists in S3.

        Args:
            image_path: S3 URI (e.g., "s3://bucket-name/path/to/image.jpg").

        Returns:
            True if the image exists, False otherwise.
        """
        if not image_path.startswith("s3://"):
            return self.exist_local_image(image_path)
        else:
            return self.exist_s3_image(image_path)

    def get_images(self, image_paths: List[str]) -> List[Image.Image]:
        """Loads multiple images synchronously. Can handle both local and S3 images.

        Args:
            image_paths: List of image paths (local or S3 URIs).

        Returns:
            List of PIL Image objects.
        """
        images = [None] * len(image_paths)  # Pre-allocate list with correct size

        if self.max_workers == 1:
            for i, path in enumerate(image_paths):
                images[i] = self.get_image(path)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all image loading tasks
                future_to_index = {
                    executor.submit(self.get_image, path): i
                    for i, path in enumerate(image_paths)
                }

                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        image = future.result()
                        images[index] = image  # Place image in correct position
                    except Exception as e:
                        raise Exception(
                            f"Error loading image '{image_paths[index]}': {e}"
                        )

        return images

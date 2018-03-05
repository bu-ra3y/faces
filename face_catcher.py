import abc
from cv2 import imwrite
from pathlib import Path
from time import time


class FaceCatcher:
    OUTPUT_FOLDER = Path.home() / 'Downloads' / 'faces'

    def __init__(self, filters=None):
        if not self.OUTPUT_FOLDER.exists():
            self.OUTPUT_FOLDER.mkdir()

        self.counter = 0

        if filters:
            self.filters = filters
        else:  # Default
            self.filters = [
                TimeFilter(),
                ConfidenceFilter(),
            ]

    @property
    def current_output_file(self):
        return (self.OUTPUT_FOLDER / str(self.counter)).with_suffix('.jpg')

    @property
    def next_output_file(self):
        while self.current_output_file.exists():
            self.counter += 1
        return self.current_output_file

    def catch(self, image, **kwargs) -> bool:
        """
        Catch the incoming image.  Save that image if it passes some filters.
        :param image:
        :return: True if the image was decided to be a keeper
        """
        if self.is_a_keeper(image=image, **kwargs):
            self._keep(image=image)
            return True
        else:
            return False

    def is_a_keeper(self, image, **kwargs) -> bool:
        """
        See if the image is a keeper.
        Must pass through all active filters.
        :return: True if the image is a keeper.
        """
        return all([filter.is_a_keeper(image=image, **kwargs) for filter in self.filters])

    def _keep(self, image):
        file = self.next_output_file
        return_code = imwrite(
            filename=str(file),
            img=image
        )
        print(f"Saved images to {file} with return value: {return_code}")


class Filter(metaclass=abc.ABCMeta):
    """Abstract Base Class for Filters.
    A Filter takes an image and decides whether to keep it or not."""

    @abc.abstractmethod
    def is_a_keeper(self, image, **kwargs):
        """Decides whether the given image is a keeper."""


class NoFilter(Filter):
    def is_a_keeper(self, image, **kwargs) -> bool:
        return True


class ConfidenceFilter(Filter):
    def __init__(self, threshold=5):
        self.threshold = threshold

    def is_a_keeper(self, image, **kwargs) -> bool:
        if kwargs and 'confidence' in kwargs.keys():
            if kwargs['confidence'] > self.threshold:
                return True
        return False


class TimeFilter(Filter):
    def __init__(self, seconds=10):
        self.interval = seconds
        self.next_time = time()  # start now

    def is_a_keeper(self, image, **kwargs) -> bool:
        if time() > self.next_time:  # if we have reached the allowed time
            self.next_time = time() + self.interval  # update next allowed time
            return True
        else:
            return False

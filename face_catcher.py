import abc
from cv2 import imwrite
from pathlib import Path
from time import time


class FaceCatcher:
    OUTPUT_FOLDER = Path.home() / 'Downloads' / 'faces'

    def __init__(self, filter=None):
        if not self.OUTPUT_FOLDER.exists():
            self.OUTPUT_FOLDER.mkdir()

        self.counter = 0

        if filter:
            self.filter = NoFilter()
        else:
            # self.filter = NoFilter()  # Default
            self.filter = TimeFilter()  # Default

    @property
    def current_output_file(self):
        return (self.OUTPUT_FOLDER / str(self.counter)).with_suffix('.jpg')

    @property
    def next_output_file(self):
        while self.current_output_file.exists():
            self.counter += 1
        return self.current_output_file

    def catch(self, image):
        if self.filter.is_a_keeper(image=image):
            self._keep(image=image)

    def _keep(self, image):
        file = self.next_output_file
        retval = imwrite(
            filename=str(file),
            img=image
        )
        print(f"Saved images to {file} with return value: {retval}")


class Filter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def is_a_keeper(self, image):
        """ docs """


class NoFilter(Filter):
    def is_a_keeper(self, image) -> bool:
        return True


class TimeFilter(Filter):
    def __init__(self, seconds=10):
        self.interval = seconds
        self.next_time = time()  # start now

    def is_a_keeper(self, image) -> bool:
        if time() > self.next_time:  # if we have reached the allowed time
            self.next_time = time() + self.interval  # update next allowed time
            return True
        else:
            return False

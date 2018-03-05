from pathlib import Path

from cv2 import imwrite


class FaceCatcher():

    OUTPUT_FOLDER = Path.home() / 'Downloads' / 'faces'

    def __init__(self):
        if not self.OUTPUT_FOLDER.exists():
            self.OUTPUT_FOLDER.mkdir()
        self.counter = 0

    @property
    def current_output_file(self):
        return (self.OUTPUT_FOLDER / str(self.counter)).with_suffix('.jpg')

    @property
    def next_output_file(self):
        while self.current_output_file.exists():
            self.counter += 1
        return self.current_output_file


    def catch(self, image):
        if self._is_a_keeper(image=image):
            self._keep(image=image)
    
    @staticmethod
    def _is_a_keeper(image) -> bool:
        return True

    def _keep(self, image):
        file = self.next_output_file
        retval = imwrite(
            filename=str(file),
            img=image
        )
        print(f"Saved images to {file} with return value: {retval}")
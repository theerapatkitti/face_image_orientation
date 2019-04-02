import io
import os
import json
import requests
import unittest
import hashlib
from PIL import Image


API_PREDICT = "http://35.247.181.180/predict"
API_CORRECT = "http://35.247.181.180/correct"

image_dir = os.path.join("..", "image")


class TestAPI(unittest.TestCase):
    """
    Test Flask app before deployment
    """
    def test_predictio(self):
        """
        Test orientation prediction of different rotation
        """
        with open(os.path.join(image_dir, "test0.jpg"), "rb") as f:
            files = {"image": ("test0.jpg", f, "image/jpeg")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 0)

        with open(os.path.join(image_dir, "test90.jpg"), "rb") as f:
            files = {"image": ("test90.jpg", f, "image/jpeg")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 90)
        
        with open(os.path.join(image_dir, "test180.jpg"), "rb") as f:
            files = {"image": ("test180.jpg", f, "image/jpeg")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 180)
        
        with open(os.path.join(image_dir, "test270.jpg"), "rb") as f:
            files = {"image": ("test270.jpg", f, "image/jpeg")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 270)

    def test_prediction_jpeg(self):
        """
        Test orientation prediction with jpeg image
        """
        with open(os.path.join(image_dir, "test.jpg"), "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 180)

    def test_prediction_png(self):
        """
        Test orientation prediction with png image
        """
        with open(os.path.join(image_dir, "test.png"), "rb") as f:
            files = {"image": ("test.png", f, "image/png")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 180)
    
    def test_prediction_bmp(self):
        """
        Test orientation prediction with bitmap image
        """
        with open(os.path.join(image_dir, "test.bmp"), "rb") as f:
            files = {"image": ("test.bmp", f, "image/bmp")}
            response = requests.post(API_PREDICT, files=files)
            response = json.loads(response.content)
            self.assertEqual(response.get("prediction").get("orientation"), 180)

    def test_correction_jpeg(self):
        """
        Test orientation correction with jpeg image
        """
        with open(os.path.join(image_dir, "test.jpg"), "rb") as f:
            files = {"image": ("test.jpg", f, "image/jpeg")}
            response = requests.post(API_CORRECT, files=files)
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            md5hash = (hashlib.md5(Image.open(io.BytesIO(response.content)).tobytes())).hexdigest()
            actual_md5hash = (hashlib.md5(Image.open(os.path.join(image_dir, "rotated.jpg")).tobytes())).hexdigest()
            self.assertEqual(actual_md5hash, md5hash)

    def test_correction_png(self):
        """
        Test orientation correction with png image
        """
        with open(os.path.join(image_dir, "test.png"), "rb") as f:
            files = {"image": ("test.png", f, "image/png")}
            response = requests.post(API_CORRECT, files=files)
            self.assertEqual(response.headers["Content-Type"], "image/png")
            md5hash = (hashlib.md5(Image.open(io.BytesIO(response.content)).tobytes())).hexdigest()
            actual_md5hash = (hashlib.md5(Image.open(os.path.join(image_dir, "rotated.png")).tobytes())).hexdigest()
            self.assertEqual(actual_md5hash, md5hash)

    def test_correction_bmp(self):
        """
        Test orientation correction with bitmap image
        """
        with open(os.path.join(image_dir, "test.bmp"), "rb") as f:
            files = {"image": ("test.bmp", f, "image/bmp")}
            response = requests.post(API_CORRECT, files=files)
            self.assertEqual(response.headers["Content-Type"], "image/bmp")
            md5hash = (hashlib.md5(Image.open(io.BytesIO(response.content)).tobytes())).hexdigest()
            actual_md5hash = (hashlib.md5(Image.open(os.path.join(image_dir, "rotated.bmp")).tobytes())).hexdigest()
            self.assertEqual(actual_md5hash, md5hash)


if __name__ == "__main__":
    unittest.main()
from pathlib import Path
import tempfile
import unittest
from dataset import label_for_filename, labeled_images

class DatasetTests(unittest.TestCase):
    def test_unknown_label_is_not_silently_assigned(self):
        self.assertEqual(label_for_filename("Autistic.1.jpg"), "Autistic")
        self.assertEqual(label_for_filename("Non_Autistic.2.jpg"), "Non_Autistic")
        with self.assertRaises(ValueError): label_for_filename("unknown.jpg")

    def test_order_is_stable_and_non_images_are_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ["Non_Autistic.2.jpg", "Autistic.1.jpg", "README.txt"]:
                (Path(folder) / name).touch()
            self.assertEqual(labeled_images(folder), [("Autistic.1.jpg", "Autistic"), ("Non_Autistic.2.jpg", "Non_Autistic")])

    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError): labeled_images(folder)

if __name__ == "__main__": unittest.main()

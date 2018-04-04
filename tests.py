import unittest

class DataSetGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.dataset_gen = DataSetGenerator('small')
        self.assertEqual(self.dataset_gen.tracks.shape, (8000, 52))
        self.assertEqual(self.dataset_gen.features.shape, (8000, 518))

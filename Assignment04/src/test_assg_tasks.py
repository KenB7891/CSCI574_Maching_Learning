import numpy as np
import pandas as pd
import sklearn
import unittest
from assg_tasks import task_1_1_load_data, task_1_2_linear_svm_classifier, gaussian_kernel, task_3_1_load_data, task_3_2_rbf_svm_classifier
from twisted.trial import unittest


class test_task_1_1_load_data(unittest.TestCase):
    def setUp(self):
        self.X, self.y = task_1_1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.X, pd.core.frame.DataFrame)
        self.assertIsInstance(self.y, np.ndarray)

    def test_X_properties(self):
        self.assertEqual(self.X.shape, (51, 2))
        self.assertEqual(list(self.X.columns), ['x_1', 'x_2'])

    def test_y_properties(self):
        self.assertEqual(self.y.shape, (51,))
        expected_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(np.all(np.isclose(self.y, expected_labels)))


class test_task_1_2_linear_svm_classifier(unittest.TestCase):
    def setUp(self):
        X, y = task_1_1_load_data()
        self.model = task_1_2_linear_svm_classifier(X, y, C=1.0)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._data.StandardScaler)
        self.assertIsInstance(self.model[1], sklearn.svm._classes.SVC)

    def test_model_parameters(self):
        self.assertAlmostEqual(self.model[1].C, 1.0)
        self.assertEqual(self.model[1].kernel, 'linear')

    def test_intercept(self):
        intercept = self.model[1].intercept_
        self.assertAlmostEqual(intercept[0], -0.56521111, places=4)

    def test_coef(self):
        coef = self.model[1].coef_[0]
        self.assertAlmostEqual(coef[0], 1.32982938, places=4)
        self.assertAlmostEqual(coef[1], 1.8493673, places=4)


class test_task_1_3_linear_svm_classifier(unittest.TestCase):
    def setUp(self):
        X, y = task_1_1_load_data()
        self.model = task_1_2_linear_svm_classifier(X, y, C=75.0)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._data.StandardScaler)
        self.assertIsInstance(self.model[1], sklearn.svm._classes.SVC)

    def test_model_parameters(self):
        self.assertAlmostEqual(self.model[1].C, 75.0)
        self.assertEqual(self.model[1].kernel, 'linear')

    def test_intercept(self):
        intercept = self.model[1].intercept_
        self.assertAlmostEqual(intercept[0], -1.50646674, places=4)

    def test_coef(self):
        coef = self.model[1].coef_[0]
        self.assertAlmostEqual(coef[0], 4.30510962, places=4)
        self.assertAlmostEqual(coef[1], 9.76428193, places=4)


class test_task_2_1_gaussian_kernel(unittest.TestCase):
    def setUp(self):
        pass

    def test_samelocation(self):
        sigma = 2.0
        xi = np.array([1, 0, -1, -3])
        xj = np.array([1, 0, -1, -3])
        self.assertAlmostEqual(gaussian_kernel(xi, xj, sigma), 1.0)

    def test_midsimilarity(self):
        sigma = 2.0
        xi = np.array([1, 1, -1, -3])
        xj = np.array([2, 0, -1, -5])
        self.assertAlmostEqual(gaussian_kernel(xi, xj, sigma), 0.472366552741)

    def test_lowsimilarity(self):
        sigma = 2.0
        xi = np.array([1, 6,  2, -2])
        xj = np.array([5, 0, -1, -5])
        self.assertAlmostEqual(gaussian_kernel(xi, xj, sigma), 0.000158461325116)

    def test_moredimensions(self):
        sigma = 2.0
        xi = np.array([1, 0, -1, -3, 5, -2, 7.8, 9.5])
        xj = np.array([1, 0, -1, -3, 5, -2, 7.6, 9.5])
        self.assertAlmostEqual(gaussian_kernel(xi, xj, sigma), 0.9950124791926823)


class test_task_3_1_load_data(unittest.TestCase):
    def setUp(self):
        self.X, self.y = task_3_1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.X, pd.core.frame.DataFrame)
        self.assertIsInstance(self.y, np.ndarray)

    def test_X_properties(self):
        self.assertEqual(self.X.shape, (863, 2))
        self.assertEqual(list(self.X.columns), ['x_1', 'x_2'])

    def test_y_properties(self):
        self.assertEqual(self.y.shape, (863,))
        self.assertTrue(sum(self.y) == 480)


class test_task_3_2_rbf_svm_classifier(unittest.TestCase):
    def setUp(self):
        X, y = task_3_1_load_data()
        self.model = task_3_2_rbf_svm_classifier(X, y, C=1.0, gamma=8.0)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._data.StandardScaler)
        self.assertIsInstance(self.model[1], sklearn.svm._classes.SVC)

    def test_model_parameters(self):
        self.assertAlmostEqual(self.model[1].C, 1.0)
        self.assertAlmostEqual(self.model[1].gamma, 8.0)
        self.assertEqual(self.model[1].kernel, 'rbf')

    def test_intercept(self):
        intercept = self.model[1].intercept_
        self.assertAlmostEqual(intercept[0], 0.2133041, places=4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
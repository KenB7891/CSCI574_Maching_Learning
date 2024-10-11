import numpy as np
import unittest
import sklearn
from assg_tasks import task1_load_data
from assg_tasks import task2_underfit_model
from assg_tasks import learning_curve_errors
from assg_tasks import task4_overfit_model
from assg_tasks import task5_lasso_model
from assg_tasks import task6_ridge_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from twisted.trial import unittest


class test_task1_load_data(unittest.TestCase):
    def setUp(self):
        self.x, self.y = task1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.x, np.ndarray)
        self.assertIsInstance(self.y, np.ndarray)

    def test_x_properties(self):
        self.assertEqual(self.x.shape, (100, 1))

    def test_y_properties(self):
        self.assertEqual(self.y.shape, (100,))

class test_task2_underfit_model(unittest.TestCase):
    def setUp(self):
        self.x, self.y = task1_load_data()
        self.model = task2_underfit_model(self.x, self.y)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
        pf = self.model[0]
        self.assertEqual(pf.degree, 2);
        pf_params = {'degree': 2, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
        self.assertDictEqual(pf.get_params(), pf_params)
        self.assertIsInstance(self.model[1], sklearn.linear_model._base.LinearRegression)
        lr = self.model[1]
        lr_params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}
        self.assertDictEqual(lr.get_params(), lr_params)
        
    def test_intercept(self):
        intercept = self.model[1].intercept_
        self.assertAlmostEqual(intercept, 2.086873037695686, places=4)
    
    def test_coef(self):
        coef = self.model[1].coef_
        self.assertAlmostEqual(coef[0], 0.81875687, places=4)
        self.assertAlmostEqual(coef[1], -1.36223923, places=4)

    def test_r2score(self):
        r2score = self.model.score(self.x, self.y)
        self.assertAlmostEqual(r2score, 0.7769591377499175, places=4)

    def test_rmse(self):
        y_predict = self.model.predict(self.x)
        rmse = root_mean_squared_error(self.y, y_predict)
        self.assertAlmostEqual(rmse, 0.31486435395552376, places=4)

class test_learning_curve_errors(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.x = np.linspace(-5.0, 5.0, 100)
        self.y = 2.5 * self.x + 1.0 + np.random.randn(100) * 1.0
        self.x = self.x.reshape(-1, 1)
        self.model = LinearRegression()
        self.train_errors, self.test_errors = learning_curve_errors(self.model, self.x, self.y)

    def test_train_errors(self):
        self.assertEqual(len(self.train_errors), 79)
        expected_errors = np.array([0.0, 1.3710242980056706e-15, 0.008226499792554355, 0.6060453617587794, 0.5519805145155924])
        self.assertTrue(np.allclose(self.train_errors[:5], expected_errors))
        self.assertAlmostEqual(max(self.train_errors), 1.0929159457839905)

    def test_test_errors(self):
        self.assertEqual(len(self.test_errors), 79)
        expected_errors = np.array([6.807111779574341, 2.8618559336108844, 2.8588563844249277, 1.8483844530375642, 1.9383035157683932])
        self.assertTrue(np.allclose(self.test_errors[:5], expected_errors))
        self.assertAlmostEqual(max(self.test_errors), 6.807111779574341)

class test_task4_overfit_model(unittest.TestCase):
    def setUp(self):
        self.x, self.y = task1_load_data()
        self.model = task4_overfit_model(self.x, self.y)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
        pf = self.model[0]
        self.assertEqual(pf.degree,100);
        pf_params = {'degree': 100, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
        self.assertDictEqual(pf.get_params(), pf_params)
        self.assertIsInstance(self.model[1], sklearn.linear_model._base.LinearRegression)
        lr = self.model[1]
        lr_params = {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'positive': False}
        self.assertDictEqual(lr.get_params(), lr_params)
        
    def test_intercept(self):
        intercept = self.model[1].intercept_
        self.assertAlmostEqual(intercept, 1.9072641972361766, places=2)
    
    def test_coef(self):
        coef = self.model[1].coef_
        self.assertEqual(len(coef), 100)
        self.assertAlmostEqual(coef[0], 0.9981342688081702, places=2)
        #self.assertAlmostEqual(coef[1], 20.55634616857911, places=2)
        #self.assertAlmostEqual(coef[10], 179123620.68405914, places=2)
        #self.assertAlmostEqual(coef[50], -475881068965.47687, places=2)
        #self.assertAlmostEqual(coef[99], 8361588790.13140673, places=0)

    def test_r2score(self):
        r2score = self.model.score(self.x, self.y)
        self.assertAlmostEqual(r2score, 0.9973959804305983, places=2)

    def test_rmse(self):
        y_predict = self.model.predict(self.x)
        rmse = root_mean_squared_error(self.y, y_predict)
        self.assertAlmostEqual(rmse, 0.03402149554332673, places=2)

class test_task5_lasso_model(unittest.TestCase):
    def setUp(self):
        self.x, self.y = task1_load_data()
        self.model = task5_lasso_model(self.x, self.y)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
        pf = self.model[0]
        self.assertEqual(pf.degree,100);
        pf_params = {'degree': 100, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
        self.assertDictEqual(pf.get_params(), pf_params)
        self.assertIsInstance(self.model[1], sklearn.linear_model._coordinate_descent.Lasso)
        params = self.model[1].get_params()
        self.assertTrue('alpha' in params.keys())
        
    def test_coef(self):
        coef = self.model[1].coef_
        self.assertEqual(len(coef), 100)

    def test_r2score(self):
        r2score = self.model.score(self.x, self.y)
        self.assertTrue(r2score > 0.9)

    def test_rmse(self):
        y_predict = self.model.predict(self.x)
        rmse = root_mean_squared_error(self.y, y_predict)
        self.assertTrue(rmse < 0.2)

class test_task6_ridge_model(unittest.TestCase):
    def setUp(self):
        self.x, self.y = task1_load_data()
        self.model = task6_ridge_model(self.x, self.y)

    def test_model_pipeline(self):
        self.assertIsInstance(self.model, sklearn.pipeline.Pipeline)
        self.assertIsInstance(self.model[0], sklearn.preprocessing._polynomial.PolynomialFeatures)
        pf = self.model[0]
        self.assertEqual(pf.degree, 100);
        pf_params = {'degree': 100, 'include_bias': False, 'interaction_only': False, 'order': 'C'}
        self.assertDictEqual(pf.get_params(), pf_params)
        self.assertIsInstance(self.model[1], sklearn.linear_model._ridge.Ridge)
        params = self.model[1].get_params()
        self.assertTrue('alpha' in params.keys())
        
    def test_coef(self):
        coef = self.model[1].coef_
        self.assertEqual(len(coef), 100)

    def test_r2score(self):
        r2score = self.model.score(self.x, self.y)
        self.assertTrue(r2score > 0.9)

    def test_rmse(self):
        y_predict = self.model.predict(self.x)
        rmse = root_mean_squared_error(self.y, y_predict)
        self.assertTrue(rmse < 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
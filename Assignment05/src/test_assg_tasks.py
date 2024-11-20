import numpy as np
import pandas as pd
import sklearn
import unittest
from assg_tasks import task_1_load_data, train_val_test_split, task_3_voting_ensemble, task_4_bag_of_trees_ensemble, create_stacked_data, task_6_stacked_ensemble
from twisted.trial import unittest


class test_task_1_load_data(unittest.TestCase):
    def setUp(self):
        self.X, self.y = task_1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.X, pd.core.frame.DataFrame)
        self.assertIsInstance(self.y, np.ndarray)

    def test_X_properties(self):
        self.assertEqual(self.X.shape, (19020, 10))
        self.assertEqual(list(self.X.columns), ['fLength:', 'fWidth:', 'fSize:', 'fConc:', 'fConc1:', 'fAsym:', 'fM3Long:', 'fM3Trans:', 'fAlpha:', 'fDist:'])

    def test_y_properties(self):
        self.assertEqual(self.y.shape, (19020,))
        self.assertEqual(self.y.sum(), 12332)


class test_train_val_test_split(unittest.TestCase):
    def setUp(self):
        pass

    def test_generated_data(self):
        np.random.seed(42)
        X = np.random.rand(1000, 5)
        y = np.random.randint(0, 2, size=(1000,))
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=500, val_size=300)

        # test train split is as expected
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertEqual(X_train.shape, (500, 5))
        self.assertEqual(y_train.shape, (500,))
        # if random shuffling uses correct seed we should pass this, index 18351
        # ends up first training sample and we end up with 6475 positive targets
        self.assertAlmostEqual(X_train[0,0], 0.06725591396736785)
        self.assertEqual(y_train.sum(), 257)

        # test validation split
        self.assertIsInstance(X_val, np.ndarray)
        self.assertIsInstance(y_val, np.ndarray)
        self.assertEqual(X_val.shape, (300, 5))
        self.assertEqual(y_val.shape, (300,))
        self.assertAlmostEqual(X_val[0,0], 0.4978125080342902)
        self.assertEqual(y_val.sum(), 159)

        # test the test split
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(X_test.shape, (200, 5))
        self.assertEqual(y_test.shape, (200,))
        self.assertAlmostEqual(X_test[0,0], 0.12786680122241756)
        self.assertEqual(y_test.sum(), 95)

    def test_assg_data(self):
        X, y = task_1_load_data()
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)

        # test train split is as expected
        self.assertIsInstance(X_train, pd.core.frame.DataFrame)
        self.assertIsInstance(y_train, np.ndarray)
        self.assertEqual(X_train.shape, (10000, 10))
        self.assertEqual(y_train.shape, (10000,))
        # if random shuffling uses correct seed we should pass this, index 18351
        # ends up first training sample and we end up with 6475 positive targets
        self.assertEqual(X_train.iloc[0].name, 18351)
        self.assertEqual(y_train.sum(), 6475)

        # test validation split
        self.assertIsInstance(X_val, pd.core.frame.DataFrame)
        self.assertIsInstance(y_val, np.ndarray)
        self.assertEqual(X_val.shape, (4510, 10))
        self.assertEqual(y_val.shape, (4510,))
        self.assertEqual(X_val.iloc[0].name, 9394)
        self.assertEqual(y_val.sum(), 2918)

        # test the test split
        self.assertIsInstance(X_test, pd.core.frame.DataFrame)
        self.assertIsInstance(y_test, np.ndarray)
        self.assertEqual(X_test.shape, (4510, 10))
        self.assertEqual(y_test.shape, (4510,))
        self.assertEqual(X_test.iloc[0].name, 8156)
        self.assertEqual(y_test.sum(), 2939)
        

class test_task_3_hard_voting_ensemble(unittest.TestCase):
    def setUp(self):
        X, y = task_1_load_data()
        X_train, y_train, self.X_val, self.y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
        self.vc = task_3_voting_ensemble(X_train, y_train, voting='hard')

    def test_knn_estimator(self):
        estimator_names = list(self.vc.named_estimators_.keys())
        self.assertTrue('knn' in estimator_names)
        model = self.vc.named_estimators_['knn']
        self.assertIsInstance(model, sklearn.neighbors._classification.KNeighborsClassifier)
        self.assertTrue(model.score(self.X_val, self.y_val) > 0.75)

    def test_dt_estimator(self):
        estimator_names = list(self.vc.named_estimators_.keys())
        self.assertTrue('dt' in estimator_names)
        model = self.vc.named_estimators_['dt']
        self.assertIsInstance(model, sklearn.tree._classes.DecisionTreeClassifier)
        self.assertTrue(model.score(self.X_val, self.y_val) > 0.75)

    def test_lr_estimator(self):
        estimator_names = list(self.vc.named_estimators_.keys())
        self.assertTrue('lr' in estimator_names)
        model = self.vc.named_estimators_['lr']
        self.assertIsInstance(model, sklearn.linear_model._logistic.LogisticRegression)
        self.assertTrue(model.score(self.X_val, self.y_val) > 0.75)

    def test_svc_estimator(self):
        estimator_names = list(self.vc.named_estimators_.keys())
        self.assertTrue('svc' in estimator_names)
        model = self.vc.named_estimators_['svc']
        self.assertIsInstance(model, sklearn.svm._classes.SVC)
        self.assertTrue(model.score(self.X_val, self.y_val) > 0.75)

    def test_mlp_estimator(self):
        estimator_names = list(self.vc.named_estimators_.keys())
        self.assertTrue('mlp' in estimator_names)
        model = self.vc.named_estimators_['mlp']
        self.assertIsInstance(model, sklearn.neural_network._multilayer_perceptron.MLPClassifier)
        self.assertTrue(model.score(self.X_val, self.y_val) > 0.75)

    def test_voting_ensemble(self):
        self.assertIsInstance(self.vc, sklearn.ensemble._voting.VotingClassifier)
        self.assertEqual(self.vc.get_params()['voting'], 'hard')
        self.assertTrue(self.vc.score(self.X_val, self.y_val) > 0.75)

class test_task_3_soft_voting_ensemble(unittest.TestCase):
    def setUp(self):
        X, y = task_1_load_data()
        X_train, y_train, self.X_val, self.y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
        self.vc = task_3_voting_ensemble(X_train, y_train, voting='soft')

    def test_voting_ensemble(self):
        # we won't retest individual models, just check the voting ensemble uses soft voting
        # correctly when asked
        self.assertIsInstance(self.vc, sklearn.ensemble._voting.VotingClassifier)
        self.assertEqual(self.vc.get_params()['voting'], 'soft')
        self.assertTrue(self.vc.score(self.X_val, self.y_val) > 0.75)

class test_task_4_bag_of_trees_ensemble(unittest.TestCase):
    def setUp(self):
        X, y = task_1_load_data()
        X_train, y_train, self.X_val, self.y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
        self.bag = task_4_bag_of_trees_ensemble(X_train, y_train)

    def test_bag_of_trees_ensemble(self):
        self.assertTrue(isinstance(self.bag, sklearn.ensemble._forest.RandomForestClassifier) or
                        isinstance(self.bag, sklearn.ensemble._forest.ExtraTreesClassifier))
        self.assertTrue(self.bag.score(self.X_val, self.y_val) > 0.85)


class test_create_stacked_data(unittest.TestCase):
    def setUp(self):
        X, y = task_1_load_data()
        self.X_train, self.y_train, self.X_val, self.y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
        self.vc = task_3_voting_ensemble(self.X_train, self.y_train, voting='soft')

    def test_stack_validation_data(self):
        X_stacked = create_stacked_data(self.vc, self.X_val)

        # we expect 5 estimators each generating 2 outputs
        self.assertEqual(X_stacked.shape, (4510, 10))

        # first two columns should be from first estimator
        X_0 = self.vc.estimators_[0].predict_proba(self.X_val)
        self.assertTrue(np.allclose(X_0, X_stacked[:,0:2]))

        # last two columns should come from last estimator
        X_4 = self.vc.estimators_[4].predict_proba(self.X_val)
        self.assertTrue(np.allclose(X_4, X_stacked[:,8:10]))

    def test_stack_random_data(self):
        # a random set of 500 inputs with 10 numeric features
        data = np.random.rand(500, 10)
        X = pd.DataFrame(data, columns=self.X_train.columns)
        X_stacked = create_stacked_data(self.vc, X)

        # we expect 5 estimators each generating 2 outputs
        self.assertEqual(X_stacked.shape, (500, 10))

        # first two columns should be from first estimator
        X_0 = self.vc.estimators_[0].predict_proba(X)
        self.assertTrue(np.allclose(X_0, X_stacked[:,0:2]))

        # last two columns should come from last estimator
        X_4 = self.vc.estimators_[4].predict_proba(X)
        self.assertTrue(np.allclose(X_4, X_stacked[:,8:10]))


class test_task_6_stacked_ensemble(unittest.TestCase):
    def setUp(self):
        X, y = task_1_load_data()
        self.X_train, self.y_train, self.X_val, self.y_val, X_test, y_test = train_val_test_split(X, y, train_size=10000, val_size=4510)
        self.blending_estimator, self.voting_ensemble = task_6_stacked_ensemble(self.X_train, self.y_train, self.X_val, self.y_val)

    def test_voting_ensemble(self):
        self.assertIsInstance(self.voting_ensemble, sklearn.ensemble._voting.VotingClassifier)
        self.assertEqual(self.voting_ensemble.get_params()['voting'], 'soft')
        # did they fit with the training data? if so should fit on 10000 samples
        self.assertEqual(self.voting_ensemble.named_estimators_['knn'].n_samples_fit_, 10000)
        
    def test_blending_estimator(self):
        self.assertIsInstance(self.blending_estimator, sklearn.svm._classes.SVC)
        # did they fit with the training data? if so should fit on 10000 samples
        self.assertEqual(self.blending_estimator.shape_fit_, (4510, 10))

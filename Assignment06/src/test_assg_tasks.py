import numpy as np
import pandas as pd
import sklearn
import unittest
from assg_tasks import task_1_1_load_data, find_closest_centroids, compute_centroids, kmeans_cluster, task_2_1_load_data, feature_normalize, pca, project_data, recover_data
from twisted.trial import unittest


class test_task_1_1_load_data(unittest.TestCase):
    def setUp(self):
        self.X = task_1_1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.X, np.ndarray)

    def test_X_properties(self):
        self.assertEqual(self.X.shape, (300, 2))
        #self.assertEqual(list(self.X.columns), ['x_1', 'x_2'])


class test_find_closest_centroids(unittest.TestCase):
    def setUp(self):
        self.X = task_1_1_load_data()

    def test_assg_initial_centroids(self):
        initial_centroids = np.array([[3.0, 3.0],
                                      [6.0, 2.0],
                                      [8.0, 5.0]])
        expected_c = np.array(
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
             1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0]            
        )
        c = find_closest_centroids(self.X, initial_centroids)
        self.assertEqual(c.shape, (300,))
        self.assertTrue(np.array_equal(c, expected_c))

    def test_two_centroids(self):
        # should work when K=2 centroids
        initial_centroids = np.array([[4.0, 1.0], 
                                      [4.0, 6.0]])
        expected_c = np.array(
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        )
        c = find_closest_centroids(self.X, initial_centroids)
        self.assertEqual(c.shape, (300,))
        self.assertTrue(np.array_equal(c, expected_c))

    def test_random_dataset(self):
        # should work with a random dataset and K=5 centroids
        np.random.seed(42)
        X = np.random.rand(20, 20)
        # use first 5 random points as centroids
        initial_centroids = X[:5]
        expected_c = np.array(
            [0, 1, 2, 3, 4, 1, 4, 1, 3, 0, 0, 4, 0, 2, 4, 1, 0, 0, 1, 3]
        )

        c = find_closest_centroids(X, initial_centroids)
        self.assertEqual(c.shape, (20,))
        self.assertTrue(np.array_equal(c, expected_c))


class test_compute_centroids(unittest.TestCase):
    def setUp(self):
        self.X = task_1_1_load_data()

    def test_assg_initial_centroids(self):
        initial_centroids = np.array([[3.0, 3.0],
                                      [6.0, 2.0],
                                      [8.0, 5.0]])
        K = initial_centroids.shape[0]
        c = find_closest_centroids(self.X, initial_centroids)
        expected_centroids = np.array(
            [[2.42830111, 3.15792418],
             [5.81350331, 2.63365645],
             [7.11938687, 3.6166844 ]]
        )
        updated_centroids = compute_centroids(self.X, c, K)
        
        self.assertEqual(updated_centroids.shape, (3, 2))
        self.assertTrue(np.all(np.isclose(updated_centroids, expected_centroids)))

    def test_two_centroids(self):
        # should work when K=2 centroids
        initial_centroids = np.array([[4.0, 1.0], 
                                      [4.0, 6.0]])
        K = initial_centroids.shape[0]
        c = find_closest_centroids(self.X, initial_centroids)
        expected_centroids = np.array(
            [[4.51321226, 1.95570575],
             [2.05132113, 5.01925539]]
        )
        updated_centroids = compute_centroids(self.X, c, K)
        
        self.assertEqual(updated_centroids.shape, (2, 2))
        self.assertTrue(np.all(np.isclose(updated_centroids, expected_centroids)))

    def test_random_dataset(self):
        # should work with a random dataset and K=5 centroids
        np.random.seed(42)
        X = np.random.rand(20, 20)
        # use first 5 random points as centroids
        initial_centroids = X[:5]
        K = initial_centroids.shape[0]
        c = find_closest_centroids(X, initial_centroids)
        expected_centroids = np.array(
            [[0.55648987, 0.54838734, 0.59442999, 0.57641605, 0.39820266,
              0.51787353, 0.43240153, 0.61809305, 0.48227664, 0.47288888,
              0.30266315, 0.75974241, 0.65150762, 0.27760649, 0.25318302,
              0.33496875, 0.52895405, 0.69230356, 0.65885493, 0.44846882],
             [0.4091162 , 0.44046588, 0.42052743, 0.47177151, 0.66320662,
              0.48666197, 0.53942239, 0.60015434, 0.37273376, 0.14845407,
              0.46906232, 0.1335971 , 0.25188756, 0.77842277, 0.698371  ,
              0.58737261, 0.60684993, 0.24848561, 0.55706702, 0.44796175],
             [0.36852273, 0.74261538, 0.08723627, 0.71382503, 0.56807653,
              0.70164545, 0.50436341, 0.61127605, 0.45310072, 0.23922315,
              0.88947289, 0.79262311, 0.90328563, 0.90403395, 0.55462119,
              0.71169526, 0.44339384, 0.4229734 , 0.37359708, 0.5605615 ],
             [0.29154175, 0.53346401, 0.69707002, 0.59000001, 0.36876511,
              0.72715984, 0.24791295, 0.38871834, 0.28864685, 0.79487154,
              0.81343809, 0.20930706, 0.29654442, 0.60612339, 0.56427789,
              0.61137697, 0.77354028, 0.29659134, 0.68202226, 0.33597763],
             [0.80454045, 0.6064233 , 0.27953981, 0.158827  , 0.3456    ,
              0.25805576, 0.74655738, 0.6085076 , 0.51818817, 0.55123353,
              0.3405792 , 0.36883447, 0.57003872, 0.43790018, 0.5090451 ,
              0.51542348, 0.52307234, 0.34267424, 0.41435288, 0.33553291]]
        )
        updated_centroids = compute_centroids(X, c, K)
        
        self.assertEqual(updated_centroids.shape, (5, 20))
        self.assertTrue(np.all(np.isclose(updated_centroids, expected_centroids)))

class test_kmeans_cluster(unittest.TestCase):
    def setUp(self):
        self.X = task_1_1_load_data()

    def test_assg_data(self):
        K = 3
        expected_labels = np.array(
            [2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]            
        )
        labels, history = kmeans_cluster(self.X, K, num_iter=10)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(labels.shape, (300,))
        self.assertTrue(np.all(np.isclose(labels, expected_labels)))

        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 11)

        expected_initial_centroids = np.array(
            [[5.64034678, 2.69385282],
             [8.20339815, 2.41693495],
             [2.4051802 , 1.11778123]]            
        )
        self.assertTrue(np.all(np.isclose(history[0], expected_initial_centroids)))

        expected_final_centroids = np.array(
            [[3.04367119, 1.01541041],
             [6.03366736, 3.00052511],
            [1.95399466, 5.02557006]]        
        )
        self.assertTrue(np.all(np.isclose(history[10], expected_final_centroids)))


class test_task_2_1_load_data(unittest.TestCase):
    def setUp(self):
        self.X = task_2_1_load_data()

    def test_loaded_types(self):
        self.assertIsInstance(self.X, np.ndarray)

    def test_X_properties(self):
        self.assertEqual(self.X.shape, (50, 2))


class test_feature_normalize(unittest.TestCase):
    def setUp(self):
        pass

    def test_assg_data(self):
        X = task_2_1_load_data()
        X_norm, mu, sigma = feature_normalize(X)

        # test X_norm properties
        self.assertIsInstance(X_norm, np.ndarray)
        self.assertEqual(X_norm.shape, (50,2))
        self.assertTrue(np.isclose(X_norm.mean(), 0.0))
        self.assertTrue(np.isclose(X_norm.std(), 1.0))

        # test expected mu and sigma were found from original data
        expected_mu = np.array([3.98926528, 5.00280585])
        self.assertTrue(np.all(np.isclose(mu, expected_mu)))

        expected_sigma = np.array([1.16126017, 1.01312201])
        self.assertTrue(np.all(np.isclose(sigma, expected_sigma)))

    def test_random_data(self):
        np.random.seed(42)
        X = np.random.rand(1000, 5)
        X_norm, mu, sigma = feature_normalize(X)

        # test X_norm properties
        self.assertIsInstance(X_norm, np.ndarray)
        self.assertEqual(X_norm.shape, (1000,5))
        self.assertTrue(np.isclose(X_norm.mean(), 0.0))
        self.assertTrue(np.isclose(X_norm.std(), 1.0))

        # test expected mu and sigma were found from original data
        expected_mu = np.array([0.4963718 , 0.50008531, 0.48985751, 0.49780181, 0.50004354])
        self.assertTrue(np.all(np.isclose(mu, expected_mu)))

        expected_sigma = np.array([0.2901722 , 0.29694818, 0.28486009, 0.28686249, 0.28891208])
        self.assertTrue(np.all(np.isclose(sigma, expected_sigma)))


class test_pca(unittest.TestCase):
    def setUp(self):
        pass

    def test_assg_data(self):
        X = task_2_1_load_data()
        U, S, mu, sigma = pca(X)
        
        # test U principal components
        self.assertEqual(U.shape, (2,2))
        expected_U = np.array(
            [[-0.70710678, -0.70710678],
             [-0.70710678,  0.70710678]]            
        )
        self.assertTrue(np.all(np.isclose(U, expected_U)))

        # test S explained variances vectors
        self.assertEqual(S.shape, (2,))
        expected_S = np.array([1.73553038, 0.26446962])
        self.assertTrue(np.all(np.isclose(S, expected_S)))

        # test expected mu and sigma were found from original data
        expected_mu = np.array([3.98926528, 5.00280585])
        self.assertTrue(np.all(np.isclose(mu, expected_mu)))

        expected_sigma = np.array([1.16126017, 1.01312201])
        self.assertTrue(np.all(np.isclose(sigma, expected_sigma)))


class test_project_data(unittest.TestCase):
    def setUp(self):
        pass

    def test_assg_data(self):
        X = task_2_1_load_data()
        X_norm, mu, sigma = feature_normalize(X)
        U, S, mu, sigma = pca(X)
        K = 1
        Z = project_data(X_norm, U, K)

        self.assertEqual(Z.shape, (50,1))
        expected_Z = np.array(
            [[ 1.49631261],
            [-0.92218067],
            [ 1.22439232],
            [ 1.64386173],
            [ 1.2732206 ],
            [-0.97681976],
            [ 1.26881187],
            [-2.34148278],
            [-0.02999141],
            [-0.78171789],
            [-0.6316777 ],
            [-0.55280135],
            [-0.0896816 ],
            [-0.5258541 ],
            [ 1.56415455],
            [-1.91610366],
            [-0.88679735],
            [ 0.95607375],
            [-2.32995679],
            [-0.47793862],
            [-2.21747195],
            [ 0.38900633],
            [-1.78482346],
            [ 0.05175486],
            [ 1.66512392],
            [ 0.50813572],
            [-1.23711018],
            [-1.17198677],
            [ 0.84221686],
            [-0.00693174],
            [-0.22794195],
            [-1.51309518],
            [ 1.33874082],
            [-0.5925244 ],
            [ 0.67907605],
            [-1.35298   ],
            [ 1.68749495],
            [-1.39235931],
            [ 2.55992598],
            [-0.27850702],
            [-0.97677692],
            [ 0.88820006],
            [ 1.29666127],
            [-0.98966774],
            [ 1.81272352],
            [-0.27196356],
            [ 3.19297722],
            [ 1.21299151],
            [ 0.36792871],
            [-1.44264131]]        )
        self.assertTrue(np.all(np.isclose(Z, expected_Z)))


class test_recover_data(unittest.TestCase):
    def setUp(self):
        pass

    def test_assg_data(self):
        X = task_2_1_load_data()
        X_norm, mu, sigma = feature_normalize(X)
        U, S, mu, sigma = pca(X)
        K = 1
        Z = project_data(X_norm, U, K)
        X_rec = recover_data(Z, U)

        self.assertEqual(X_rec.shape, (50,2))
        expected_X_rec = np.array(
            [[-1.05805279, -1.05805279],
            [ 0.65208021,  0.65208021],
            [-0.86577611, -0.86577611],
            [-1.16238578, -1.16238578],
            [-0.90030292, -0.90030292],
            [ 0.69071588,  0.69071588],
            [-0.89718548, -0.89718548],
            [ 1.65567835,  1.65567835],
            [ 0.02120713,  0.02120713],
            [ 0.55275802,  0.55275802],
            [ 0.44666359,  0.44666359],
            [ 0.39088959,  0.39088959],
            [ 0.06341447,  0.06341447],
            [ 0.371835  ,  0.371835  ],
            [-1.10602429, -1.10602429],
            [ 1.35488989,  1.35488989],
            [ 0.62706042,  0.62706042],
            [-0.67604623, -0.67604623],
            [ 1.64752825,  1.64752825],
            [ 0.33795364,  0.33795364],
            [ 1.56798945,  1.56798945],
            [-0.27506901, -0.27506901],
            [ 1.26206077,  1.26206077],
            [-0.03659622, -0.03659622],
            [-1.17742041, -1.17742041],
            [-0.35930621, -0.35930621],
            [ 0.874769  ,  0.874769  ],
            [ 0.82871979,  0.82871979],
            [-0.59553725, -0.59553725],
            [ 0.00490148,  0.00490148],
            [ 0.1611793 ,  0.1611793 ],
            [ 1.06991986,  1.06991986],
            [-0.94663271, -0.94663271],
            [ 0.41897802,  0.41897802],
            [-0.48017928, -0.48017928],
            [ 0.95670134,  0.95670134],
            [-1.19323912, -1.19323912],
            [ 0.98454671,  0.98454671],
            [-1.81014102, -1.81014102],
            [ 0.1969342 ,  0.1969342 ],
            [ 0.69068559,  0.69068559],
            [-0.62805228, -0.62805228],
            [-0.91687797, -0.91687797],
            [ 0.69980077,  0.69980077],
            [-1.28178909, -1.28178909],
            [ 0.19230728,  0.19230728],
            [-2.25777584, -2.25777584],
            [-0.85771452, -0.85771452],
            [-0.26016489, -0.26016489],
            [ 1.02010145,  1.02010145]]
        )
        self.assertTrue(np.all(np.isclose(X_rec, expected_X_rec)))

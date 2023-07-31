from unittest import TestCase
from Kalman import KF
import numpy as np


class TestKF(TestCase):
    def test_can_construct_x_v(self):
        x = 0.1
        v = 0.2
         
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)
        # self.assertAlmostEqual(kf.pos,(0.1,0.2))

    def test_can_predict(self):
        x = 0.1
        v = 0.2
         
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)
        kf.predict(dt=0.1)

    def test_shape_of_x_and_P_are_correct(self):
        x = 0.1
        v = 2.4
         
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)
        kf.predict(dt=0.1)

        self.assertEqual(kf.cov.shape,(4,4))
        self.assertEqual(kf.mean.shape,(4,1))

    def test_predict_increases_uncertainity(self):
        x = 0.1
        v = 0.2
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)
            self.assertGreater(det_after,det_before)

    def test_update_function_works(self):
        x = 0.1
        v = 0.2
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)

        kf.update(measured_x=23,measured_y=25)

    def test_update_decreases_uncertainity(self):
        kf = KF(acc_x_dir=0.1,acc_y_dir=0.2,std_acc=0.5 ,x_std_meas=0.1,y_std_meas=0.1)
        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.update(measured_x=23+i,measured_y=25)
            det_after = np.linalg.det(kf.cov)
            self.assertLess(det_after,det_before)

"""
集束调整代价函数模块
正确写法：继承 pyceres.CostFunction，手动实现 Evaluate
"""
import numpy as np
import pyceres


def _rodrigues(rvec):
    """Rodrigues 旋转向量 → 旋转矩阵，纯 numpy 实现"""
    theta = np.linalg.norm(rvec)
    if theta < 1e-10:
        return np.eye(3)
    r = rvec / theta
    r_cross = np.array([
        [    0, -r[2],  r[1]],
        [ r[2],     0, -r[0]],
        [-r[1],  r[0],     0]
    ])
    return (np.cos(theta) * np.eye(3)
            + (1 - np.cos(theta)) * np.outer(r, r)
            + np.sin(theta) * r_cross)


class ReprojectionCost(pyceres.CostFunction):
    """重投影误差代价函数"""

    def __init__(self, observed_pt: np.ndarray, K: np.ndarray):
        super().__init__()
        self.observed_x = float(observed_pt[0])
        self.observed_y = float(observed_pt[1])
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3])


    def Evaluate(self, parameters, residuals, jacobians):
        rvec  = np.array(parameters[0][:3], dtype=np.float64)
        tvec  = np.array(parameters[0][3:], dtype=np.float64)
        point = np.array(parameters[1],     dtype=np.float64)

        R     = _rodrigues(rvec)
        p_cam = R @ point + tvec
        Xc, Yc, Zc = p_cam

        if abs(Zc) < 1e-10:
            residuals[0] = 0.0
            residuals[1] = 0.0
            # ✅ 必须把 jacobians 也清零，否则 Ceres 读到垃圾内存
            if jacobians is not None:
                if jacobians[0] is not None:
                    jacobians[0][:] = 0.0
                if jacobians[1] is not None:
                    jacobians[1][:] = 0.0
            return True

        inv_z  = 1.0 / Zc
        inv_z2 = inv_z * inv_z

        residuals[0] = self.fx * Xc * inv_z + self.cx - self.observed_x
        residuals[1] = self.fy * Yc * inv_z + self.cy - self.observed_y

        if jacobians is not None:
            if jacobians[0] is not None:
                J = np.zeros((2, 6), dtype=np.float64)
                # 对平移 [tx, ty, tz]
                J[0, 3] =  self.fx * inv_z
                J[0, 4] =  0.0
                J[0, 5] = -self.fx * Xc * inv_z2
                J[1, 3] =  0.0
                J[1, 4] =  self.fy * inv_z
                J[1, 5] = -self.fy * Yc * inv_z2
                # 对旋转 [rx, ry, rz]：数值微分
                eps = 1e-7
                for k in range(3):
                    rv_p = rvec.copy(); rv_p[k] += eps
                    rv_m = rvec.copy(); rv_m[k] -= eps
                    p_p = _rodrigues(rv_p) @ point + tvec
                    p_m = _rodrigues(rv_m) @ point + tvec
                    J[0, k] = self.fx * (p_p[0]/p_p[2] - p_m[0]/p_m[2]) / (2*eps)
                    J[1, k] = self.fy * (p_p[1]/p_p[2] - p_m[1]/p_m[2]) / (2*eps)
                # ✅ 用 np.copyto 赋值，避免 shape 广播错误
                np.copyto(jacobians[0], J.flatten())

            if jacobians[1] is not None:
                J = np.zeros((2, 3), dtype=np.float64)
                for k in range(3):
                    J[0, k] = self.fx * (R[0, k] * inv_z - Xc * R[2, k] * inv_z2)
                    J[1, k] = self.fy * (R[1, k] * inv_z - Yc * R[2, k] * inv_z2)
                # ✅ 用 np.copyto 赋值
                np.copyto(jacobians[1], J.flatten())

        return True


class GPSPositionCost(pyceres.CostFunction):
    """GPS位置先验代价函数"""

    def __init__(self, gps_position: np.ndarray, weight: float = 1.0):
        super().__init__()
        self.gps    = np.array(gps_position, dtype=np.float64)
        self.weight = float(weight)
        self.set_num_residuals(3)
        self.set_parameter_block_sizes([6])


    def Evaluate(self, parameters, residuals, jacobians):
        rvec = np.array(parameters[0][:3], dtype=np.float64)
        tvec = np.array(parameters[0][3:], dtype=np.float64)

        R = _rodrigues(rvec)
        camera_center = -R.T @ tvec

        err = (camera_center - self.gps) * self.weight
        residuals[0] = err[0]
        residuals[1] = err[1]
        residuals[2] = err[2]

        if jacobians is not None and jacobians[0] is not None:
            J   = np.zeros((3, 6), dtype=np.float64)
            eps = 1e-7
            params = np.concatenate([rvec, tvec])
            for k in range(6):
                p_p = params.copy(); p_p[k] += eps
                p_m = params.copy(); p_m[k] -= eps
                c_p = -_rodrigues(p_p[:3]).T @ p_p[3:]
                c_m = -_rodrigues(p_m[:3]).T @ p_m[3:]
                J[:, k] = self.weight * (c_p - c_m) / (2*eps)
            # ✅ 用 np.copyto 赋值
            np.copyto(jacobians[0], J.flatten())

        return True
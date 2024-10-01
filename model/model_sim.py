import numpy as np
import casadi as ca


class Quadrotor_Sim:
    def __init__(self):
        """
        Quadrotor model with a modified Mellinger controller
        (Differential Flatness-based Controller)

        The open-loop model is are more realistic:
            1. Inertia and Mass Bias
            2. Aerodynamics
            3. Others if needed
        """
        self.__dynamic_param()
        self.__control_param()
        self.__saturation_params()

        # Model Information
        self.ol_x_dim = 12  # Open-loop Model state dim
        self.x_dim = 18
        self.u_dim = 4
        self.ref_dim = 12

        # Disturbance
        self.disturb = np.zeros(6)

    def __dynamic_param(self):
        self.m = 0.83
        self.m_actual = self.m
        self.Ixx = 3e-3
        self.Iyy = 3e-3
        self.Izz = 4e-3
        self.Ixx_actual = self.Ixx
        self.Iyy_actual = self.Iyy
        self.Izz_actual = self.Izz
        self.__compute_J()

        torque_coef = 0.01
        arm_length = 0.150
        self.CoeffM = np.array(
            [
                [
                    0.25,
                    -0.353553 / arm_length,
                    0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    -0.353553 / arm_length,
                    0.25 / torque_coef,
                ],
                [
                    0.25,
                    0.353553 / arm_length,
                    0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
                [
                    0.25,
                    -0.353553 / arm_length,
                    -0.353553 / arm_length,
                    -0.25 / torque_coef,
                ],
            ]
        )  # Ttau -> MF
        self.CoeffM_inv = np.linalg.inv(self.CoeffM)  # MF -> Ttau
        self.aero_D = np.diag(np.zeros(3))

    def __compute_J(self):
        self.J = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        self.J_actual = np.diag(
            np.array([self.Ixx_actual, self.Iyy_actual, self.Izz_actual])
        )
        self.J_inv = np.diag(np.array([1 / self.Ixx, 1 / self.Iyy, 1 / self.Izz]))
        self.J_actual_inv = np.diag(
            np.array([1 / self.Ixx_actual, 1 / self.Iyy_actual, 1 / self.Izz_actual])
        )

    def __control_param(self):
        self.pos_gain = np.diag(np.array([1.0, 1.0, 0.7])) * 2
        self.vel_gain = self.pos_gain * 4
        self.eul_gain = np.diag(np.array([10.0, 10.0, 4.0]))
        self.omega_P = np.diag(np.array([40.0, 40.0, 16.0]))
        self.omega_I = np.diag(np.array([10.0, 10.0, 5.0]))
        self.omgea_D = np.diag(np.array([0.5, 0.5, 0.0]))

    def __saturation_params(self):
        self.u_lb = np.array([0.0, 0.0, 0.0, 0.0])
        self.u_ub = np.array([4.0, 4.0, 4.0, 4.0]) * 1.5

    def __rpy2DCMbe_sym(self, eul_rpy):
        # eul_zyx <- eul_rpy 1,3 switch
        eul_z = eul_rpy[2]
        eul_y = eul_rpy[1]
        eul_x = eul_rpy[0]
        R1 = ca.vertcat(
            ca.horzcat(ca.cos(eul_z), -ca.sin(eul_z), 0),
            ca.horzcat(ca.sin(eul_z), ca.cos(eul_z), 0),
            ca.horzcat(0, 0, 1),
        )
        R2 = ca.vertcat(
            ca.horzcat(ca.cos(eul_y), 0, ca.sin(eul_y)),
            ca.horzcat(0, 1, 0),
            ca.horzcat(-ca.sin(eul_y), 0, ca.cos(eul_y)),
        )
        R3 = ca.vertcat(
            ca.horzcat(1, 0, 0),
            ca.horzcat(0, ca.cos(eul_x), -ca.sin(eul_x)),
            ca.horzcat(0, ca.sin(eul_x), ca.cos(eul_x)),
        )
        return R1 @ R2 @ R3

    def __rpy2DCMbe_num(self, eul_rpy):
        # eul_zyx <- eul_rpy 1,3 switch
        eul_z = eul_rpy[2]
        eul_y = eul_rpy[1]
        eul_x = eul_rpy[0]
        R1 = np.vstack(
            [
                np.hstack([np.cos(eul_z), -np.sin(eul_z), 0]),
                np.hstack([np.sin(eul_z), np.cos(eul_z), 0]),
                np.hstack([0, 0, 1]),
            ]
        )
        R2 = np.vstack(
            [
                np.hstack([np.cos(eul_y), 0, np.sin(eul_y)]),
                np.hstack([0, 1, 0]),
                np.hstack([-np.sin(eul_y), 0, np.cos(eul_y)]),
            ]
        )
        R3 = np.vstack(
            [
                np.hstack([1, 0, 0]),
                np.hstack([0, np.cos(eul_x), -np.sin(eul_x)]),
                np.hstack([0, np.sin(eul_x), np.cos(eul_x)]),
            ]
        )
        return R1 @ R2 @ R3

    def __dEul2omega_sym(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (ca.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * ca.cos(Eul[0])) + (
            dEul_des[2] * ca.sin(Eul[0]) * ca.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * ca.sin(Eul[0])) + (
            dEul_des[2] * ca.cos(Eul[0]) * ca.cos(Eul[1])
        )
        return ca.vertcat(domega_xdes, domega_ydes, domega_zdes)

    def __dEul2omega_num(self, dEul_des, Eul):
        # Strap Down Equations
        domega_xdes = dEul_des[0] - (np.sin(Eul[1]) * dEul_des[2])
        domega_ydes = (dEul_des[1] * np.cos(Eul[0])) + (
            dEul_des[2] * np.sin(Eul[0]) * np.cos(Eul[1])
        )
        domega_zdes = -(dEul_des[1] * np.sin(Eul[0])) + (
            dEul_des[2] * np.cos(Eul[0]) * np.cos(Eul[1])
        )
        return np.hstack([domega_xdes, domega_ydes, domega_zdes])

    def __invert_eul_sym(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return ca.vertcat(m1, m2, m3)

    def __invert_eul_num(self, moment_des, omega):
        m1 = moment_des[0] + omega[1] * omega[2] * (self.Izz - self.Iyy)
        m2 = moment_des[1] + omega[0] * omega[2] * (self.Ixx - self.Izz)
        m3 = moment_des[2] + omega[0] * omega[1] * (self.Iyy - self.Ixx)
        return np.hstack([m1, m2, m3])

    def __derative3(self, interstate, x):
        # c=0.05 tf = s/(c*s+1)
        d_interstate = -10 * np.eye(3) @ interstate + 8 * np.eye(3) @ x
        x_der = -12.5 * np.eye(3) @ interstate + 10 * np.eye(3) @ x
        return d_interstate, x_der

    def nominal_sym(self, x, MF):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m_actual
            * (ca.cos(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) + ca.sin(x[8]) * ca.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m_actual
            * (ca.sin(x[8]) * ca.sin(x[7]) * ca.cos(x[6]) - ca.cos(x[8]) * ca.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m_actual * (ca.cos(x[6]) * ca.cos(x[7]))
        R = self.__rpy2DCMbe_sym(x[6:9])
        dv = (
            ca.vertcat(dvx, dvy, dvz)
            - R @ self.aero_D @ R.T @ dp
            + self.disturb[0:3] / self.m_actual
        )

        deul = (
            ca.vertcat(
                ca.horzcat(1, ca.tan(x[7]) * ca.sin(x[6]), ca.tan(x[7]) * ca.cos(x[6])),
                ca.horzcat(0, ca.cos(x[6]), -ca.sin(x[6])),
                ca.horzcat(0, ca.sin(x[6]) / ca.cos(x[7]), ca.cos(x[6]) / ca.cos(x[7])),
            )
            @ x[9:12]
        )

        domega = self.J_actual_inv @ (
            -ca.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4] + self.disturb[3:6]
        )

        return ca.vertcat(dp, dv, deul, domega)

    def nominal_num(self, x, MF):
        Ttau = self.CoeffM_inv @ MF

        dp = x[3:6]

        dvx = (
            -Ttau[0]
            / self.m_actual
            * (np.cos(x[8]) * np.sin(x[7]) * np.cos(x[6]) + np.sin(x[8]) * np.sin(x[6]))
        )
        dvy = (
            -Ttau[0]
            / self.m_actual
            * (np.sin(x[8]) * np.sin(x[7]) * np.cos(x[6]) - np.cos(x[8]) * np.sin(x[6]))
        )
        dvz = 9.8 - Ttau[0] / self.m_actual * (np.cos(x[6]) * np.cos(x[7]))

        R = self.__rpy2DCMbe_num(x[6:9])
        dv = (
            np.array([dvx, dvy, dvz])
            - R @ self.aero_D @ R.T @ dp
            + self.disturb[0:3] / self.m_actual
        )

        deul = (
            np.vstack(
                [
                    np.hstack(
                        [1, np.tan(x[7]) * np.sin(x[6]), np.tan(x[7]) * np.cos(x[6])]
                    ),
                    np.hstack([0, np.cos(x[6]), -np.sin(x[6])]),
                    np.hstack(
                        [0, np.sin(x[6]) / np.cos(x[7]), np.cos(x[6]) / np.cos(x[7])]
                    ),
                ]
            )
            @ x[9:12]
        )

        domega = self.J_actual_inv @ (
            -np.cross(x[9:12], self.J @ x[9:12]) + Ttau[1:4] + self.disturb[3:6]
        )

        return np.hstack([dp, dv, deul, domega])

    def ctrlmap_sym(self, x, CMD, omega_err_inte, omega_err_ds):
        # demux
        pos = x[0:3]
        vel = x[3:6]
        eul = x[6:9]
        omega = x[9:12]
        pos_CMD = CMD[0:3]
        vel_CMD = CMD[3:6]
        acc_CMD = CMD[6:9]
        jer_CMD = CMD[9:12]

        # Translational Loop
        a_des = acc_CMD + self.vel_gain @ (
            self.pos_gain @ (pos_CMD - pos) + vel_CMD - vel
        )

        # Obtain Desire Rebs
        Zb = ca.vertcat(-a_des[0], -a_des[1], 9.8 - a_des[2]) / ca.norm_2(
            ca.vertcat(-a_des[0], -a_des[1], 9.8 - a_des[2])
        )

        Xc = ca.vertcat(ca.cos(0.0), ca.sin(0.0), 0.0)
        Yb_ = ca.cross(Zb, Xc)
        Yb = Yb_ / ca.norm_2(Yb_)
        Xb = ca.cross(Yb, Zb)
        # Reb_des = np.vstack([Xb, Yb, Zb]).T
        Reb_des = ca.horzcat(Xb, Yb, Zb)

        # Obtain Desire Eul, Omega and dOmega
        # eul_des = 'ZYX' + 1,3 Switch
        eul_des = ca.vertcat(
            ca.arctan2(Reb_des[2, 1], Reb_des[2, 2]),
            ca.arctan2(
                -Reb_des[2, 0], ca.sqrt(Reb_des[1, 0] ** 2 + Reb_des[0, 0] ** 2)
            ),
            ca.arctan2(Reb_des[1, 0], Reb_des[0, 0]),
        )

        T = -self.m * ca.dot(Zb, (a_des - np.array([0.0, 0.0, 9.8])))
        h1 = -self.m / T * (jer_CMD - ca.dot(Zb, jer_CMD) * Zb)

        omega_des = ca.vertcat(-ca.dot(h1, Yb), ca.dot(h1, Xb), 0.0)

        # h2 = (
        #     -ca.cross(omega_des, ca.cross(omega_des, Zb))
        #     + self.m / T * ca.dot(jer_CMD, Zb) * ca.cross(omega_des, Zb)
        #     + 2 * self.m / T * ca.dot(Zb, jer_CMD) * ca.cross(omega_des, Zb)
        # )
        # domega_des = ca.vertcat(-ca.dot(h2, Yb), ca.dot(h2, Xb), 0.0)

        # Attitude Loop
        dEul_des = self.eul_gain @ (eul_des - eul)
        omega_err = omega_des - omega + self.__dEul2omega_sym(dEul_des, eul)
        d_omega_err_ds, omega_err_der = self.__derative3(omega_err_ds, omega_err)
        att_out = (
            self.omega_P @ omega_err
            + self.omega_I @ omega_err_inte
            + self.omgea_D @ omega_err_der
            + ca.cross(omega, self.J @ omega)
            # + self.J @ domega_des
        )
        moment_des = self.J @ att_out
        tau = self.__invert_eul_sym(moment_des, omega)

        return (
            self.CoeffM @ ca.vertcat(T, tau[0], tau[1], tau[2]),
            omega_err,
            d_omega_err_ds,
        )

    def ctrlmap_num(self, x, CMD, omega_err_inte, omega_err_ds):
        # demux
        pos = x[0:3]
        vel = x[3:6]
        eul = x[6:9]
        omega = x[9:12]
        pos_CMD = CMD[0:3]
        vel_CMD = CMD[3:6]
        acc_CMD = CMD[6:9]
        jer_CMD = CMD[9:12]

        # Translational Loop
        a_des = (
            acc_CMD + self.pos_gain @ (pos_CMD - pos) + self.vel_gain @ (vel_CMD - vel)
        )
        # Obtain Desire Rebs
        Zb = np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]]) / np.linalg.norm(
            np.hstack([-a_des[0], -a_des[1], 9.8 - a_des[2]])
        )

        Xc = np.hstack([np.cos(0.0), np.sin(0.0), 0.0])
        Yb_ = np.cross(Zb, Xc)
        Yb = Yb_ / np.linalg.norm(Yb_)
        Xb = np.cross(Yb, Zb)
        Reb_des = np.vstack([Xb, Yb, Zb]).T
        # Reb_des = ca.horzcat(Xb, Yb, Zb)

        # Obtain Desire Eul, Omega and dOmega
        # eul_des = 'ZYX' + 1,3 Switch
        eul_des = np.hstack(
            [
                np.arctan2(Reb_des[2, 1], Reb_des[2, 2]),
                np.arctan2(
                    -Reb_des[2, 0], np.sqrt(Reb_des[1, 0] ** 2 + Reb_des[0, 0] ** 2)
                ),
                np.arctan2(Reb_des[1, 0], Reb_des[0, 0]),
            ]
        )

        T = -self.m * np.dot(Zb, (a_des - np.hstack([0.0, 0.0, 9.8])))
        h1 = -self.m / T * (jer_CMD - np.dot(Zb, jer_CMD) * Zb)

        omega_des = np.hstack([-np.dot(h1, Yb), np.dot(h1, Xb), 0.0])

        # h2 = (
        #     -np.cross(omega_des, np.cross(omega_des, Zb))
        #     + self.m / T * np.dot(jer_CMD, Zb) * np.cross(omega_des, Zb)
        #     + 2 * self.m / T * np.dot(Zb, jer_CMD) * np.cross(omega_des, Zb)
        # )
        # domega_des = np.array(
        #     [-np.dot(h2, Yb), np.dot(h2, Xb), 0.0]
        # )  # yaw, dyaw, ddyaw = 0

        # Attitude Loop
        dEul_des = self.eul_gain @ (eul_des - eul)
        omega_err = omega_des - omega + self.__dEul2omega_num(dEul_des, eul)
        d_omega_err_ds, omega_err_der = self.__derative3(omega_err_ds, omega_err)
        att_out = (
            self.omega_P @ omega_err
            + self.omega_I @ omega_err_inte
            + self.omgea_D @ omega_err_der
            + np.cross(omega, self.J @ omega)
            # + self.J @ domega_des
        )
        moment_des = self.J @ att_out
        tau = self.__invert_eul_num(moment_des, omega)

        return (
            self.CoeffM @ np.hstack([T, tau[0], tau[1], tau[2]]),
            omega_err,
            d_omega_err_ds,
        )

    def cldyn_sym(self, x, CMD):
        MF, omega_err, d_omega_err_ds = self.ctrlmap_sym(
            x[0:12], CMD, x[12:15], x[15:18]
        )
        dx = self.nominal_sym(x, MF)
        return ca.vertcat(dx, omega_err, d_omega_err_ds)

    def cldyn_num(self, x, CMD):
        MF, omega_err, d_omega_err_ds = self.ctrlmap_num(
            x[0:12], CMD, x[12:15], x[15:18]
        )
        dx = self.nominal_num(x, MF)
        return np.hstack([dx, omega_err, d_omega_err_ds])

    def openloop_sym_exRK4(self, xk, MF, dt):
        # Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.nominal_sym(xk, MF)
        k2 = self.nominal_sym((xk + 0.5 * h * k1), MF)
        k3 = self.nominal_sym((xk + 0.5 * h * k2), MF)
        k4 = self.nominal_sym((xk + h * k3), MF)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def openloop_num_exRK4(self, xk, MF, dt):
        # Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.nominal_num(xk, MF)
        k2 = self.nominal_num((xk + 0.5 * h * k1), MF)
        k3 = self.nominal_num((xk + 0.5 * h * k2), MF)
        k4 = self.nominal_num((xk + h * k3), MF)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def cldyn_sym_exRK4(self, xk, CMD, dt):
        # Closed-loop Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.cldyn_sym(xk, CMD)
        k2 = self.cldyn_sym((xk + 0.5 * h * k1), CMD)
        k3 = self.cldyn_sym((xk + 0.5 * h * k2), CMD)
        k4 = self.cldyn_sym((xk + h * k3), CMD)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def cldyn_num_exRK4(self, xk, CMD, dt):
        # Closed-loop Nominal Dynamics: RK4 Discrete
        h = dt
        k1 = self.cldyn_num(xk, CMD)
        k2 = self.cldyn_num((xk + 0.5 * h * k1), CMD)
        k3 = self.cldyn_num((xk + 0.5 * h * k2), CMD)
        k4 = self.cldyn_num((xk + h * k3), CMD)
        xk1 = xk + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return xk1

    def controller_sym(self, xk, refk):
        u = self.ctrlmap_sym(xk[0 : self.ol_x_dim], refk, xk[12:15], xk[15:18])[0]
        return u

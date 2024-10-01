import numpy as np
import casadi as ca

# Simulation & Control Setup
SIMSTEPS = 300
TIMESTEP = 0.02
HORIZON = 10
TRAJ_SPEEDRATE = 1.0

"""Import Models"""
from model.model_sim import Quadrotor_Sim  # FOR SIMULATION
from model.model_learn import Quadrotor_Learn  # FOR MPC
from model.model_nominal import Quadrotor  # FOR MPC
from mpc.solver import *

# add bias in aerodynamic effects, mass and inertia
model_sim = Quadrotor_Sim()
model_sim.disturb = np.array([0.3] * 3 + [0.0] * 3)
model_sim.aero_D = np.diag([0.6, 0.6, 0.15])
model_sim.m_actual += 0.5
model_sim.Ixx_actual += 2e-3
model_sim.Iyy_actual += 2e-3

# Prepare model for model predictive control
model_nominal = Quadrotor()

model_learn = Quadrotor_Learn(x_dim=18, ol_x_dim=12, aux_input_dim=4, discrete_h=0.02)
model_learn.load_params(params=ca.DM(np.load("learning/temp/model_param.npy")))

# Prepare mpc controllers
mpc_standard_nominal = MPC(model=model_nominal, discrete_h=TIMESTEP, H=HORIZON)
mpc_standard_learned = MPC(model=model_learn, discrete_h=TIMESTEP, H=HORIZON)
mpc_multistep_nominal = MPC_MultiStep(
    model=model_nominal, discrete_h=TIMESTEP, H=HORIZON
)
mpc_multistep_learned = MPC_MultiStep(model=model_learn, discrete_h=TIMESTEP, H=HORIZON)
mpc_multistep_nominal.set_GainAndDecay(L_init=np.diag([3.0] * 12), decay_rate=0.1)
mpc_multistep_learned.set_GainAndDecay(L_init=np.diag([3.0] * 12), decay_rate=0.1)


"""Simulation Setup: Tracking 3D Lissajous Trajectory"""
from aux_module.trajectory_tools import coeff_to_pointsLissajous

# Get initial state for trajectory tracking
# using differential flatness mapping
pvaj_0 = coeff_to_pointsLissajous(
    0.0,
    a=1.0,
    a0=TRAJ_SPEEDRATE,
    Radi=np.array([3.0, 3.0, 0.5]),
    Period=np.array([6.0, 3.0, 3.0]),
    h=-0.5,
)
x0 = model_nominal.cmd2x_map(pvaj_0)

# Get position reference trajectory for plotting
pos_reference = []
for i in range(SIMSTEPS):
    pvaj = coeff_to_pointsLissajous(
        i * TIMESTEP,
        a=1.0,
        a0=TRAJ_SPEEDRATE,
        Radi=np.array([3.0, 3.0, 0.5]),
        Period=np.array([6.0, 3.0, 3.0]),
        h=-0.5,
    )
    pos_reference += [pvaj[:3].reshape((-1, 1))]
pos_reference = np.hstack(pos_reference)


# Obtaining the upcomming state and input reference trajectories
def get_XUref(t):
    xrefk_seq = []
    urefk_seq = []
    for i in range(HORIZON):
        pvaj = coeff_to_pointsLissajous(
            t + i * TIMESTEP,
            a=1.0,
            a0=TRAJ_SPEEDRATE,
            Radi=np.array([3.0, 3.0, 0.5]),
            Period=np.array([6.0, 3.0, 3.0]),
            h=-0.5,
        )
        # Compute x, u reference using differential flatness mapping (pvaj -> x, u)
        xrefk_seq += [model_nominal.cmd2x_map(pvaj).reshape((-1, 1))]
        urefk_seq += [model_nominal.cmd2u_map(pvaj).reshape((-1, 1))]

    return np.hstack(xrefk_seq), np.hstack(urefk_seq)


# Simulation Main Function
def sim_main(idx, controller):
    dx_real = []
    dx_pred = []
    lossk_seq = []
    t = 0
    xk = np.squeeze(x0, axis=1)
    xsim_seq = [x0]
    # Refresh predictor in mpc_multistep
    if idx >= 3:
        controller.refresh_predictor()
    for i in range(SIMSTEPS):  # main loop for simulation
        if i % 10 == 0:
            print("Current Case: {0}, step {1}/{2}".format(idx, i, SIMSTEPS))
        # Get reference trajectory
        xrefk_seq, urefk_seq = get_XUref(t)
        # Solve OC
        if idx <= 2:
            u_opt_seq = controller.solve_OC(xk, xrefk_seq, urefk_seq)[0]
        else:
            u_opt_seq = controller.solve_OC_MultiStepPred(xk, xrefk_seq, urefk_seq)[0]
        uk = u_opt_seq[:, 0]
        # Compute losskx
        lossk_seq += [controller.loss_kx(xk, xrefk_seq[:, 0])]
        # Rollout out simulation model
        xk1 = model_sim.openloop_sym_exRK4(xk, uk, TIMESTEP)
        # Get predicted dx v.s. real dx
        dx_real_i = model_sim.nominal_sym(xk, uk)
        dx_pred_i = controller.dx_modelpredict(xk, uk)
        dx_real += [dx_real_i]
        dx_pred += [dx_pred_i]
        # Store sim state
        xsim_seq += [xk1.reshape((-1, 1))]
        # Update sim state
        xk = xk1
        # Update time step
        t += TIMESTEP

    return np.hstack(xsim_seq), np.hstack(dx_real), np.hstack(dx_pred), np.array(lossk_seq)


"""Go Simulation and Visualization"""
from scipy.io import savemat

controller_list = [
    (1, mpc_standard_nominal),
    (2, mpc_standard_learned),
    (3, mpc_multistep_nominal),
    (4, mpc_multistep_learned),
]


for idx, controller in controller_list:
    # Simulate
    xsim_seq, dx_real, dx_pred, lossk_seq = sim_main(idx, controller)

    # Save Trajectory
    savemat(
        "sim_mpc_trajs/sim_case{}.mat".format(idx),
        {
            "xsim_seq": xsim_seq,
            "pos_reference": pos_reference,
            "dx_real": dx_real,
            "dx_pred": dx_pred,
            "lossk_seq": lossk_seq
        },
    )

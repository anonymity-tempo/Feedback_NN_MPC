# Feedback Favors the Generalization of Neural ODEs

**Learning Aerodynamic Effects + Model Predictive Control using FNN**

This repository contains the code related to the **robotics application** of proposed feedback neural network. The aerodynamic effects are firstly learned using a neural ODE augmented model, and then embedded into a MPC controller with multi-step prediction method that utilizes feedbacks for higher precision accuracy and control performance.

| Directory                 | Introduction                                                 | Details                                                      |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Folder: **aux_module**    | Modules required for visualization and polynomial trajectory generation. | ---                                                          |
| Folder: **img**           | Storage of visualization results.                            | ---                                                          |
| Folder: **learning**      | The learning framework that allow training neural odes or neural ode augmented models with auxiliary inputs using minibatching. | **core.py** contains differentiable numerical integrators, defined loss functions and data-processing functions.  **trainer.py** contains the main loop of learning together with the analytic gradient computation algorithm. |
| Folder: **model**         | Models used for simulations and learning.                    | **model_learn.py**: the neural ODE augmented model for learning, **model_nominal.py**: a nominal model with no learning-based parts, **model_sim.py**: a model used for simulation, including more realistic settings. |
| Folder: **mpc**           | Include **solver.py** containing a standard MPC and a MPC with mutli-step prediction algorithm. | ---                                                          |
| Folder: **sim_mpc_trajs** | Storage of simulated trajectories of different MPC setups.   | ---                                                          |
| **1_mk_traindata.py**     | Make dataset for training.                                   | ---                                                          |
| **2_dynamics_learn.py**   | Learn the neural ODE augmented dynamics.                     | ---                                                          |
| **3_visualize_learn.py**  | Plot training trajectories, test results and loss evaluation. | ---                                                          |
| **4_mpc_cases.py**        | Trajectory tracking simulation with different MPC setups.    | ---                                                          |
| **5_visualize_mpc.py**    | Plot simulated trajectories of different MPC setups.         | ---                                                          |

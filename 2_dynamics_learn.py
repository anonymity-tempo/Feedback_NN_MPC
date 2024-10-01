import numpy as np
import casadi as ca

from learning.trainer import E2E_Trainer
from model.model_learn import Quadrotor_Learn


""" Neural ODE learning using mini-batching """

model = Quadrotor_Learn(discrete_h=0.02)
trainer = E2E_Trainer(model_wrapper=model, 
                      param_reg=args.param_reg # regularization
                      )

params_init = (np.random.rand(model.params_dim, 1) - 0.5) * 0.1

params = trainer.optimize_minibatch(
    params_init,
    step_init=1e-3,
    minibatch_size=100,
    epochs=50,
)

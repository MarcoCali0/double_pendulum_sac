import numpy as np
from model.symbolic_plant import SymbolicDoublePendulum


def get_swingup_time(
    T,
    X,
    eps=[1e-2, 1e-2, 1e-2, 1e-2],
    has_to_stay=True,
    mpar=None,
    method="height",
    height=0.9,
):
    """get_swingup_time.
    get the swingup time from a data_dict.

    Parameters
    ----------
    T : array-like
        time points, unit=[s]
        shape=(N,)
    X : array-like
        shape=(N, 4)
        states, units=[rad, rad, rad/s, rad/s]
        order=[angle1, angle2, velocity1, velocity2]
    U : array-like
        shape=(N, 2)
        actuations/motor torques
        order=[u1, u2],
        units=[Nm]
    eps : list
        list with len(eps) = 4. The thresholds for the swingup to be
        successfull ([position, velocity])
        default = [1e-2, 1e-2, 1e-2, 1e-2]
    has_to_stay : bool
        whether the pendulum has to stay upright until the end of the trajectory
        default=True

    Returns
    -------
    float
        swingup time
    """
    if method == "epsilon":
        goal = np.array([np.pi, 0.0, 0.0, 0.0])

        dist_x0 = np.abs(np.mod(X.T[0] - goal[0] + np.pi, 2 * np.pi) - np.pi)
        ddist_x0 = np.where(dist_x0 < eps[0], 0.0, dist_x0)
        n_x0 = np.argwhere(ddist_x0 == 0.0)

        dist_x1 = np.abs(np.mod(X.T[1] - goal[1] + np.pi, 2 * np.pi) - np.pi)
        ddist_x1 = np.where(dist_x1 < eps[1], 0.0, dist_x1)
        n_x1 = np.argwhere(ddist_x1 == 0.0)

        dist_x2 = np.abs(X.T[2] - goal[2])
        ddist_x2 = np.where(dist_x2 < eps[2], 0.0, dist_x2)
        n_x2 = np.argwhere(ddist_x2 == 0.0)

        dist_x3 = np.abs(X.T[3] - goal[3])
        ddist_x3 = np.where(dist_x3 < eps[3], 0.0, dist_x3)
        n_x3 = np.argwhere(ddist_x3 == 0.0)

        n = np.intersect1d(n_x0, n_x1)
        n = np.intersect1d(n, n_x2)
        n = np.intersect1d(n, n_x3)

        time_index = len(T) - 1
        if has_to_stay:
            if len(n) > 0:
                for i in range(len(n) - 2, 0, -1):
                    if n[i] + 1 == n[i + 1]:
                        time_index = n[i]
                    else:
                        break
        else:
            if len(n) > 0:
                time_index = n[0]
        time = T[time_index]
    elif method == "height":
        plant = SymbolicDoublePendulum(model_pars=mpar)
        fk = plant.forward_kinematics(X.T[:2])
        ee_pos_y = fk[1][1]

        goal_height = height * (mpar.l[0] + mpar.l[1])

        up = np.where(ee_pos_y > goal_height, True, False)

        time_index = len(T) - 1
        if has_to_stay:
            for i in range(len(up) - 2, 0, -1):
                if up[i]:
                    time_index = i
                else:
                    break

        else:
            time_index = np.argwhere(up)[0][0]
        time = T[time_index]

    else:
        time = np.inf

    return time

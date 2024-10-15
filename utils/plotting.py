import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


def plot(thetas1, thetas2, omegas1, omegas2, torques, robot, dt, t_final):
    # Unwrap angles
    thetas1 = np.unwrap(thetas1)
    thetas2 = np.unwrap(thetas2)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))

    # Plot theta1 and theta2 on the first subplot
    ax1.plot(thetas1, label="ϑ1")
    ax1.plot(thetas2, label="ϑ2")
    ax1.set_ylabel("q [rad]")

    # Plot angular velocities on the second subplot
    ax2.plot(omegas1, label="ω1")
    ax2.plot(omegas2, label="ω2")
    ax2.set_ylabel("ω [rad/s]")

    # Plot torque on the third subplot
    ax3.plot(torques, label="Torque")
    ax3.set_ylabel("Torque [Nm]")
    ax3.set_xlabel("Time [s]")

    max_steps = int(t_final / dt) + 1
    for ax in [ax1, ax2, ax3]:
        # Set tick positions at every 100 time steps
        ax.set_xticks([i for i in range(0, max_steps, int(1 / dt))])
        # Convert time steps to seconds
        ax.set_xticklabels([i / (1 / dt) for i in range(0, max_steps, int(1 / dt))])
        ax.legend()
        ax.grid()

    ax3.get_legend().remove()

    fig.suptitle(f"{robot.capitalize()} swing-up and stabilisation")
    plt.tight_layout()
    plt.show()


def rewards_plot(rewards, robot, desired_reward, dt, t_final):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(rewards)
    max_steps = int(t_final / dt) + 1
    ax.set_ylabel("Reward")
    ax.set_xlabel("Time Step")
    ax.axhline(y=desired_reward, color="r", linestyle="--", label=f"Desired Reward")
    ax.set_xticks([i for i in range(0, max_steps, int(1 / dt))])
    ax.set_xticklabels([i / (1 / dt) for i in range(0, max_steps, int(1 / dt))])
    ax.grid()
    fig.suptitle(f"{robot.capitalize()} Rewards")
    plt.tight_layout()
    plt.show()


def energy_plot(kinetic_energy, potential_energy, robot, dt, t_final):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(kinetic_energy, label="Kinetic Energy")
    ax.plot(potential_energy, label="Potential Energy")
    max_steps = int(t_final / dt) + 1
    ax.set_ylabel("Energy [J]")
    ax.set_xlabel("Time")
    ax.set_xticks([i for i in range(0, max_steps, int(1 / dt))])
    ax.set_xticklabels([i / (1 / dt) for i in range(0, max_steps, int(1 / dt))])
    ax.legend()
    ax.grid()
    fig.suptitle(f"{robot.capitalize()} Energy")
    plt.show()


def plot_timeseries(
    T,
    X=None,
    U=None,
    ACC=None,
    energy=None,
    plot_pos=True,
    plot_vel=True,
    plot_acc=False,
    plot_tau=True,
    plot_energy=False,
    pos_x_lines=[],
    pos_y_lines=[],
    vel_x_lines=[],
    vel_y_lines=[],
    acc_x_lines=[],
    acc_y_lines=[],
    tau_x_lines=[],
    tau_y_lines=[],
    energy_x_lines=[],
    energy_y_lines=[],
    T_des=None,
    X_des=None,
    U_des=None,
    X_meas=None,
    X_filt=None,
    U_con=None,
    U_friccomp=None,
    ACC_des=None,
    save_to=None,
    show=True,
    scale=1.0,
):
    n_subplots = np.sum([plot_pos, plot_vel, plot_tau, plot_acc, plot_energy])

    SMALL_SIZE = 16 * scale
    MEDIUM_SIZE = 20 * scale
    BIGGER_SIZE = 24 * scale

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(
        n_subplots, 1, figsize=(18 * scale, n_subplots * 3 * scale), sharex="all"
    )

    i = 0
    if plot_pos:
        ax[i].plot(T[: len(X)], np.asarray(X).T[0], label=r"$q_1$", color="blue")
        ax[i].plot(T[: len(X)], np.asarray(X).T[1], label=r"$q_2$", color="red")
        if not (X_des is None):
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[0],
                ls="--",
                label=r"$q_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[1],
                ls="--",
                label=r"$q_2$ desired",
                color="orange",
            )
        if not (X_meas is None):
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[0],
                ls="-",
                label=r"$q_1$ measured",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[1],
                ls="-",
                label=r"$q_2$ measured",
                color="red",
                alpha=0.2,
            )
        if not (X_filt is None):
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[0],
                ls="-",
                label=r"$q_1$ filtered",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[1],
                ls="-",
                label=r"$q_2$ filtered",
                color="brown",
            )
        for line in pos_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[:2]), np.max(X.T[:2])], ls="--", color="gray"
            )
        for line in pos_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("angle [rad]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_vel:
        i += 1
        ax[i].plot(T[: len(X)], np.asarray(X).T[2], label=r"$\dot{q}_1$", color="blue")
        ax[i].plot(T[: len(X)], np.asarray(X).T[3], label=r"$\dot{q}_2$", color="red")
        if not (X_des is None):
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[2],
                ls="--",
                label=r"$\dot{q}_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(X_des)],
                np.asarray(X_des).T[3],
                ls="--",
                label=r"$\dot{q}_2$ desired",
                color="orange",
            )
        if not (X_meas is None):
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[2],
                ls="-",
                label=r"$\dot{q}_1$ measured",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(X_meas)],
                np.asarray(X_meas).T[3],
                ls="-",
                label=r"$\dot{q}_2$ measured",
                color="red",
                alpha=0.2,
            )
        if not (X_filt is None):
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[2],
                ls="-",
                label=r"$\dot{q}_1$ filtered",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(X_filt)],
                np.asarray(X_filt).T[3],
                ls="-",
                label=r"$\dot{q}_2$ filtered",
                color="brown",
            )
        for line in vel_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[2:]), np.max(X.T[2:])], ls="--", color="gray"
            )
        for line in vel_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("velocity [rad/s]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_acc:
        i += 1
        ax[i].plot(
            T[: len(ACC)], np.asarray(ACC).T[0], label=r"$\ddot{q}_1$", color="blue"
        )
        ax[i].plot(
            T[: len(ACC)], np.asarray(ACC).T[1], label=r"$\ddot{q}_2$", color="red"
        )
        if not (ACC_des is None):
            ax[i].plot(
                T_des[: len(ACC_des)],
                np.asarray(ACC_des).T[0],
                ls="--",
                label=r"$\ddot{q}_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(ACC_des)],
                np.asarray(ACC_des).T[1],
                ls="--",
                label=r"$\ddot{q}_2$ desired",
                color="orange",
            )
        for line in acc_x_lines:
            ax[i].plot(
                [line, line], [np.min(X.T[2:]), np.max(X.T[2:])], ls="--", color="gray"
            )
        for line in acc_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("acceleration [rad/s^2]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_tau:
        i += 1
        ax[i].plot(
            T[: len(U)], np.asarray(U).T[0, : len(T)], label=r"$u_1$", color="blue"
        )
        ax[i].plot(
            T[: len(U)], np.asarray(U).T[1, : len(T)], label=r"$u_2$", color="red"
        )
        if not (U_des is None):
            ax[i].plot(
                T_des[: len(U_des)],
                np.asarray(U_des).T[0],
                ls="--",
                label=r"$u_1$ desired",
                color="lightblue",
            )
            ax[i].plot(
                T_des[: len(U_des)],
                np.asarray(U_des).T[1],
                ls="--",
                label=r"$u_2$ desired",
                color="orange",
            )
        if not (U_con is None):
            ax[i].plot(
                T[: len(U_con)],
                np.asarray(U_con).T[0],
                ls="-",
                label=r"$u_1$ controller",
                color="blue",
                alpha=0.2,
            )
            ax[i].plot(
                T[: len(U_con)],
                np.asarray(U_con).T[1],
                ls="-",
                label=r"$u_2$ controller",
                color="red",
                alpha=0.2,
            )
        if not (U_friccomp is None):
            ax[i].plot(
                T[: len(U_friccomp)],
                np.asarray(U_friccomp).T[0],
                ls="-",
                label=r"$u_1$ friction comp.",
                color="darkblue",
            )
            ax[i].plot(
                T[: len(U_friccomp)],
                np.asarray(U_friccomp).T[1],
                ls="-",
                label=r"$u_2$ friction comp.",
                color="brown",
            )
        for line in tau_x_lines:
            ax[i].plot([line, line], [np.min(U), np.max(U)], ls="--", color="gray")
        for line in tau_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("torque [Nm]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if plot_energy:
        i += 1
        ax[i].plot(T[: len(energy)], np.asarray(energy), label="energy")
        for line in energy_x_lines:
            ax[i].plot(
                [line, line], [np.min(energy), np.max(energy)], ls="--", color="gray"
            )
        for line in energy_y_lines:
            ax[i].plot([T[0], T[-1]], [line, line], ls="--", color="gray")
        ax[i].set_ylabel("energy [J]", fontsize=MEDIUM_SIZE)
        # ax[i].legend(loc="best")
        ax[i].legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    ax[i].set_xlabel("time [s]", fontsize=MEDIUM_SIZE)
    if not (save_to is None):
        plt.savefig(save_to, bbox_inches="tight")
    if show:
        plt.tight_layout()
        plt.show()
    plt.close()

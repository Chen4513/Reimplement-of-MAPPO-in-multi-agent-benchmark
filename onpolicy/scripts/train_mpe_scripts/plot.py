import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/home/xiyue/mappo/onpolicy/scripts/train_mpe_scripts"

envs = ["simple_reference", "simple_spread"]

methods = {
    "mappo": "MAPPO",
    "ippo": "IPPO",
    "MPE_qmix": "QMIX"
}

colors = {
    "mappo": "blue",
    "ippo": "orange",
    "MPE_qmix": "green"
}

SMOOTH = 40


# ===== Utils =====
def moving_avg(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def load_file(path):
    df = pd.read_csv(path)

    if "reward" in df.columns:
        reward_col = "reward"
    else:
        reward_col = "average_episode_rewards"

    df = df.groupby("timestep", as_index=False)[reward_col].mean()

    return df["timestep"].to_numpy(), df[reward_col].to_numpy()


fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for idx, env in enumerate(envs):
    ax = axes[idx]

    for key in methods.keys():
        paths = glob.glob(os.path.join(BASE_DIR, env, f"{key}_*.csv"))

        if len(paths) == 0:
            print(f"Skip {env}-{key}")
            continue

        runs = []
        timesteps_ref = None

        for p in paths:
            t, r = load_file(p)

            if timesteps_ref is None:
                timesteps_ref = t

            runs.append(r)

        min_len = min([len(r) for r in runs])
        runs = [r[:min_len] for r in runs]
        timesteps_ref = timesteps_ref[:min_len]

        runs = np.stack(runs)

        # smoothing
        runs_s = np.array([moving_avg(r, SMOOTH) for r in runs])

        mean = runs_s.mean(axis=0)
        std = runs_s.std(axis=0)

        ax.plot(timesteps_ref, mean, label=methods[key], color=colors[key])
        ax.fill_between(timesteps_ref, mean - std, mean + std, alpha=0.2, color=colors[key])

    ax.set_title(env.replace("_", " ").title())
    ax.set_xlabel("Environment Steps")
    ax.grid(False)

axes[0].set_ylabel("Episode Reward")
handles, labels = axes[0].get_legend_handles_labels()

# 去重（保持顺序）
unique = []
seen = set()
for h, l in zip(handles, labels):
    if l not in seen:
        unique.append((h, l))
        seen.add(l)

handles, labels = zip(*unique)

fig.legend(handles, labels,
           loc="upper center",
           ncol=3)

# fig.legend(loc="upper center", ncol=3)
plt.tight_layout()
plt.subplots_adjust(top=0.85)

plt.savefig("mpe_learning_curve.png", dpi=300)
plt.show()
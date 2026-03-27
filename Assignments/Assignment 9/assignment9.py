import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy







def main():
    map_name = input("Enter map size (4x4 or 8x8): ").strip()
    is_slippery = input("Is slippery? (True/False): ").strip().lower() == "true"

    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    n_states = env.observation_space.n

    env = TransformObservation(
        env,
        lambda obs: np.eye(n_states, dtype=np.float32)[obs],
        observation_space=spaces.Box(
            low=0.0,
            high=1.0,
            shape=(n_states,),
            dtype=np.float32
        )
    )

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        seed=42,
        exploration_fraction=0.35,
        batch_size=64,
        exploration_final_eps=0.01,
        learning_rate=1e-4,
    )

    if map_name == "4x4" and not is_slippery: #4x4 and not slippery
        timesteps = 90000
    elif map_name == "4x4" and is_slippery: #4x4 and slippery
        timesteps = 320000
    else:  # 8x8 and slippery
        timesteps = 500000
    

    model.learn(total_timesteps=timesteps)

    mean_reward, _ = evaluate_policy(
        model,
        model.get_env(),
        n_eval_episodes=100,
        deterministic=True
    )

    print("\nMean reward:", mean_reward)
 

    if map_name == "4x4":
        action_names = ["Left", "Down", "Right", "Up"]
        print("\nLearned policy:")
        for s in range(n_states):
            obs = np.eye(n_states, dtype=np.float32)[s]
            action, _ = model.predict(obs, deterministic=True)
            print(f"State {s}: {action_names[int(action)]}")


if __name__ == "__main__":
    main()
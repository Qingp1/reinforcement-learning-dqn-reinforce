import os
import sys
import argparse
from typing import List, Tuple, Optional, Union
import gymnasium as gym
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import plot, xlabel, ylabel, title, grid, show
from numpy import arange
from src.visual import Visual
try:
    from src.reinforce import Reinforce
    from src.reinforce_with_baseline import ReinforceWithBaseline
    from src.deep_q import DeepQModel
    from src.train import train
except:
    print("Please make sure to have the model files in the src directory.")



def parse_arguments() -> argparse.Namespace:
    """
    HELPER - do not edit.
    
    This allows you to run your models easily and quickly adjust the number of parameters to you are using to
    train your models. For more information on how to use them, refer to the handout.

    NOTE: If you are ever confused on how to use the arguments, you can run
    python assignment.py --help
    to see a list of all the arguments and their descriptions.
    """
    parser = argparse.ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('--model', type=str, choices=['REINFORCE', 'REINFORCE_BASELINE', 'DEEP_Q'], required=True, help='Model type to use')
    parser.add_argument('--env', type=str, default=None, help='Environment name')
    parser.add_argument('--watch', action='store_true', help='Watch mode to render the agent after training')
    parser.add_argument('--load-path', type=str, help='Path to load model weights from')
    parser.add_argument('--save-path', type=str, help='Path to save model weights to')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-episodes', type=int, default=650, help='Number of training episodes')
    parser.add_argument('--submit', action='store_true', help='Save training plots to submission folder')
    return parser.parse_args()

def get_env(env_name: str, render_mode: Optional[str] = None):
    """
    Helper to get the environment with appropriate wrappers.
    """
    env = gym.make(env_name, render_mode=render_mode)
    env = gym.wrappers.FlattenObservation(env)
    return env

def visualize_episode(model: tf.keras.Model, env_name: str) -> None:
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    env = get_env(env_name, render_mode="human")
    state = env.reset()[0]
    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        # if sum of probabilities is not 1, take max
        if np.sum(newProb) != 1:
            action = np.argmax(newProb)
        else:
            action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)
        state, _, term, trunc, _ = env.step(action)
        if term or trunc:
            done = True
    

def visualize_data(total_rewards: List[float]) -> None:
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()

def select_environment() -> str:
    """
    HELPER - do not edit.
    If you do not specify an environment, this helper will prompt you with options
    
    :returns: The selected environment name
    """
    environments = [
        'CartPole-v1',
        'LunarLander-v3',
        'MountainCar-v0',
        'Acrobot-v1',
    ]
    
    print("\n" + "="*50)
    print("Available Environments:")
    for i, env in enumerate(environments, 1):
        print(f"{i}. {env}")
    print("="*50)
    
    while True:
        try:
            choice = input("\nSelect an environment (enter number): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(environments):
                selected_env = environments[choice_idx]
                print(f"\nSelected: {selected_env}\n")
                return selected_env
            else:
                print(f"Please enter a number between 1 and {len(environments)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            exit()

def get_weights_path(model_name: str, env_name: str) -> str:
    """
    HELPER - do not edit.
    Returns the standardized path for saving/loading model weights.
    
    :param model_name: Name of the model: 'REINFORCE', 'DEEP_Q', etc.
    :param env_name: Name of the environment: 'CartPole-v1', 'LunarLander-v2', etc.
    :returns: Path to the weights directory
    """
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    return os.path.join(weights_dir, f"{model_name}_{env_name}_best")

def main() -> None:
    """
    HELPER - do not edit.
    The main function for running the assignment. The functionality includes
    training the model, testing the model, and visualizing the model.
    """

    args = parse_arguments()
    
    # If no environment is specified, prompt the user to select one
    if args.env is None:
        args.env = select_environment()
    
    try:
        env = get_env(args.env)
        env_name = args.env
    except Exception as e:
        print(f"Incorrect Environment Name: {args.env}, make sure the environment is exactly as written in the gym documentation")
        exit()

    state_size = env.observation_space.shape
    state_size = state_size[0]
    num_actions = env.action_space.n

    # Initialize your model instance
    if args.model == "REINFORCE":
        model = Reinforce(state_size, num_actions, args.lr)
    elif args.model == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions, args.lr)
    elif args.model == "DEEP_Q":
        model = DeepQModel(state_size, num_actions, args.lr)

    # Use best weights or load path
    if args.load_path:
        try:
            model.load_weights(args.load_path)
            print(f"Loaded weights from {args.load_path}")
        except Exception as e:
            print(f"Could not load weights from {args.load_path}: {e}")
    elif args.watch:
        best_weights_path = get_weights_path(args.model, env_name)
        if os.path.exists(best_weights_path + ".index"):
            try:
                model.load_weights(best_weights_path)
                print(f"Loaded best weights from {best_weights_path}")
            except Exception as e:
                print(f"Could not load weights from {best_weights_path}: {e}")
                return
        else:
            print(f"No saved weights found at {best_weights_path}")
            print(f"Train the model first with: python assignment.py --model {args.model} --env {env_name}")
            return

    if args.watch:
        print("Watch mode enabled. Rendering environment...")
        visualize_episode(model, env_name)
        return

    # NOTE: This is the main training loop! You might want to look at this if you are interested in how the training loop works.
    totalReward = []
    num_episodes = args.num_episodes

    # NOTE: This visualizer is special to this assignment to help you see your agent in progress. This is very helpful in RL settings
    # where model training is easier to visualize than in other settings.
    visual = Visual(num_episodes)
    viz_every = 50
    memory = None
    best_avg_reward = -np.inf
    best_weights_path = get_weights_path(args.model, env_name)

    for episode in range(num_episodes):
        if args.model == "DEEP_Q":
            reward, memory, loss = train(env, model, memory=memory, epsilon=1-episode/num_episodes)
        else:
            reward, loss = train(env, model)
        if episode in range(0, num_episodes):
            totalReward.append(reward)
        
        visual.update(episode, loss, reward)
        
        # NOTE: This is checkpointing for those of you interested. Here, we are using the average reward as our metric and whenever 
        # we find a new model instance with the best performance, we save it. This is very helpful in setting where agents may end
        # setting on sub-optimal training strategies as training progresses. 
        current_best = visual.get_best_avg_reward()
        if current_best > best_avg_reward + 5.0:
            best_avg_reward = current_best
            model.save_weights(best_weights_path, save_format='tf')
            print("=" * 50)
            print(f"\n New Best Model! Avg reward: {best_avg_reward:.2f}")
            print("=" * 50)

        if episode%viz_every == 0:
            visualize_episode(model, env_name)
    visualize_episode(model, env_name)
    env.close()
    
    if args.save_path:
        model.save_weights(args.save_path)
        print(f"\nSaved final weights to {args.save_path}")
    
    print(sum(totalReward)/len(totalReward))
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Best model saved at: {best_weights_path}")
    visual.plot_results(save_for_submission=args.submit, model_name=args.model)
if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import time
import sys

class Visual:
    def __init__(self, total_episodes):
        self.losses = []
        self.rewards = []
        self.episodes = []
        self.total_episodes = total_episodes
        self.start_time = time.time()

    def update(self, episode, loss, reward):
        self.episodes.append(episode)
        self.losses.append(loss)
        self.rewards.append(reward)
        self.draw(episode, loss, reward)

    def draw(self, episode, loss, reward):
        """
        Draws the progress bar and statistics for the current episode.

        :param episode: Current episode number
        :param loss: Loss value for the current episode
        :param reward: Reward obtained in the current episode
        """
        avg_reward = np.mean(self.rewards[-50:]) if len(self.rewards) >= 50 else np.mean(self.rewards)
        
        progress = (episode + 1) / self.total_episodes
        bar_length = 20
        block = int(round(bar_length * progress))
        # ASCII MEGA STRING :) 
        text = "Episode: {}/{} | [{}] {:.1f}% | Loss: {:.4f} | Reward: {:.2f} | Avg Reward (last 50): {:.2f}".format(
            episode + 1, 
            self.total_episodes, 
            "=" * block + " " * (bar_length - block), 
            progress * 100, 
            loss, 
            reward, 
            avg_reward
        )
        sys.stdout.write(f"\r\033[K{text}")
        sys.stdout.flush()

    def get_best_avg_reward(self):
        """
        Returns the best moving average reward of thelast 50 episodes
        
        :returns: The best average reward or 0 if no rewards have been recorded
        """
        if len(self.rewards) == 0:
            return 0
        
        if len(self.rewards) < 50:
            return np.mean(self.rewards)
        
        # Calculate moving average for all windows of size 50
        best_avg = -np.inf
        for i in range(len(self.rewards) - 49):
            avg = np.mean(self.rewards[i:i+50])
            if avg > best_avg:
                best_avg = avg
        
        return best_avg

    def plot_results(self, save_for_submission=False, model_name="model"):
        """
        Plots the rewards and losses over episodes.
        
        :param save_for_submission: If True, saves the plot to a local output folder
        :param model_name: Name of the model (used for filename)
        """
        print("\nPlotting results...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(self.episodes, self.rewards, label='Reward', color='blue', alpha=0.6)
        window_size = 50
        if len(self.rewards) >= window_size:
            moving_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(self.episodes[window_size-1:], moving_avg, label=f'Avg Reward (ws:{window_size})', color='red', linewidth=2)
        
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.episodes, self.losses, label='Loss', color='orange', alpha=0.6)
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Episode')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_for_submission:
            import os
            submission_dir = "submission"
            os.makedirs(submission_dir, exist_ok=True)
            filename = f"{model_name}_train_plot.png"
            filepath = os.path.join(submission_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved training plot to {filepath}")
        
        plt.show()

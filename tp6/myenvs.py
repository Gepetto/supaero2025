import gymnasium as gym
import numpy as np


class EnvMountainCarFullyDiscrete(gym.Wrapper):
    def __init__(self, n_bins=(51, 51), **kwargs):
        env = gym.make("MountainCar-v0", **kwargs).unwrapped
        super().__init__(env)
        self.n_bins = n_bins

        # Définition des bornes des états
        self.state_bounds = [
            [-1.2, 0.6],  # Position du chariot
            [-0.1, 0.1],  # Vitesse du chariot
        ]

        # Création des bins pour discrétiser l'état
        self.bins = [
            np.linspace(low - 1e-6, high + 1e-6, num=n, endpoint=True)
            for (low, high), n in zip(self.state_bounds, self.n_bins)
        ]
        # Basis value for representing the bin indexes as a single integer
        self.bin_base = [int(np.prod(self.n_bins[:n])) for n in range(len(self.n_bins))]

        # Définition de l'espace d'observation discret
        # self.observation_space = gym.spaces.MultiDiscrete(self.n_bins)
        self.observation_space = gym.spaces.Discrete(np.prod(self.n_bins))

    def discrete_state_to_index(self, bins):
        return sum([idx * base for (idx, base) in zip(bins, self.bin_base)])

    def index_to_discrete_state(self, index):
        bins = []
        for n in self.n_bins:
            bins.append(index % n)
            index = index // n
        return bins

    def discretize_state(self, state):
        """Convertit un état continu en un index discret."""
        return [np.digitize(s, bins) for s, bins in zip(state, self.bins)]

    def undiscretize_state(self, discrete_state):
        """Transforme un état discret en état continu en prenant le centre du bin."""
        assert self.is_discrete_state_in_range(discrete_state)
        return np.array(
            [
                (
                    (self.bins[i][d - 1] + self.bins[i][d]) / 2
                    if 0 < d < len(self.bins[i])
                    else self.state_bounds[i][d == 0]
                )
                for i, d in enumerate(discrete_state)
            ]
        )

    def is_discrete_state_in_range(self, discrete_state):
        """Check that non of the bins is out of the state space"""
        res = 0 not in discrete_state
        res = res and np.all([d < n for d, n in zip(discrete_state, self.n_bins)])
        return res

    def is_index_in_range(self, index):
        if index < 0 or index > self.observation_space.n:
            return False
        return self.is_discrete_state_in_range(self.index_to_discrete_state(index))

    def step(self, action):
        """Exécute une action et retourne l'état sous forme discrète."""
        # Step 10 times in the continuous space to ensure enough movements
        # to pass a discrete bin.
        for i in range(10):
            next_state_cont, reward, done, trunc, info = self.env.step(action)
            if done:
                break
        next_state_discrete = self.discretize_state(next_state_cont)
        if not self.is_discrete_state_in_range(next_state_discrete):
            done = False
            trunc = True
            self.state = None
        else:
            # Reconvertir en état continu pour rester dans un espace propre
            self.env.state = self.undiscretize_state(next_state_discrete)
            self.discrete_state = next_state_discrete
            self.state = self.discrete_state_to_index(next_state_discrete)
        return self.state, reward, done, trunc, info

    def reset(self, **kwargs):
        """Réinitialise l'environnement avec un état strictement discret."""
        continuous_state, info = self.env.reset(**kwargs)
        self.discrete_state = self.discretize_state(continuous_state)
        assert self.is_discrete_state_in_range(self.discrete_state)
        self.state = self.discrete_state_to_index(self.discrete_state)
        # On force le reset sur un état discretisé
        self.env.state = self.undiscretize_state(self.discrete_state)
        return self.state, info


if __name__ == "__main__":
    env = EnvMountainCarFullyDiscrete()
    env.reset()
    env.step(0)

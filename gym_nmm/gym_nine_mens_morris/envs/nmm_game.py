# ... (der gesamte vorherige Code)
import numpy as np
from gym_nmm.gym_nine_mens_morris.envs.nmm_v2 import NineMensMorrisEnvV2

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def choose_action(self):
        legal_actions = self.env.get_legal_actions()
        flattened_actions = NineMensMorrisEnvV2.flatten_actions(legal_actions)
        random_index = np.random.randint(0, len(flattened_actions))
        return flattened_actions[random_index]


def play_human_vs_agent(env, agent):
    env.reset()
    while not env.done:
        env.render()
        if env.turn == 1:  # Menschlicher Spieler
            action = get_human_input(env)
        else:  # Agent
            action = agent.choose_action()
        _, _, done, _ = env.step(action)
    env.render()
    if env.winner == 1:
        print("Gratulation! Du hast gewonnen!")
    elif env.winner == -1:
        print("Der Agent hat gewonnen!")
    else:
        print("Unentschieden!")

def get_human_input(env):
    legal_actions = env.get_legal_actions()
    flattened_actions = NineMensMorrisEnvV2.flatten_actions(legal_actions)
    print("Mögliche Züge:", flattened_actions)

    while True:
        try:
            position = int(input("Wähle eine Position (0-23): "))
            if env.is_phase_1():
                move = None
            else:
                move = int(input("Wähle eine Bewegungsrichtung (0-3): "))
            kill = input("Wähle eine zu entfernende Figur des Gegners oder drücke Enter: ")
            kill = int(kill) if kill else None

            action = (position, move, kill)
            if action in flattened_actions:
                return action
            else:
                print("Ungültiger Zug. Versuche es erneut.")
        except ValueError:
            print("Ungültige Eingabe. Versuche es erneut.")


if __name__ == "__main__":
    env = NineMensMorrisEnvV2()
    agent = RandomAgent(env)
    play_human_vs_agent(env, agent)

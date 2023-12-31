import numpy as np
import torch
from gym_nmm.nine_mens_morris.nmm import env
from gym_nmm.nine_mens_morris.nmm_alphazero import AlphaZeroNMM, MCTS


class AlphaZeroAgent:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.mcts = MCTS(self.model, self.device)

    def act(self, observation, environment):  # Pass the actual environment object
        state = observation['observation']
        action_type = observation['action_type']

        [self.mcts.simulate(environment) for _ in range(500)]
        p = self.mcts.get_action_prob(environment, action_type, temperature=0.5)  # Added temperature parameter
        valid_actions = [i for i, valid in enumerate(observation['action_mask']) if valid]
        action_probs = np.array([p[i] for i in valid_actions])
        action = np.argmax(action_probs)

        return valid_actions[action]
def play_game():
    environment = env(render_mode='human')
    human_agent = "player_0"

    device = torch.device("cpu")
    model_path = '/Users/Batu-Privat/PycharmProjects/gym_nine_mens_morris2/gym_nmm/nine_mens_morris/model_2.pth'
    model = AlphaZeroNMM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    alpha_zero_agent = AlphaZeroAgent(model, device)
    environment.reset()
    done = False

    while not done:
        agent_to_act = environment.agent_selection
        observation = environment.observe(agent_to_act)

        if agent_to_act == human_agent:
            print("Your turn!")
            render_output = environment.render()
            if render_output:
                print(render_output)

            valid_actions = [i for i, valid in enumerate(observation['action_mask']) if valid]
            print("Valid actions:", valid_actions)
            action = int(input("Choose your action: "))
            while action not in valid_actions:
                print("Invalid action. Try again.")
                action = int(input("Choose your action: "))
        else:
            action = alpha_zero_agent.act(observation, environment)  # Pass the environment to act
            print("AI's turn!")
            render_output = environment.render()
            if render_output:
                print(render_output)

        environment.step(action)
        done = environment.terminations[agent_to_act]

    print("Game Over!")
    render_output = environment.render()
    if render_output:
        print(render_output)


if __name__ == "__main__":
    play_game()
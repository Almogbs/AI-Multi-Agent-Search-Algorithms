import os
import argparse


studpid_agents = [  "random",
                    "greedy"]

agents = [  "random",
            "greedy",
            "improvedgreedy",
            "minimax",
            #"alphabeta",
            #"expectimax"
            ]

def run_tests():
    """
    parser = argparse.ArgumentParser(description='Test your submission by pitting agents against each other.')
    parser.add_argument('agent0', type=str,
                        help='First agent')
    parser.add_argument('agent1', type=str,
                        help='Second agent')
    parser.add_argument('-t', '--time_limit', type=float, nargs='?', help='Time limit for each turn in seconds', default=1)
    parser.add_argument('-c', '--count_steps', nargs='?', type=int, help='Number of steps each taxi gets before game is over',
                        default=4761)
    parser.add_argument('--print_game', action='store_true')

    args = parser.parse_args()
    """ 
    time_limit = 0.5
    seed = 1234
    steps = 1000
    for agent in agents:
        for rival_agent in agents:
            if agent != rival_agent and agent not in studpid_agents:
                print(f"\n\n********* {agent} VS. {rival_agent} *********")
                print(f"seed={seed}, time_limit={time_limit}, step_limit={steps}")

                os.system(f"python main.py {agent} {rival_agent}  -t {time_limit} -s {seed} -c {steps}")
                print(f"********* END OF TEST *********")


if __name__ == "__main__":
    run_tests()
import os
import argparse
from time import time
import re
import subprocess

from soupsieve import match


agents = [  "random",
            "greedy",
            "improvedgreedy",
            "minimax",
            "alphabeta",
            "expectimax"
            ]

def check_results(agent, rival, agent_score, rival_score, winner):
    winner = int(winner)

    ## CASE0:  improvedgreedy VS random
    if agent == "improvedgreedy" and rival == "random":
        return winner == 0

    #CASE1:  random VS improvedgreedy
    if agent == "random" and rival == "improvedgreedy":
        return winner == 1
    
    #CASE2:  improvedgreedy VS greedy
    if agent == "improvedgreedy" and rival == "greedy":
        return winner == 0
        
    #CASE3:  greedy VS improvedgreedy
    if agent == "greedy" and rival == "improvedgreedy":
        return winner == 1

    #CASE4:  improvedgreedy VS minimax
    if agent == "improvedgreedy" and rival == "minimax":
        return winner == 1
        
    #CASE5:  mininax VS improvedgreedy
    if agent == "minimax" and rival == "improvedgreedy":
        return winner == 0

    #CASE6:  improvedgreedy VS alphabeta
    if agent == "improvedgreedy" and rival == "alphabeta":
        return winner == 1
        
    #CASE7:  alphabeta VS improvedgreedy
    if agent == "alphabeta" and rival == "improvedgreedy":
        return winner == 0

    #CASE8:  minimax VS greedy
    if agent == "minimax" and rival == "greedy":
        return winner == 0
        
    #CASE9:  greedy VS minimax
    if agent == "greedy" and rival == "minimax":
        return winner == 1

    #CASE10:  minimax VS alphabeta
    if agent == "minimax" and rival == "alphabeta":
        return True
        
    #CASE11:  alphabeta VS minimax
    if agent == "alphabeta" and rival == "minimax":
        return True

    #CASE12:  minimax VS random
    if agent == "minimax" and rival == "random":
        return winner == 0
        
    #CASE13:  random VS minimax
    if agent == "random" and rival == "minimax":
        return winner == 1

    #CASE14:  alphabeta VS greedy
    if agent == "alphabeta" and rival == "greedy":
        return winner == 0
        
    #CASE15:  greedy VS alphabeta
    if agent == "greedy" and rival == "alphabeta":
        return winner == 1

    #CASE16:  alphabeta VS random
    if agent == "alphabeta" and rival == "random":
        return winner == 0
        
    #CASE17:  random VS alphabeta
    if agent == "random" and rival == "alphabeta":
        return winner == 1

    #CASE18:  random VS greedy
    if agent == "random" and rival == "greedy":
        return True

    #CASE19:  greedy VS random
    if agent == "greedy" and rival == "random":
        return True

    #CASE20:  expectimax VS random
    if agent == "expectimax" and rival == "random":
        return winner == 0

    #CASE21:  expectimax VS greedy
    if agent == "expectimax" and rival == "greedy":
        return winner == 0

    #CASE22:  expectimax VS improvedgreedy
    if agent == "expectimax" and rival == "improvedgreedy":
        return winner == 0

    #CASE23:  expectimax VS minimax
    if agent == "expectimax" and rival == "minimax":
        return winner == 1

    #CASE24:  expectimax VS alphabeta
    if agent == "expectimax" and rival == "alphabeta":
        return winner == 0

    #CASE25:  alphabeta VS expectimax
    if agent == "alphabeta" and rival == "expectimax":
        return winner == 0

    #CASE26:  minimax VS expectimax
    if agent == "minimax" and rival == "expectimax":
        return winner == 0

    #CASE27:  improvedgreedy VS expectimax
    if agent == "improvedgreedy" and rival == "expectimax":
        return winner == 0

    #CASE28:  random VS expectimax
    if agent == "random" and rival == "expectimax":
        return winner == 1

    #CASE29:  greedy VS expectimax
    if agent == "greedy" and rival == "expectimax":
        return winner == 1

    return False


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

    total = 0
    failed = 0

    for agent in agents:
        for rival_agent in agents:
            total += 1
            start = time()
            if agent != rival_agent:
                print(f"\n\n\n********* {agent} VS. {rival_agent} *********")
                print(f"seed={seed}, time_limit={time_limit}, step_limit={steps}")

                res = subprocess.check_output(f"python main.py {agent} {rival_agent}  -t {time_limit} -s {seed} -c {steps}", shell=True)
                res = res.decode()
                match = re.findall('[0-9]+', res)
                
                # Draw
                if len(match) == 2:
                    match.append(2)
                
                print(res)

                if check_results(agent, rival_agent, *match):
                    print("PASS")
                else:
                    failed += 1
                    print("FAILED")

                print(f"********* END OF TEST (time elapsed:{time()-start}) *********")

    print(f"\nFailed: {failed} out of total: {total}")
            


if __name__ == "__main__":
    run_tests()
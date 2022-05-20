from time import time
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import sys


INF = sys.maxsize
MINUS_INF = -sys.maxsize - 1
TIME_BUFF = 0.01



class AgentGreedyImproved(AgentGreedy):
    """
        Improved (heuristic) Greedy Agent:

    """
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators, children = self.successors(env, taxi_id)

        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)            
        index_selected = children_heuristics.index(max_heuristic)

        return operators[index_selected]


    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)

        if taxi.passenger:

            return 5*taxi.cash - manhattan_distance(taxi.position, taxi.passenger.destination)

        else:
            taxi_pos = taxi.position
            pass_pos = env.passengers[0].position
            pass_dest = env.passengers[0].destination
            dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            for passenger in env.passengers:
                if manhattan_distance(taxi_pos, passenger.position) < dist_pass_taxi:
                    pass_pos = passenger.position
                    pass_dest = passenger.destination
                    dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            return 5*taxi.cash - dist_pass_taxi - manhattan_distance(pass_dest, pass_pos)



class AgentMinimax(Agent):
    """
        Minimax Agent:
        
    """
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        end = time()
        best_res = MINUS_INF
        best_res_op = None
        depth = 0

        while time_limit > TIME_BUFF:
            start = time()
            curr_res, curr_res_op = self.rb_minimax(self, env, agent_id, depth, True, time_limit - TIME_BUFF)
            if best_res < curr_res:
                best_res = curr_res
                best_res_op = curr_res_op

            depth += 1
            end = time()
            time_limit -= (end-start)
            
        return best_res_op
    

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)

        if taxi.passenger:

            return 1 + 5*(taxi.cash-other_taxi.cash) - manhattan_distance(taxi.position, taxi.passenger.destination)

        else:
            taxi_pos = taxi.position
            pass_pos = env.passengers[0].position
            pass_dest = env.passengers[0].destination
            dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            for passenger in env.passengers:
                if manhattan_distance(taxi_pos, passenger.position) < dist_pass_taxi:
                    pass_pos = passenger.position
                    pass_dest = passenger.destination
                    dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            return 5*(taxi.cash-other_taxi.cash) - dist_pass_taxi - manhattan_distance(pass_dest, pass_pos)


    @staticmethod
    def rb_minimax(self, env: TaxiEnv, agent_id: int, depth: int, our_turn: bool, time_limit: float):
        start = time()
        operators, children = self.successors(env, self.get_actual_id(agent_id, our_turn))

        if depth <= 0 or env.done() or time_limit <= TIME_BUFF:
            
            return self.heuristic(env, agent_id), operators[0]

        if our_turn:
            curr_max = MINUS_INF
            curr_max_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_minimax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start))

                if curr_max < curr:
                    curr_max = curr
                    curr_max_child = i
        
            return (curr_max, operators[curr_max_child])
        
        # not our turn
        else:
            curr_min = INF
            curr_min_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_minimax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start))

                if curr_min > curr:
                    curr_min = curr
                    curr_min_child = i
        
            return (curr_min, operators[curr_min_child])


    @staticmethod
    def get_actual_id(agent_id: int, our_turn: bool):
        if our_turn:

            return agent_id

        else:

            return (agent_id + 1) % 2



class AgentAlphaBeta(Agent):
    """
        Alpha-Beta Pruning Minimax Agent:
        
    """
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        end = time()
        best_res = MINUS_INF
        best_res_op = None
        depth = 0

        while time_limit > TIME_BUFF:
            start = time()
            curr_res, curr_res_op = self.rb_alphabeta(self, env, agent_id, depth, True, time_limit - TIME_BUFF, MINUS_INF, INF)
            if best_res < curr_res:
                best_res = curr_res
                best_res_op = curr_res_op

            depth += 1
            end = time()
            time_limit -= (end-start)
            
        return best_res_op
    

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        other_taxi = env.get_taxi((taxi_id + 1) % 2)

        if taxi.passenger:

            return 1 + 5*(taxi.cash-other_taxi.cash) - manhattan_distance(taxi.position, taxi.passenger.destination)

        else:
            taxi_pos = taxi.position
            pass_pos = env.passengers[0].position
            pass_dest = env.passengers[0].destination
            dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            for passenger in env.passengers:
                if manhattan_distance(taxi_pos, passenger.position) < dist_pass_taxi:
                    pass_pos = passenger.position
                    pass_dest = passenger.destination
                    dist_pass_taxi = manhattan_distance(taxi_pos, pass_pos)

            return 5*(taxi.cash-other_taxi.cash) - dist_pass_taxi - manhattan_distance(pass_dest, pass_pos)


    @staticmethod
    def rb_alphabeta(self, env: TaxiEnv, agent_id: int, depth: int, our_turn: bool, time_limit: float, alpha: int, beta: int):
        start = time()
        operators, children = self.successors(env, self.get_actual_id(agent_id, our_turn))

        if depth <= 0 or env.done() or time_limit <= TIME_BUFF:
            
            return self.heuristic(env, agent_id), operators[0]

        if our_turn:
            curr_max = MINUS_INF
            curr_max_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_alphabeta(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), alpha, beta)

                if curr_max < curr:
                    curr_max = curr
                    curr_max_child = i
                
                alpha = max(alpha, curr_max)
                if curr_max >= beta:
                    return INF, None
        
            return (curr_max, operators[curr_max_child])
        
        # not our turn
        else:
            curr_min = INF
            curr_min_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_alphabeta(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), alpha, beta)

                if curr_min > curr:
                    curr_min = curr
                    curr_min_child = i
        
                beta = min(beta, curr_min)
                if curr_min <= alpha:
                    return MINUS_INF, None

            return (curr_min, operators[curr_min_child])


    @staticmethod
    def get_actual_id(agent_id: int, our_turn: bool):
        if our_turn:

            return agent_id

        else:

            return (agent_id + 1) % 2


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()

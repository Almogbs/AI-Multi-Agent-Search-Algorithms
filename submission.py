from time import time
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import sys


INF = sys.maxsize
MINUS_INF = -sys.maxsize - 1
TIME_BUFF = 0.01



def shared_heuristic(env: TaxiEnv, taxi_id: int):
    taxi = env.get_taxi(taxi_id)
    other_taxi = env.get_taxi((taxi_id + 1) % 2)
    have_enough_fule_for_pass = [False, False]
    max_profit = calculate_max_profit(env, taxi)
    res = max_profit + taxi.cash

    for i, curr_pass in enumerate(env.passengers):
        have_enough_fule_for_pass[i] = have_enough_fule(taxi, curr_pass)
    if taxi.passenger and have_enough_fule(taxi, taxi.passenger):
        profit = manhattan_distance(taxi.passenger.position, taxi.passenger.destination)
        res += profit*(taxi.fuel - manhattan_distance(taxi.position, taxi.passenger.destination))

    if True not in have_enough_fule_for_pass and min(env.num_steps // 2, taxi.cash) >= max_profit > other_taxi.cash:
        res += taxi.fuel - manhattan_distance(taxi.position, closest_gas_position(taxi, env))
    return res

def have_enough_fule(taxi, passenger):
    if passenger == taxi.passenger:
        return manhattan_distance(passenger.position, passenger.destination) <= taxi.fuel
    return manhattan_distance(taxi.position, passenger.position) +\
            manhattan_distance(passenger.position, passenger.destination) <= taxi.fuel

def calculate_profit(env: TaxiEnv, taxi, passenger):
    distance_to_travel = manhattan_distance(passenger.position, passenger.destination)\
                            + manhattan_distance(taxi.position, passenger.position)

    if distance_to_travel + 2 <= env.num_steps // 2 + 1:
        return manhattan_distance(passenger.position, passenger.destination)\
                - manhattan_distance(taxi.position, passenger.position)
    else:
        return 0

def calculate_max_profit(env: TaxiEnv, taxi):
    return max([calculate_profit(env, taxi, passenger) for passenger in env.passengers] + [0])

def closest_gas_position(taxi, env: TaxiEnv):
    if manhattan_distance(taxi.position, env.gas_stations[0].position)\
            < manhattan_distance(taxi.position, env.gas_stations[1].position):
        return env.gas_stations[0].position
    return env.gas_stations[1].position



def sshared_heuristic(env: TaxiEnv, taxi_id: int):
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


def get_actual_id(agent_id: int, our_turn: bool):
    if our_turn:
        return agent_id
    else:
        return (agent_id + 1) % 2




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
        return shared_heuristic(env, taxi_id)




class AgentMinimax(Agent):
    """
        Minimax Agent:
        
    """
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        end = time()
        best_res_op = None
        depth = 0

        while time_limit > TIME_BUFF:
            start = time()
            _, curr_res_op = self.rb_minimax(self, env, agent_id, depth, True, time_limit - TIME_BUFF, "park")

            if curr_res_op is not None:
                best_res_op = curr_res_op

            depth += 1
            end = time()
            time_limit -= (end-start)
        return best_res_op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return shared_heuristic(env, taxi_id)

    @staticmethod
    def rb_minimax(self, env: TaxiEnv, agent_id: int, depth: int, our_turn: bool, time_limit: float, prev_op):
        start = time()
        operators, children = self.successors(env, get_actual_id(agent_id, our_turn))

        if depth <= 0 or env.done() or (time_limit <= TIME_BUFF and depth <= 1):
            return self.heuristic(env, agent_id), prev_op

        if time_limit <= TIME_BUFF: 
            return self.heuristic(env, agent_id), None


        if our_turn:
            curr_max = MINUS_INF
            curr_max_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_minimax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), operators[i])

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
                curr, _ = self.rb_minimax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), operators[i])

                if curr_min > curr:
                    curr_min = curr
                    curr_min_child = i
            return (curr_min, operators[curr_min_child])






class AgentAlphaBeta(Agent):
    """
        Alpha-Beta Pruning Minimax Agent:
        
    """
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        end = time()
        best_res_op = None
        depth = 0

        while time_limit > TIME_BUFF:
            start = time()
            _, curr_res_op = self.rb_alphabeta(self, env, agent_id, depth, True, time_limit - TIME_BUFF, MINUS_INF, INF, "park")

            if curr_res_op is not None:
                best_res_op = curr_res_op

            depth += 1
            end = time()
            time_limit -= (end-start)
            
        return best_res_op
    
    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return shared_heuristic(env, taxi_id)



    @staticmethod
    def rb_alphabeta(self, env: TaxiEnv, agent_id: int, depth: int, our_turn: bool, time_limit: float, alpha: int, beta: int, prev_op):
        start = time()
        operators, children = self.successors(env, get_actual_id(agent_id, our_turn))

        if depth <= 0 or env.done() or (time_limit <= TIME_BUFF and depth <= 1):
            return self.heuristic(env, agent_id), prev_op
        if time_limit <= TIME_BUFF: 
            return self.heuristic(env, agent_id), None

        if our_turn:
            curr_max = MINUS_INF
            curr_max_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_alphabeta(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), alpha, beta, operators[i])

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
                curr, _ = self.rb_alphabeta(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), alpha, beta, operators[i])

                if curr_min > curr:
                    curr_min = curr
                    curr_min_child = i
        
                beta = min(beta, curr_min)
                if curr_min <= alpha:
                    return MINUS_INF, None
            return (curr_min, operators[curr_min_child])




class AgentExpectimax(Agent):
    """
        Expectimax Agent:
        
    """
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        end = time()
        best_res_op = None
        depth = 0

        while time_limit > TIME_BUFF:
            start = time()
            _, curr_res_op = self.rb_expectedmax(self, env, agent_id, depth, True, time_limit - TIME_BUFF, "park")
            
            if curr_res_op is not None:
                best_res_op = curr_res_op

            depth += 1
            end = time()
            time_limit -= (end-start)
        return best_res_op

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        return shared_heuristic(env, taxi_id)

    @staticmethod
    def rb_expectedmax(self, env: TaxiEnv, agent_id: int, depth: int, our_turn: bool, time_limit: float, prev_op):
        start = time()
        operators, children = self.successors(env, get_actual_id(agent_id, our_turn))
        
        if depth <= 0 or env.done() or (time_limit <= TIME_BUFF and depth <= 1):
            return self.heuristic(env, agent_id), prev_op

        if time_limit <= TIME_BUFF: 
            return self.heuristic(env, agent_id), None

        if our_turn:
            curr_max = MINUS_INF
            curr_max_child = 0

            for i, child in enumerate(children):
                end = time()
                curr, _ = self.rb_expectedmax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), operators[i])

                if curr_max < curr:
                    curr_max = curr
                    curr_max_child = i
            return (curr_max, operators[curr_max_child])
        
        # not our turn
        else:
            vals = []
            for i, child in enumerate(children):
                end = time()
                val, _ = self.rb_expectedmax(self, child, agent_id, depth - 1, not our_turn, time_limit - (end-start), operators[i])
                vals.append(self.probabilistic(operators[i]) * val)
            return sum(vals), prev_op

    @staticmethod
    def probabilistic(op):
        if op in ["move north", "move south", "move east", "move west"]:
            return 1/12
        else:
            return 2/12
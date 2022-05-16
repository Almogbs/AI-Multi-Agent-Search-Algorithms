from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


class AgentGreedyImproved(AgentGreedy):
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        if "drop off passenger" in operators:
            return operators[operators.index("drop off passenger")]

        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)            
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if taxi.passenger:
            return taxi.cash - manhattan_distance(taxi.position, taxi.passenger.destination)
        else:
            taxi_pos = taxi.position
            pass0_pos = env.passengers[0].position
            pass1_pos = env.passengers[1].position
            pass0_dest = env.passengers[0].destination
            pass1_dest = env.passengers[1].destination
            dist_pass0_taxi = manhattan_distance(taxi_pos, pass0_pos)
            dist_pass1_taxi = manhattan_distance(taxi_pos, pass1_pos)
            if dist_pass0_taxi < dist_pass1_taxi:
                return taxi.cash - manhattan_distance(taxi_pos, pass0_pos) - manhattan_distance(pass0_dest, pass0_pos)
            else:
                return taxi.cash - manhattan_distance(taxi_pos, pass1_pos) - manhattan_distance(pass1_dest, pass1_pos)



class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()

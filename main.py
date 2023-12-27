from math import sqrt
import random as rand
import numpy as np
import torch
import time


from params import *
from utils import *
from simulation import Simulation


from neural_network import CellNeuralNetwork 
def run_test():
    network = CellNeuralNetwork().cuda()       
    for _ in range(20):
        #Random input on 1 and 0, where probability of 1 is 0.1
        input = torch.rand(1, 6, 400, 400).cuda()
        input = torch.where(input > 0.9, torch.ones_like(input), torch.zeros_like(input))
        input = input.reshape(1, cell_memory*2, view_size, view_size)
        decisions = network(input, 0.2, 0.6)
        #Chose max index in decisions 
        decision = torch.argmax(decisions).item()
        print(decisions)
        print(decision)



def main():
    #run_test()
    #exit()
    start_time = time.time()
    simulation = Simulation(n_cells=n_cells, cell_size=start_cell_size)
    simulation.run_simulation(sim_steps)
    print(measures)
    print(f"Total time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
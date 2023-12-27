from math import sqrt
import os
import random as rand
import numpy as np
import torch
import time
import signal
import argparse

from params import *
from utils import *
import sys
from simulation import Simulation

simulation = None
simulation_name = ""
save_state = False
visualize = False


def signal_handler(sig, frame):
    if (save_state):
        print("Saving state...")
        simulation.save_state(simulation_name)
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main():
    #Add command line arguments:
    # -n : name of the simulation - string
    # -s : save state of the simulation - bool
    # -v : save information for rendering - bool
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name of the simulation", default="default")
    parser.add_argument("-s", "--save", help="Save state of the simulation", action="store_true")
    parser.add_argument("-v", "--visualize", help="Save information for rendering", action="store_true")
    args = parser.parse_args()
    
    global simulation_name, save_state, visualize, simulation
    simulation_name = args.name
    save_state = args.save
    visualize = args.visualize
    
    start_time = time.time()
    if (save_state==False or not os.path.exists(f"saved_state/{simulation_name}.txt")):
        simulation = Simulation(n_cells=n_cells, cell_size=start_cell_size)
    else:
        print("Loading state...")
        simulation = Simulation.load_from(f"saved_state/{simulation_name}.txt", n_cells=n_cells, cell_size=start_cell_size)
    simulation.run_simulation(sim_steps, visualize=visualize, simulation_name=simulation_name)
    print(measures)
    print(f"Total time: {time.time() - start_time}")

if __name__ == "__main__":
    main()
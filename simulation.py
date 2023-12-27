from math import sqrt
import random as rand
import numpy as np
import torch
from neural_network import CellNeuralNetwork

from params import *
from utils import *
from cell import Cell
from graphics import Graphics

FORCES_BY_DECISION = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4] 

class Simulation:
    def __init__(self, n_cells, cell_size):
        self.cells = [Cell(cell_size) for _ in range(n_cells)]

    def load_from(filename, n_cells, cell_size):
        with open(filename) as f:
            lines = f.readlines()
            cells = []
            for line in lines:
                values = line.split(",")
                position = np.array([float(values[0]), float(values[1])])
                size = float(values[2])
                speed = np.array([float(values[3]), float(values[4])])
                lifespan = float(values[5])
                neural_network = None
                if (len(values) > 6 and len(values[6].strip()) > 0):
                    neural_network = CellNeuralNetwork().cuda()
                    neural_network.load_state_dict(torch.load(values[6].strip()))
                cells.append(Cell.get_from(position, size, speed, lifespan, neural_network))
        s = Simulation(n_cells, cell_size)
        s.cells = cells
        return s

    def run_simulation(self, n_steps):
        #Initialize graphics
        graphics = Graphics()
        graphics.initialize(chess_size, chess_size)
        #Run simulation
        i = 0
        f = self.cells[0]
        n_created = 0
        while(i < n_steps or n_steps == -1):
            n_created += self.step(i)
            graphics.update(self.cells)
            if (i%3000==0):
                print(f"Step {i} - {len(self.cells)} cells, of which {len([c for c in self.cells if c.lifespan>0])} alive - {n_created} created")
                n_created = 0
            i += 1
        #Close graphics
        graphics.close()
    
    def step(self, iteration):
        #1: Update position of cells, using speed, and update speed using friction
        TimeMeasure.start()
        for cell in self.cells:
            cell.update_position()
        TimeMeasure.stop("Update_position")
        #2: Check for collisions and cells eating others 
        TimeMeasure.start()
        self.check_collisions()
        TimeMeasure.stop("Check_collisions")
        #3: Action decision for each cell
        if (iteration % update_cells_memory_every == 0):
            TimeMeasure.start()
            visual_map = self.get_visual_map()
            TimeMeasure.stop("Get_visual_map")
        if (iteration % run_network_every == 0):
            for index, cell in enumerate(self.cells):
                cell.update_visual(visual_map)
                TimeMeasure.start()
                decision = cell.make_decision()
                if (decision == 0):
                    continue
                elif (decision == 9):
                    if (cell.size >= 6):
                        rand_dir = np.random.normal(size=2)
                        new_cell = Cell.get_from(cell.position - (rand_dir * cell.size * 0.6), cell.size/2, (cell.speed-rand_dir)/2, cell_lifespan, cell.neural_network.get_copy())
                        self.cells.append(new_cell)
                        cell.size = cell.size / 2
                        cell.position += (rand_dir * cell.size * 0.6)
                        cell.speed = (cell.speed+rand_dir)/2
                        cell.lifespan = cell_lifespan
                else:
                    c = cell.apply_force(FORCES_BY_DECISION[decision-1])
                    if (c is not None):
                        self.cells.append(c)
                TimeMeasure.stop("Other")
        #4: Split large dead cells
        if (iteration % 30 == 0):
            threshold = 15 if (iteration%100 == 0) else 50
            for cell in self.cells:
                if (cell.lifespan <= 0):
                    while(cell.size > threshold):
                        #Random direction as np array
                        rand_dir = np.random.normal(size=2)
                        new_cell = Cell.get_from(cell.position - (rand_dir*cell.size * 0.7), cell.size/2, cell.speed, 0)
                        self.cells.append(new_cell)
                        cell.size = cell.size / 2
                        cell.position += (rand_dir*cell.size * 0.7)
        #5: Create new cells if not enough live ones
        created_cells = 0
        if (iteration % 50 == 0):
            live_cells = len([cell for cell in self.cells if cell.lifespan > 0])
            if (live_cells < min_n_cells):
                dead_cells = [cell for cell in self.cells if cell.lifespan <= 0]
                for _ in range(n_cells - min_n_cells):
                    #Chose a random dead_cell
                    if (len(dead_cells)==0):
                        break
                    created_cells += 1
                    dead_cell = rand.choice(dead_cells)
                    #Remove it from the list
                    dead_cells.remove(dead_cell)
                    #Revitalize
                    dead_cell.revitalize()
        #6 check total mass
        if (iteration % 200 == 0):
            total_mass = sum([3.14*cell.size**2 for cell in self.cells])
            if (total_mass > max_mass):
                dead_cells = [cell for cell in self.cells if cell.lifespan <= 0]
                while(total_mass > max_mass):
                    #Chose a random dead_cell
                    if (len(dead_cells)==0):
                        break
                    dead_cell = rand.choice(dead_cells)
                    #Remove it from the list
                    total_mass -= (dead_cell.size**2 * 3.14)
                    dead_cells.remove(dead_cell)
                    self.cells.remove(dead_cell)
            elif (total_mass < min_mass):
                #Create a new dead cell
                while(total_mass < min_mass):
                    new_cell = Cell.get_from(np.array([rand.randint(0,chess_size), rand.randint(0,chess_size)]), 10, np.array([0, 0]), 0)
                    total_mass += (new_cell.size**2 * 3.14)
                    self.cells.append(new_cell)
        return created_cells    
                
    def check_collisions(self):
        #Create matrix of quadrants
        quadrants = [ [[] for _ in range(chess_size//quadrant_size)] for _ in range(chess_size//quadrant_size) ]
        #Fill quadrants by looping on cells
        for cell in self.cells:
            x, y = cell.position
            x_q = int(x//quadrant_size)
            y_q = int(y//quadrant_size)
            if (y_q >= chess_size//quadrant_size):
                y_q = chess_size//quadrant_size - 1
            if (x_q in range(chess_size//quadrant_size) and y_q in range(chess_size//quadrant_size) and cell.size > 0):
                quadrants[x_q][y_q].append(cell)
            #If cell size > quadrant size, add to other quadrants too
            overflow = int(cell.size // quadrant_size)+1
            for i in range(1, overflow+1):
                if (x_q+i < chess_size//quadrant_size):
                    quadrants[x_q+i][y_q].append(cell)
                    for j in range(1, overflow+1):
                        if (y_q+j < chess_size//quadrant_size):
                            quadrants[x_q+i][y_q+j].append(cell)
                        if (y_q-j >= 0):
                            quadrants[x_q+i][y_q-j].append(cell)
                if (x_q-i >= 0 and x_q-1 < chess_size//quadrant_size):
                    quadrants[x_q-i][y_q].append(cell)
                    for j in range(1, overflow+1):
                        if (y_q+j < chess_size//quadrant_size):
                            quadrants[x_q-i][y_q+j].append(cell)
                        if (y_q-j >= 0):
                            quadrants[x_q-i][y_q-j].append(cell)
        #Check collisions
        def check_single_collision(cell, other_cell):
            distance = np.linalg.norm(cell.position - other_cell.position)
            if distance < (cell.size + other_cell.size):
                if (cell.lifespan <= 0):
                    if (other_cell.lifespan <= 0):
                        return
                    winner = other_cell; loses = cell;
                elif (other_cell.lifespan <= 0):
                    winner = cell; loses = other_cell;
                else:
                    winner = cell if cell.size >= other_cell.size else other_cell
                    loses = cell if cell.size < other_cell.size else other_cell
                size_of_collision = min(loses.size, loses.size +  winner.size - distance, winner.size)
                loses.size -= size_of_collision
                #Compute increase in size of winner
                r_new = sqrt(
                    winner.size**2 + loses.size**2 + (loses.size+size_of_collision)**2
                )
                winner.speed *= sqrt((winner.size**2)/(r_new**2))
                winner.size = r_new
                if (winner.size > max_cell_size):
                    winner.size = max_cell_size
                if loses.size <= 0:
                    loses.size = 0
                    if (loses in self.cells):
                        self.cells.remove(loses)

        for i in range(len(quadrants)):
            for j in range(len(quadrants[i])):
                for index, cell in enumerate(quadrants[i][j]):
                    for other_cell in quadrants[i][j][index+1:]:
                        check_single_collision(cell, other_cell)

    def check_collisions_quadratic(self):
        #TODo this method must be optimized
        for index, cell in enumerate(self.cells):
            for other_cell in self.cells[index+1:]:
                distance = np.linalg.norm(cell.position - other_cell.position)
                if distance < (cell.size + other_cell.size):
                    winner = cell if cell.size >= other_cell.size else other_cell
                    loses = cell if cell.size < other_cell.size else other_cell
                    size_of_collision = min(loses.size, loses.size +  winner.size - distance)
                    loses.size -= size_of_collision
                    #Compute increase in size of winner
                    r_new = sqrt(
                        winner.size**2 + loses.size**2 + (loses.size+size_of_collision)**2
                    )
                    winner.size = r_new
                    if loses.size <= 0:
                        loses.size = 0
                        if (loses in self.cells):
                            self.cells.remove(loses)

    def get_visual_map(self):
        #Return a matrix of size chess_size * chess_size with 1 if cell present there, 0 otherwise
        #The matrix is made with torch so it can be used for the neural network
        map = torch.zeros((chess_size, chess_size))
        map = map.cuda()
        for cell in self.cells:
            map = get_map(map, cell)
        #Set each position in map to 1 if > 1
        map[map > 1] = 1
        #Resize the map to size view_size * view_size
        map = torch.nn.functional.interpolate(map.reshape(1, 1, chess_size, chess_size), size=(view_size, view_size), mode='nearest')
        map = map.reshape(view_size, view_size, 1)
        return map
    
    def save_state(self, name):
        with open(f"saved_state/{name}.txt", "w") as f:
            for index, cell in enumerate(self.cells):
                fn = f"saved_state/neural_weights/{name}_{index}.txt"
                if (cell.export_parameters(fn)==False):
                    fn = ""
                f.write(f"{cell.position[0]},{cell.position[1]},{cell.size},{cell.speed[0]},{cell.speed[1]},{cell.lifespan},{fn}\n")
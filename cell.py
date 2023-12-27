from math import sqrt
import random as rand
import numpy as np
import pygame as pg
import torch
from params import *
from utils import *
from neural_network import CellNeuralNetwork

class Cell:
    def __init__(self, cell_size = 20):
        #Position = (random, random)
        self.position = np.array([rand.random()*chess_size, rand.random()*chess_size])
        self.size = cell_size
        self.speed = np.array([0, 0])
        self.lifespan = cell_lifespan
        self.memory = None
        self.adjust_position()
        self.neural_network = CellNeuralNetwork().cuda()
        self.last_decision = -1
    def get_from(position, size, speed, lifespan, neural_network=None):
        self = Cell()
        self.position = position
        self.size = size
        self.speed = speed
        self.lifespan = lifespan
        self.memory = None
        self.adjust_position()
        self.neural_network = neural_network
        return self 
    
    def export_parameters(self, filename):
        if (self.neural_network):
            self.neural_network.export(filename)
            return True
        return False
    
    def revitalize(self):
        self.lifespan = cell_lifespan
        self.memory = None
        self.neural_network = CellNeuralNetwork().cuda()
    
    def adjust_position(self):
        border_left = self.position[0] - self.size
        border_right = self.position[0] + self.size
        border_top = self.position[1] - self.size
        border_bottom = self.position[1] + self.size
        changed = [1, 1]
        if (border_left < 0):
            self.position[0] = self.size
            changed[0] = -1
        elif (border_right > chess_size):
            self.position[0] = chess_size - self.size
            changed[0] = -1
        if (border_top < 0):
            self.position[1] = self.size
            changed[1] = -1
        elif (border_bottom > chess_size):
            self.position[1] = chess_size - self.size
            changed[1] = -1
        return changed
    
    def update_position(self):
        self.position = self.position + self.speed
        self.speed = self.speed * friction
        self.lifespan -= 1
        self.speed *= self.adjust_position()


    def apply_force(self, direction):
        #Direction is in degree
        if (self.size < 1):
            return None
        r_new = sqrt(
                        self.size**2 - (self.size - 1)**2
                    )
        
        angle_coords = np.array([np.cos(direction), np.sin(direction)])
        self.speed = self.speed + (angle_coords * cell_speed / sqrt(self.size**2))
        self.size -= 1
        return Cell.get_from(self.position - (angle_coords * (self.size+1+r_new)), r_new, -angle_coords*5, 0)
    
    def export_memory(self):
        #Open file in append mode
        #Export the torch tensor to the file
        t_np = self.memory[:,:,-1].cpu().numpy() #convert to Numpy array
        np.savetxt("memory.csv", t_np, delimiter=",")
    
    def update_visual(self, visual_map):
        if (self.lifespan <= 0):
            return 0
        TimeMeasure.start()
        #Make self-visual_map a 3D matrix of size chess_size * chess_size
        self_visual = torch.zeros((chess_size, chess_size)).cuda()
        self_visual = get_map(self_visual, self)
        #Resize the map to size view_size * view_size
        self_visual = torch.nn.functional.interpolate(self_visual.reshape(1, 1, chess_size, chess_size), size=(view_size, view_size), mode='nearest')
        self_visual = self_visual.reshape(view_size, view_size, 1)
        #Make a 3D matrix of size chess_size * chess_size * 2 containing both of them 
        if (self.memory is None):
            #self.memory equals visual_map repeated cell_memory times along the 3rd dimension
            self.memory = torch.cat((self_visual, visual_map), dim=2)
            #Repeat self.memory 3 times over dim 2
            self.memory = self.memory.repeat(1, 1, cell_memory)
        else:
            self.memory = torch.cat((self.memory[:,:,2:], torch.cat((self_visual, visual_map), dim=2)), dim=2)
        TimeMeasure.stop("Update_visual")
    
    def make_decision(self, print_it=False):
        if (self.lifespan <= 0 or self.size < 1):
            return 0
        decision = self.neural_network(self.memory.reshape(1, cell_memory*2, view_size, view_size), self.lifespan, self.size)
        #Get index of max element in decision
        if (print_it):
            print(decision)
        decision = torch.argmax(decision).item()
        return decision
    #Return values: 0: nothing; 1-8: direction to apply force; 9: reproduce

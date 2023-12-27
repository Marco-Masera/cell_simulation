import torch
import time
from params import *

measures = {}
class TimeMeasure:
    time_ = None
    def start():
        TimeMeasure.time_ = time.time()
    def stop(name):
        if (name in measures):
            measures[name] += time.time() - TimeMeasure.time_
        else:
            measures[name] = time.time() - TimeMeasure.time_


single_circle = torch.zeros((view_size, view_size))
for i in range(view_size):
    for j in range(view_size):
        if ((i-view_size/2)**2 + (j-view_size/2)**2 < view_size**2/4):
            single_circle[i, j] = 1
#Resize the map to size view_size * view_size
single_circle = single_circle.cuda()
single_circle = torch.nn.functional.interpolate(single_circle.reshape(1, 1, view_size, view_size), size=(chess_size, chess_size), mode='nearest').reshape(chess_size, chess_size)

def get_map(map, cell):
    #Create a copy of self.single_circle
    circle = single_circle.clone()
    scaled_size = int(cell.size * 2)
    if (scaled_size == 0):
        scaled_size = 1
    elif (scaled_size > chess_size):
        scaled_size = chess_size
    circle = torch.nn.functional.interpolate(circle.reshape(1, 1, chess_size, chess_size), size=(scaled_size, scaled_size), mode='nearest').reshape(scaled_size, scaled_size)
    #Traslate the matrix to the position of the cell using 0s as padding
    x, y = cell.position
    x = int(x) - scaled_size//2
    y = int(y) - scaled_size//2
    end_x = x + scaled_size
    end_y = y + scaled_size
    if (x < 0):
        x = 0
    elif (x + scaled_size >= chess_size):
        end_x = chess_size-1
    if (y < 0):
        y = 0
    elif (y + scaled_size >= chess_size):
        end_y = chess_size-1
    map[x:end_x, y:end_y] += circle[:(end_x-x), :(end_y-y)]
    return map
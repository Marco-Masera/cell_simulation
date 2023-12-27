import pygame as pg

white = (255, 255, 255)
class Graphics:
    def initialize(self, width, height):
        #Create empty window of size window_size
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption("Title")
    def update(self, cells):
        #Update the window with the new visual_map
        self.screen.fill(white)
        for cell in cells:
            if (cell.lifespan > 0):
                color = (0, 0, 0)
            else:
                color = (255, 0, 0)
            pg.draw.circle(self.screen, color, (cell.position[0], cell.position[1]), cell.size)
        pg.display.flip()
    def debug_use_map(self, map):
        print(map.shape)
        self.screen.fill(white)
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if (map[i, j] >= 1):
                    pg.draw.circle(self.screen, (1,0,0), (i, j), 1)
        pg.display.flip()
    def close(self):
        #Close the window
        pass 
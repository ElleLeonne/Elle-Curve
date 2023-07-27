import numpy as np
import matplotlib.pyplot as plt
__version__ = 0.1

#An initial implementation of our space-filling curve. 
#TO-DO: interpolation and worker sharding

def graph_grid(grid):
    plt.matshow(grid)
    for (i, j), value in np.ndenumerate(grid):
        plt.text(j, i, int(value), ha='center', va='center', color='red')
    plt.show()

def test(object):
    """Tests every grid permutation size between (1, 1) and (10, 10) to ensure the algorithm is sound"""
    import math
    object.test = True
    for i in range(5, 10):
        for j in range(5, 10):
            grid, iter, max = object(i, j)
            if iter < 0:
                iter = 0
            if max < 0:
                max = 0
            if iter != max:
                raise Exception(f"Expected {max} moves, but got {iter}")
            total = (i*j) - 1
            expected_sum = total * (total + 1) // 2
            actual_sum = np.sum(grid)
            if expected_sum != actual_sum:
                raise Exception(f"Expected {expected_sum} total units in test array, got {actual_sum}")
    print("All checks passed, EC curve-fit algorithm functions as expected.")
            
class ElleCurve():
    """An EC is a space-filling curve specifically designed for CV-based machine learning model positional embeddings,
    because it prioritizes the center of the image while still preserving spatial locality.
    Inspired by Hilbert Curves, <3 you big guy, promise V2 won't let you down so much :)"""

    def __init__(self, debug=False):
        """
        debug: Whether to allow the user to step through the individual generation steps.
        """
        #Lookup tables for easier coding
        self.dir = {"left": np.array([0, -1]), "right": np.array([0, 1]), "up": np.array([-1, 0]), "down": np.array([1, 0])}
        self.invert = {"left": "right", "right": "left", "up": "down", "down": "up"}
        #We ain't livin' if we ain't cachin'
        self.cache = {}
        self.debug = debug
        self.test = False

    def init(self, h, w):
        """We actually want to explicitly call this version to reset the maze
        We use class variables to avoid passing a million values around our functions."""
        #Remaining x & y squares left to be consumed
        self.r_h, self.r_w, self.iter = h, w, 0 #And then grid coordinates are 0-indexed
        self.max = h*w-1
        self.grid = np.zeros((h, w), dtype=np.int32)
        self.path = ["up", "left", "down", "right"] #Ordering of jog directions
        self.lr_roto = ["left", "right"] #Rotations for jogs
        self.ud_roto = ["down", "up"]
        turtle = np.array([h-1, w-1])

        #debug
        if self.debug is True:
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.matshow(self.grid, cmap='gray')
            self.texts = np.array([[self.ax.text(j, i, str(self.grid[i,j]), ha='center', va='center', color='red')
                                    for j in range(self.grid.shape[1])] for i in range(self.grid.shape[0])])
            plt.show(block=False)
        #debug
        return turtle

    def init2(self, turtle):
        """Our init for an actual EC trace"""
        if self.r_w > 1:
            if self.r_h % 2 == 0: #If even, we go right first
                self.lr_roto.reverse()
                turtle = turtle + self.dir["left"] #We want an inner-start

            #jog 1 is here, because it's special and ignores rules
            direction = self.path.pop(0)
            turtle = self.pivot(turtle) #Jog #1 starts w/ a pivot first
            for _ in range(self.r_h-1):
                turtle = self.move(turtle, direction)
                turtle = self.pivot(turtle)
            self.cleanup(2, direction)

        return turtle, "outer" #Jog #1 always ends w/ "outer" positioning

    #----Directional Checks----
    @property
    def vertical(self): #This gets called after .pop(), so it's technically inverted.
        """The direction of our jog"""
        return True if self.path[0] in ["left", "right"] else False

    def debug_grid(self):
        self.im.set_array(self.grid)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                self.texts[i, j].set_text(str(self.grid[i, j]))
        self.fig.canvas.draw()
        input()

    #------Movement Logic------
    def mark_grid(self, turtle):
        """Stamps current iter with the location of our cursor"""
        if self.iter < self.max: #Conditional is needed to handle really small images. May be faster to hard-code those.
            self.iter += 1
            self.grid[turtle[0], turtle[1]] = self.iter

            if self.debug is True:
                self.debug_grid()
        
    def move(self, turtle, direction):
        """incrementing move"""
        turtle = turtle + self.dir[direction]
        self.mark_grid(turtle)
        return turtle
    
    def pivot(self, turtle):
        """Non-incrementing move, oscillates between two directors"""
        if self.vertical:
            turtle = turtle + self.dir[self.lr_roto[0]]
            self.lr_roto.reverse()
        else:
            turtle = turtle + self.dir[self.ud_roto[0]]
            self.ud_roto.reverse()
        self.mark_grid(turtle)
        return turtle

    def cleanup(self, width, direction):
        if self.vertical:
            self.r_w -= width
        elif self.vertical is False:
            self.r_h -= width
        self.path.append(direction)
    #----------------------------

    def get_remaining(self, position):
        """Determines whether we need to wrap around the corner or not"""
        remaining, max_width = (self.r_h, self.r_w) if self.vertical is True else (self.r_w, self.r_h)
        need_wrap = False

        if remaining % 2 == 0:
            if position == "outer":
                need_wrap = True
        else:
            if position == "inner":
                need_wrap = True
        return remaining, max_width, need_wrap
    
    def jog(self, turtle, position):
        """Pathing logic"""
        direction = self.path.pop(0)
        distance, max_width, wrap_corner = self.get_remaining(position)
        if (max_width == 3 and position == "outer") and wrap_corner is True:
            """Stops us from getting stuck in corners, and also adds some extra locality information at the expense of missing the center sometimes."""
            for _ in range(distance):
                turtle = self.move(turtle, direction)
                side_dir = self.lr_roto[0] if self.vertical else self.ud_roto[0]
                turtle = self.move(turtle, side_dir)
                turtle = self.move(turtle, side_dir)
                self.lr_roto.reverse() if self.vertical else self.ud_roto.reverse()
            self.cleanup(3, direction)
        elif wrap_corner and max_width > 1:
            for _ in range(distance-2):
                turtle = self.move(turtle, direction)
                turtle = self.pivot(turtle)
            turtle = self.move(turtle, direction)
            turtle = self.move(turtle, direction)
            turtle = self.pivot(turtle)
            self.ud_roto.reverse() if self.vertical else self.lr_roto.reverse()
            turtle = self.move(turtle, self.invert[direction])
            self.cleanup(2, direction)
        else:
            for _ in range(distance):
                turtle = self.move(turtle, direction)
                if max_width > 1:
                    turtle = self.pivot(turtle)
            self.cleanup(2, direction)

        return turtle, "inner" if wrap_corner else "outer"
    
    def spiral(self, turtle):
        """For small images, we get better results by just using a spiral pattern. Also conveniently avoids messy logic hacks"""
        #first jog is always different because we can't start off the grid
        direction = self.path.pop(0)
        distance = self.r_h - 1 if self.vertical is True else self.r_w - 1
        print(distance)
        for _ in range(distance):
            turtle = self.move(turtle, direction)
        self.cleanup(1, direction)

        while self.iter < self.max:
            direction = self.path.pop(0)
            distance = self.r_h if self.vertical is True else self.r_w
            for _ in range(distance):
                print(distance, direction)
                turtle = self.move(turtle, direction)
            self.cleanup(1, direction)

        if self.test:
            return self.grid, self.iter, self.max
        return self.grid

    def __call__(self, h, w):
        """We use (h, w) to cope w/ numpy's (y, x) convention"""
        turtle = self.init(h, w) #Help Terry the Turtle solve our image compression algorithm!
        if self.r_h < 5 or self.r_w < 5:
            return self.spiral(turtle)
        turtle, position = self.init2(turtle)

        while self.r_h > 0 or self.r_w > 0:
            turtle, position = self.jog(turtle, position)

        if self.test == True:
            return self.grid, self.iter, self.max
        return self.grid


curve_gen = ElleCurve()
#test(curve_gen)
graph_grid(curve_gen(5, 5))



"""
Each jog can end at an inner-join or an outer-joint, where the position is w/r/t the outside or inside of the image
Leg #1 is special and can't be generalized. Jog legs occur in pairs of 2
-If even, start "inner" if odd, start "outer". Both end "outer".
Then, from there:

Leg #2 will always start outer.
If remaining_h (or w, whatever jog direction) is:
--Even, and starting position is outer, we need an L hook to round the corner.
--Odd, and starting position is inner, we need an L hook.
L hooks always return "inner" starting position.
Else, we return "outer" starting position.
"""        

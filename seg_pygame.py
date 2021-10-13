'''
Schelling's segregation agent based simulation
** WITH PYGAME INTERFACE **

Data scientists live in a 2D gridword. Each cell can only be inhabited
by a single data scientist.

Data scientists either favour Python or R. They have a similarity threshold
where they desire a percentage of their neighbours to code in their favourite
language too!  If the agents neighbours are not similar enough the to a random
empty location in the world!

The module contains three classes:

* DataScientist: an abstraction of a data scientist with a coding
language preference
* GridWorld: the environment that a DataScientist lives in
* Model: The simulation model itself.

PYGAME MODIFICATIONS: 
---------------------
The main modifications occur in the Model class.  These are highlighted in the
code using comments.  Key places to look:

* Imports and constants - Lots of setup code. Incl. code for colours etc.

* Model class
    * .__init__() - setup display objects and variables
    * .run() - now calls a new set of display methods after each iteration
    * .draw_agents() - code to actually show the agents on the display surface.


RUNNING THE CODE:
-----------------
The Github repo contains `environment.yml`. 
To create a virtual environment on a machine run:
* `conda env create -f environment.yml`
* `conda activate hds_abs`

The key dependency is pygame 2.0.2 (pip install pygame==2.0.2)

The code that creates a game and runs it is in main(). You can find this at the 
end of the module.

You can either run the code in an IDE like VScode or Spyder or you can run 
from an anaconda command/prompt (Windows) or Terminal (Linux/Mac).  For a 
cmd prompt/terminal use:

`python seg_pygame.py`

'''

import itertools
import random
import time
import matplotlib.pyplot as plt

# PYGAME IMPORTS ##################################################

import sys
import pygame
from pygame.locals import QUIT, KEYUP, K_ESCAPE

###################################################################

# grid defaults
N_ROWS = 50
N_COLS = 50
N_CELLS = N_ROWS * N_COLS

# coder language constants
LANG_PYTHON = 'PYTHON'
LANG_R = 'R'

# SIMULATION PARAMETERS
RATIO_R_TO_PYTHON = 0.5
PERCENT_EMPTY = 0.3
SIMILARITY_THRESHOLD = 0.3
MAX_ITER = 500

# PYGAME MODIFICATIONS ##########################################

# pygame interface constants
FPS = 15
UPDATE_DELAY = 250
WINDOWWIDTH = 640
WINDOWHEIGHT = 640

# cell width AND HEIGHT in pixels
CELLWIDTH = WINDOWWIDTH / N_COLS
CELLHEIGHT = WINDOWHEIGHT / N_ROWS

# Colour pallete
#               R    G    B
WHITE       = (255, 255, 255)
BLACK       = (  0,   0,   0)
RED         = (255,   0,   0)
GREEN       = (  0, 255,   0)
DARKGREEN   = (  0, 155,   0)
DARKGRAY    = ( 40,  40,  40)
BRIGHTRED   = (255,   0,   0)
RED         = (155,   0,   0)
BRIGHTBLUE  = (  0,   0, 255)
BLUE        = (  0,   0, 155)
YELLOW      = (255, 255,   0)  

# color coding
BGCOLOR_START = DARKGRAY
BGCOLOR = BLACK
PYTHON_COLOR = BLUE
PYTHON_INNER_COLOR = BRIGHTBLUE
R_COLOR = RED
R_INNER_COLOR = BRIGHTRED

# dicts for lookup of colors.
AGENT_COLORS = {LANG_PYTHON: PYTHON_COLOR,
                LANG_R: R_COLOR}
AGENT_INNER_COLORS = {LANG_PYTHON: PYTHON_INNER_COLOR,
                      LANG_R: R_INNER_COLOR}

TITLE_CAPTION = 'Data Scientist Segregation Model'
PRESS_ANY = 'Press any key to run simulation.'

################################################################


class DataScientist:
    '''
    Encapsulates a coder agents attributes and behaviour
    '''
    def __init__(self, id, row, col, language, env,
                 threshold=SIMILARITY_THRESHOLD):
        '''
        DataScientist

        Params:
        ------
        id: int
            unique id
        row: int
            row in gridworld
        col: int
            col in gridworld
        language: str
            preference for python or R
        env: GridWorld
            environment in which the Data Scientist lives and
            interacts with other agents
        threshold: float
            similarity threshold for immediate neighbourhood.
        '''
        self.id = id
        self.row = row
        self.col = col
        self.language = language
        self.env = env
        self.threshold = threshold

    def _get_coordinates(self):
        '''
        Get the coordinates of the agent in gridworld

        Returns:
        -------
        tuple (row, col)
        '''
        return (self.row, self.col)

    def _set_coordinates(self, coords):
        '''
        Set coordinates of the agent in gridworld

        Params:
        ------
        coords: (int, int)
        '''
        if type(coords) is tuple and len(coords) == 2:
            self.row, self.col = coords[0], coords[1]
        else:
            raise ValueError('Coordinartes should be (int, int)')

    def is_unsatified_with_neighbours(self):
        '''
        Does the data scientists neighbourhood violate
        its similarity constraint?
        I.e. is the number of neighbours below similarity threshold.

        Returns:
        -------
        bool
        '''
        neighbours = self.env.get_neighbours(self.row, self.col)
        n_similar = len([n for n in neighbours if n.language == self.language])

        if len(neighbours) == 0:
            return False
        else:
            return (float(n_similar) / float(len(neighbours))) < self.threshold

    coordinates = property(_get_coordinates, _set_coordinates)


class GridWorld:
    '''
    Encapsulates a gridworld environment for coders to live in.
    '''
    def __init__(self, n_rows, n_cols, n_empty, random_seed=None):
        '''
        GridWorld - a 2D grid environment containing data scientists
        who code in either Python or R and are intolerant to each
        other!

        Params:
        ------
        n_rows: int
            Number of rows in the grid

        n_cols: int
            Number of cols in the grid

        n_empty: int
            Number of cells in the grid that remain empty
            at any one time

        random_seed: int, optional (default=None)
            Control the randomisation for a repeatable
            run of the simulation.

        '''
        # initial grid is all empty -
        self.n_agents = N_CELLS - n_empty
        # grid world represented as list of lists - all cells empty on init
        self.grid = [[None for i in range(n_cols)] for j in range(n_rows)]
        # 2d grid coordinates
        coords = list(itertools.product(range(n_rows), range(n_cols)))
        # shuffle coordinates to use to place agents
        random.seed(random_seed)
        random.shuffle(coords)

        # create agents
        self.agents = []
        for i in range(self.n_agents):
            if i < int(self.n_agents * RATIO_R_TO_PYTHON):
                lang = LANG_R
            else:
                lang = LANG_PYTHON

            # create agent and store in list of agents and grid
            # remember that coords has been shuffled (randomised) beforehand
            agent = DataScientist(id=i, row=coords[i][0], col=coords[i][1],
                                  language=lang, env=self)
            self.grid[agent.row][agent.col] = agent
            # its useful to have a seperate list of agents
            self.agents.append(agent)

        # the empty cells in the gridworld
        self.empty_cells = [coord for coord in coords[self.n_agents:]]

        print(f'empty cells: {len(self.empty_cells)}, expected: {n_empty}')
        print(f'agents: {len(self.agents)}')
        print(f'grid size: {N_CELLS}')

    def get_neighbours(self, row, col):
        '''
        Return the feasible neighbours for row and col

        Params:
        -------
        row: int
            The row in gridworld

        col: int
            The column in gridworld

        Returns:
        --------
        list
            List of agents in current neighbourhood of
            row and col parameters
        '''
        # coordinates of neighbours (some may be infeasible)
        coords = ((row - 1, col - 1),
                  (row - 1, col),
                  (row - 1, col + 1),
                  (row, col - 1),
                  (row, col + 1),
                  (row + 1, col - 1),
                  (row + 1, col),
                  (row + 1, col + 1))

        # convert feasible non empty coordinates into list of agents
        feasible = [self.grid[c[0]][c[1]] for c in coords
                    if c[0] >= 0 and c[1] >= 0
                    and c[0] < N_ROWS and c[1] < N_COLS
                    and c not in self.empty_cells]

        return feasible

    def relocate(self, agent):
        '''
        Stochastic movement of an agent to a unused location
        '''
        new_loc = self.get_random_empty_cell()
        old_loc = agent.coordinates

        # swap locations in grid and update agents coordinates.
        self.grid[new_loc[0]][new_loc[1]] = agent
        self.grid[old_loc[0]][old_loc[1]] = None
        agent.coordinates = (new_loc[0], new_loc[1])

        # swap empty and full coords in lookup
        self.empty_cells.remove(new_loc)
        self.empty_cells.append(old_loc)

    def get_random_empty_cell(self):
        '''
        Return a random empty cell
        '''
        return random.choice(self.empty_cells)


def plot(agents):
    '''
    Plot agents gridworld a 2d matplotlib chart
    '''
    _, ax = plt.subplots()
    agent_colors = {LANG_PYTHON: 'b', LANG_R: 'r'}
    for agent in agents:
        ax.scatter(agent.row+0.5, agent.col+0.5,
                   color=agent_colors[agent.language])

    plt.show()


class Model:
    '''
    Data Scientist Python-R segregation model.
    '''
    def __init__(self, environment, max_iter=MAX_ITER):
        self.env = environment
        self.max_iter = max_iter

        # PYGAME modifications ###############################################

        # dict of agent colours
        self.agent_colors = {LANG_PYTHON: PYTHON_COLOR, LANG_R: R_COLOR}
        self.inner_colors = {LANG_PYTHON: PYTHON_COLOR, LANG_R: R_COLOR}

        # initialise pygame framework
        pygame.init()

        # clock
        self.FPSCLOCK = pygame.time.Clock()
        
        # display surface
        self.DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        
        # basic font for text included in pygame
        self.BASICFONT = pygame.font.Font('freesansbold.ttf', 18)
        self.STARTFONT = pygame.font.Font('freesansbold.ttf', 32)

        # window caption
        pygame.display.set_caption(TITLE_CAPTION)
        self.DISPLAYSURF.fill(BGCOLOR)
        
        #######################################################################

    def run(self):
        '''
        Run the segregation simulation model
        '''
        
        # PYGAME START SIM MODIFICATION #######################################
    
        # This prompts the user to press a key to start (ESC cancels)
        self.wait_for_user_to_press_key()
       
        # Draw initial solution
        self.draw_environment()
        pygame.display.update()
        self.FPSCLOCK.tick()
        
        #######################################################################
        
        start = time.time()
        print('iterations: ')
        converged, iteration = False, 0
        while not converged and iteration < self.max_iter:
            converged = True
            iteration += 1
            # agents that are disatisified
            to_move = []
            for agent in self.env.agents:

                if agent.is_unsatified_with_neighbours():
                    # store a reference to the disatisified agent
                    to_move.append(agent)

            if len(to_move) > 0:
                converged = False

            # relocated the agents
            for agent in to_move:
                self.env.relocate(agent)

            # PYGAME MODIFICATIONS ###########################################

            # Update the display after simulation iteration
            self.draw_environment()

            # Update the screen
            pygame.time.wait(UPDATE_DELAY)
            pygame.display.update()
            self.FPSCLOCK.tick()

            ##################################################################

            # report every 20 iterations
            if iteration % 20 == 0:
                print(f'{iteration}: proportion population moved: '
                      + f'{len(to_move)/len(self.env.agents)}')

        print(f'\nSimulation completed after {iteration} iterations')
        duration = time.time() - start
        print(f'Runtime = {duration:.2f}')

    # PYGAME MODIFICATIONS TO METHODS ########################################

    def draw_environment(self):
        '''
        Draw the environment
        '''
        self.DISPLAYSURF.fill(BGCOLOR)
        self.draw_agents()


    def draw_agents(self):
        # loop through each agent and color code based on preference
        for agent in self.env.agents:
            # top left pixel = coordinates * CELLWIDTH/HEIGHT
            left = agent.col * CELLWIDTH
            top = agent.row * CELLHEIGHT
            agent_rect = pygame.Rect(left, top, CELLWIDTH, CELLHEIGHT)
            pygame.draw.rect(self.DISPLAYSURF,
                             AGENT_COLORS[agent.language],
                             agent_rect)
            inner_segment_rect = pygame.Rect(left + 4, top + 4, CELLWIDTH - 8,
                                             CELLHEIGHT - 8)
            pygame.draw.rect(self.DISPLAYSURF,
                             AGENT_INNER_COLORS[agent.language],
                             inner_segment_rect)

            
    def wait_for_user_to_press_key(self):
        '''
        Press any key message on splash screen
    
        Returns
        -------
        None.
    
        '''
        
        self.DISPLAYSURF.fill(BGCOLOR_START)
        # load the R and python image - this is a surface
        py_img = pygame.image.load('python-logo.png')
        r_img = pygame.image.load('Rlogo.png')
        
        # clear out any key presses in the event queue
        self.check_for_key_press()  
        
        self.DISPLAYSURF.blit(py_img, (0, 0))
        self.DISPLAYSURF.blit(r_img, (400, 0))
                
        press_key_surf = self.STARTFONT.render(PRESS_ANY, True, WHITE)
        press_key_rect = press_key_surf.get_rect()
        press_key_rect.topleft = (WINDOWWIDTH / 9, 
                                  WINDOWHEIGHT / 2)
        self.DISPLAYSURF.blit(press_key_surf, press_key_rect)
        pygame.display.update()
        
        # wait for key press
        while True:
            if self.check_for_key_press():
                pygame.event.get() # clear event queue
                return
        
    def check_for_key_press(self):
        if len(pygame.event.get(QUIT)) > 0:
            self.terminate()
    
        key_up_events = pygame.event.get(KEYUP)
    
        if len(key_up_events) == 0:
            return None
    
        if key_up_events[0].key == K_ESCAPE:
            self.terminate()
    
        return key_up_events[0].key
    
    def terminate(self):
        pygame.quit()
        sys.exit()
    
    ##################################################################






def main():
    '''
    Main routine.
    Creates a gridworld and runs a segregation model
    Plots reults in 2d matplotlib chart
    '''
    print('Running Data Scientist Segregation Model')
    env = GridWorld(n_rows=N_ROWS, n_cols=N_COLS,
                    n_empty=int(PERCENT_EMPTY * N_CELLS),
                    random_seed=42)
    model = Model(env)
    model.run()


if __name__ == '__main__':
    main()

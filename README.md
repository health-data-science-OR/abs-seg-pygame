**MSc materials for running the ABS segregation simulator with a simple pygame interface.**

# Schelling's segregation agent based simulation **WITH PYGAME INTERFACE**

Data scientists live in a 2D gridword. Each cell can only be inhabited
by a single data scientist.

Data scientists either favour Python or R. They have a similarity threshold
where they desire a percentage of their neighbours to code in their favourite
language too!  If the agents neighbours are not similar enough the to a random
empty location in the world!

The module `seg_pygame.py` contains three classes:

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

The key dependency is pygame 2.0.2 (`pip install pygame==2.0.2`)

The code that creates a game and runs it is in `main()`. You can find this at the 
end of the module.

You can either run the code in an IDE like Visual Studio Code or Spyder, or you can run 
from an Anaconda Prompt (Windows) or Terminal (Linux/Mac).  For Anaconda Prompt/Terminal use:

* `python seg_pygame.py`
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random
import logging
import numpy as np
import time



def agent(obs, config):
	board = Board(obs, config)
	me = board.current_player


	for ship in me.ships:
		if(len(me.shipyards) < 1):
			ship.next_action = ShipAction.CONVERT
		else:
			ship.next_action = ShipAction.NORTH

	for shipyard in me.shipyards:
		if(me.halite > 2000):
			shipyard.next_action = ShipyardAction.SPAWN


	return me.next_actions
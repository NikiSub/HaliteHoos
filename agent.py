from random import choice
from kaggle_environments.envs.halite.halite import get_to_pos
import logging
import numpy as np
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)


CONVERT = "CONVERT"
SPAWN = "SPAWN"
DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]

SIZE = 21


def get_2d_halite_board(obs):
	return np.array(obs.halite).reshape(SIZE,SIZE)

def convert_pos_to_2d(ship_info):
	# my_ships[agent_id][0]%15,my_ships[agent_id][0]//15
	return (ship_info[0]%SIZE, ship_info[0]//SIZE)

def get_nearest_halite(curr_pos, halite_board):
	hx, hy = np.where(halite_board > 100)
	halite_coords = list(zip(hx,hy))
	distances = {}

	for i in halite_coords:
		# find euclidean distance, doesn't take into account wrap around
		dist = np.sqrt((i[0] - curr_pos[0])**2 + (i[1] - curr_pos[1])**2)
		distances[i] = dist
	# from the dict get the closest set of coords     
	closest_xy =  min(distances, key=distances.get)
	return closest_xy

def go_to_location(curr_pos, target_pos):
	if(curr_pos[1] > target_pos[1]):
		return DIRS[0]
	elif(curr_pos[1] < target_pos[1]):
		return DIRS[1]
	elif(curr_pos[0] < target_pos[0]):
		return DIRS[2]
	elif(curr_pos[0] > target_pos[0]):
		return DIRS[3]
	else:
		return None

global states
states = {}

COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"



def agent(obs):
	actions = {}
	halite, shipyards, ships = obs.players[obs.player]
	opp_halite, opp_shipyards, opp_ships = obs.players[1]

	#logging.debug(str(len(obs.halite)) +" , "+ str(obs.halite))

	halite_board = get_2d_halite_board(obs)

	for uid, shipyard in shipyards.items():
		if(len(ships) == 0):
			actions[uid] = SPAWN

	for uid, ship in ships.items():
		if(len(shipyards) == 0):
			actions[uid] = CONVERT
			continue

	for uid, ship_info in ships.items():
		print(ship_info)
		ship_coords = convert_pos_to_2d(ship_info)
		if(uid not in states):
			states[uid] = COLLECT
		if(states[uid] == COLLECT):
			if(ship_info[1] > 500):
				states[uid] = DEPOSIT
			else:
				halite_coords = get_nearest_halite(ship_coords, halite_board)
				action = go_to_location(ship_coords, halite_coords)
				if(action is not None):
					actions[uid] = action

				#logging.debug(str(halite) + ", " +str(shipyards) + ", "+str(ships))
				#logging.debug(str(action) + " " + str(halite_coords) + " " + str(ship_coords))
				#print(action, ship_coords, halite_coords)
	
	return actions
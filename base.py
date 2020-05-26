import random
from kaggle_environments.envs.halite.halite import get_to_pos
import logging
import numpy as np
import time
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)

DIRS = ["NORTH", "SOUTH", "EAST", "WEST", None]
DIRS_TO_NUM = {"NORTH":-15, "SOUTH":15, "EAST":1, "WEST":-1, None:0}

COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"

CONVERT = "CONVERT"
SPAWN = "SPAWN"

BOARD_DIMS = 21

def int_to_coords(num):
	return (num%BOARD_DIMS, int(num//BOARD_DIMS))

def manhattan_distance(p1, p2):
	return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


class HaliteBoard():
	def __init__(self, obs):
		self.height = self.width = BOARD_DIMS

		_, self.shipyards, self.ships = obs.players[obs.player]
		self.enemy_data = []
		for i in range(1, len(obs.players)):
			self.enemy_data.append((None, obs.players[i][1], obs.players[i][2]))

		self.agent_board_1d = np.array(list(range(self.width*self.height)))
		self.agent_board_2d = np.array(list(range(self.width*self.height))).reshape(self.width, self.height)

		self.halite_board_1d = np.array(obs.halite)
		self.halite_board_2d = np.array(obs.halite).reshape(self.width,self.height)

	# gets halite locations on the map above a certain value threshold
	# returns a list of 2d coordinates of halite positions satisfying value > threshold
	def get_halite_locations(self, threshold):
		hx, hy = np.where(self.halite_board_2d > threshold)
		halite_coords = list(zip(hx,hy))
		return halite_coords

	# returns positions of ally shipyards
	def get_shipyard_locations(self):
		return [int_to_coords(shipyard) for uid, shipyard in self.shipyards.items()]

	def get_closest_shipyard(self, curr_pos):
		shipyard_coords = self.get_shipyard_locations()
		distances = {}
		for i in shipyard_coords:
			# find euclidean distance, doesn't take into account wrap around
			dist = np.sqrt((i[0] - curr_pos[0])**2 + (i[1] - curr_pos[1])**2)
			distances[i] = dist
		# from the dict get the closest set of coords     
		closest_xy =  min(distances, key=distances.get)
		return closest_xy

	# returns the position of the closest halite deposit above a certain threshold
	def get_closest_halite(self, curr_pos, threshold):
		halite_coords = self.get_halite_locations(threshold)
		distances = {}
		for i in halite_coords:
			# find euclidean distance, doesn't take into account wrap around
			dist = np.sqrt((i[0] - curr_pos[0])**2 + (i[1] - curr_pos[1])**2)
			distances[i] = dist
		# from the dict get the closest set of coords     
		closest_xy =  min(distances, key=distances.get)
		return closest_xy



class Agent():
	def __init__(self, ship, agent_id):
		self.halite = ship[0]
		self.agent_id = agent_id
		self.coords_1d = ship[0]
		self.coords_2d = int_to_coords(self.coords_1d)
		self.role = COLLECT

	def return_random_action(self):
		action = random.choice(DIRS)
		self.coords_1d += DIRS_TO_NUM[action]
		self.coords_2d = int_to_coords(self.coords_1d)
		return action

	def move_to_target_location(self, target_xy):
		posistion_xy = self.coords_2d
		if posistion_xy[0] < target_xy[0]:
			return 'EAST'
		if posistion_xy[0] > target_xy[0]:
			return 'WEST'
		if posistion_xy[1] < target_xy[1]:
			return 'SOUTH'
		if posistion_xy[1] > target_xy[1]:
			 return 'NORTH'
		return None

global states
states = {}

def agent(obs):
	start = time.time()
	actions = {}
	halite, shipyards, ships = obs.players[obs.player]
	#opp_halite, opp_shipyards, opp_ships = obs.players[1]

	board = HaliteBoard(obs)

	for uid, shipyard in shipyards.items():
		if(len(ships) == 0 or halite >= 3000):
			actions[uid] = SPAWN

	for uid, ship in ships.items():
		if(len(shipyards) == 0 and halite >= 2000):
			actions[uid] = CONVERT
			continue

	for uid, ship_info in ships.items():
		curr_ship = Agent(ship_info, uid)
		#print('STATE: ', curr_ship.agent_id, states[curr_ship.agent_id])
		if(uid not in states):
			states[uid] = COLLECT

		# collection logic: randomly move around until storage is > 500, at which point path to the closest shipyard
		if(states[uid] == COLLECT):
			if(ship_info[1] > 500):
				states[uid] = DEPOSIT
			else:
				action = curr_ship.return_random_action()
				if(action is not None):
					actions[uid] = action

		# deposit logic: path naively back to the closest shipyard
		if(states[uid] == DEPOSIT):
			closest_shipyard = board.get_closest_shipyard(curr_ship.coords_2d)
			if(curr_ship.coords_2d == closest_shipyard):
				states[uid] = COLLECT
			#print('DEPOSITING to ', curr_ship.coords_2d, closest_shipyard, len(board.get_shipyard_locations()))
			moves = curr_ship.move_to_target_location(closest_shipyard)
			if moves == None:
				ship_action = None
			else:
				ship_action = moves
			if ship_action is not None:
				actions[uid] = ship_action

	
	end = time.time()
	#print(end-start)
	return actions


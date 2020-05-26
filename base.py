import random
from kaggle_environments.envs.halite.halite import get_to_pos
import logging
import numpy as np
import time
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)


BOARD_DIMS = 21
DIRS = ["NORTH", "SOUTH", "EAST", "WEST", None]
DIRS_TO_NUM = {"NORTH":-BOARD_DIMS, "SOUTH":BOARD_DIMS, "EAST":1, "WEST":-1, None:0}

COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"

CONVERT = "CONVERT"
SPAWN = "SPAWN"

#All coordinates will be in (row, column) form (y,x) because that is how the 2d Halite board is laid out

def int_to_coords(num):
	return (int(num//BOARD_DIMS),num%BOARD_DIMS) #(row,column)

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
		self.halite_board_2d = np.array(obs.halite).reshape(self.width,self.height) # [r][c]

	# gets halite locations on the map above a certain value threshold
	# returns a list of 2d coordinates of halite positions satisfying value > threshold
	def get_halite_locations(self, threshold):
		hy, hx = np.where(self.halite_board_2d > threshold)
		halite_coords = list(zip(hy,hx))
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
		closest_yx =  min(distances, key=distances.get)
		return closest_yx

	# returns the position of the closest halite deposit above a certain threshold, blacklist contains a list of locations not to go to.
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
		self.halite = ship[1]
		self.agent_id = agent_id
		self.coords_1d = ship[0]
		self.coords_2d = int_to_coords(self.coords_1d)
		self.role = COLLECT

	def return_random_action(self):
		action = random.choice(DIRS)
		#self.coords_1d += DIRS_TO_NUM[action]
		#self.coords_2d = int_to_coords(self.coords_1d)
		return action

	def move_to_target_location(self, target_yx):
		posistion_yx = self.coords_2d
		if posistion_yx[0] < target_yx[0]:
			return 'SOUTH'
		if posistion_yx[0] > target_yx[0]:
			return 'NORTH'
		if posistion_yx[1] < target_yx[1]:
			return 'EAST'
		if posistion_yx[1] > target_yx[1]:
			 return 'WEST'
		return None

class Yard():
	def __init__(self, yard, agent_id):
		self.agent_id = agent_id
		self.coords_1d = yard
		self.coords_2d = int_to_coords(self.coords_1d)

states = {}

#OBS
#obs.player, obs.step (turn), obs.halite (map), obs.players

#obs.players - array that can be indexed using obs.player. 
#Each element returns halite count, shipyard dictionary, and ship dictionary
#shipyard dictionary matches name to index in obs.halite map
#ship dictionary matches to an array. index in obs.halite map is the first element. Second element is cargo of that ship.
#Key values for the dictionary: "step_created - id" Ex: "2-1" is an object created on step 2 with an ID of 1.

#Halite Map Index maping (r,c) maps to c+r*(total_number_of_columns)
#Guess: total_number_of_columns = 21

def agent(obs):
	global states
	#index = obs.step
	#if(index==0 or index==1):
		#print(obs)
		#print(len(obs.halite))
	start = time.time()
	#print(obs.step)
	actions = {}
	halite, shipyards, ships = obs.players[obs.player]
	#opp_halite, opp_shipyards, opp_ships = obs.players[1]
	board = HaliteBoard(obs)
	if(obs.step==6): #DEBUG 
		print("Step: ", obs.step)
		#print(board.agent_board_1d)
		#print(board.agent_board_2d)
		#print(board.halite_board_1d)
		#print(board.halite_board_2d)
		#print(board.halite_board_2d[2][1])

	for uid, shipyard in shipyards.items():
		if(len(ships) == 0 or halite >= 3000):
			actions[uid] = SPAWN
		curr_yard = Yard(shipyard, uid)
		if(obs.step==6):#DEBUG 
			print("Yard Position:", curr_yard.coords_2d)


	for uid, ship in ships.items():
		if(len(shipyards) == 0 and halite >= 2000):
			actions[uid] = CONVERT
			continue

	for uid, ship_info in ships.items():
		curr_ship = Agent(ship_info, uid)
		if(obs.step==6):#DEBUG 
			print("SHIP Position:", curr_ship.coords_2d)
		#print('STATE: ', curr_ship.agent_id, states[curr_ship.agent_id])
		if(uid not in states):
			states[uid] = COLLECT
			print("COLLECT",uid)

		# collection logic: Move toward halite until storage is > 1000, at which point path to the closest shipyard
		if(states[uid] == COLLECT):
			if(ship_info[1] > 1000):
				states[uid] = DEPOSIT
				print("DEPOSIT ",uid)
			else:
				a = board.get_closest_halite(curr_ship.coords_2d, obs.step+10) #The threshold should increase as time goes on because halite grows over time.
				action = curr_ship.move_to_target_location(a)#curr_ship.return_random_action()#curr_ship.move_to_target_location(a) #Move to the closest Halite Cell
				if(obs.step==25): #Debug step
					print(obs.step)
					print("POSITION: ",curr_ship.coords_2d)
					print("Closest Halite: ",a)
					print("Halite: ", board.halite_board_2d[a[0]][a[1]])
					#print("Surrounding Halite: ",board.halite_board_2d[a[0]-1][a[1]],board.halite_board_2d[a[0]+1][a[1]],board.halite_board_2d[a[0]][a[1]-1],board.halite_board_2d[a[0]][a[1]+1]) #board.halite_board_2d[a[0]-1][a[1]]
				if(action is not None):
					actions[uid] = action
				#else:
					#print("STAY STILL")
					#action = curr_ship.return_random_action()
					#if(action is not None):
						#actions[uid] = action
						#print("RANDOM MOVE")
					#a=board.get_closest_halite(curr_ship.coords_2d, 0.0)
					#print("POSITION: ",curr_ship.coords_2d)
					#print("Closest Halite: ",a)
					#print("Halite: ", board.halite_board_2d[a[0]][a[1]],board.halite_board_2d[a[0]-1][a[1]])
					#print(obs.step)
					#if(obs.step==398):
					#	print(board.halite_board_2d)

		# deposit logic: path naively back to the closest shipyard
		if(states[uid] == DEPOSIT):
			closest_shipyard = board.get_closest_shipyard(curr_ship.coords_2d)
			if(curr_ship.coords_2d == closest_shipyard):
				states[uid] = COLLECT
			#print('DEPOSITING to ', curr_ship.coords_2d, closest_shipyard, len(board.get_shipyard_locations()))
			ship_action = curr_ship.move_to_target_location(closest_shipyard)
			#if moves == None:
			#	ship_action = None
			#else:
			#	ship_action = moves
			if ship_action is not None:
				actions[uid] = ship_action
			else:  
				states[uid] = COLLECT #Once deposited, go back and collect
	
	end = time.time()
	#print(end-start)
	return actions


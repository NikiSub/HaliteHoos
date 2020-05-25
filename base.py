import random
from kaggle_environments.envs.halite.halite import get_to_pos
import logging
import numpy as np
import time
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)


BOARD_DIMS = 21
starting_halite = 5000
DIRS = ["NORTH", "SOUTH", "EAST", "WEST", None]
DIRS_TO_NUM = {"NORTH":-BOARD_DIMS, "SOUTH":BOARD_DIMS, "EAST":1, "WEST":-1, None:0}

COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"

CONVERT = "CONVERT"
SPAWN = "SPAWN"

#All coordinates will be in (row, column) form (y,x) because that is how the 2d Halite board is laid out

def int_to_coords(num):
	return (int(num//BOARD_DIMS),num%BOARD_DIMS) #(row,column)

def same_pos_2d(p1,p2):
	if(p1[0]==p2[0] and p1[1]==p2[1]):
		return True
	return False

class HaliteBoard():
	def __init__(self, obs):
		self.height = self.width = BOARD_DIMS

		_, self.shipyards, self.ships = obs.players[obs.player]

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
	#checks for collision. returns the ships next position if this is a valid action. returns None otherwise. TODO: improve to not run into enemies
	def valid_action(self,curr_pos,action,next_locations):
		if(action=="SOUTH" and curr_pos[0]<(BOARD_DIMS-1)):
			next_pos=(curr_pos[0]+1,curr_pos[1])
		elif(action=="NORTH" and curr_pos[0]>0):
			next_pos=(curr_pos[0]-1,curr_pos[1])
		elif(action=="EAST" and curr_pos[1]<(BOARD_DIMS-1)):
			next_pos=(curr_pos[0],curr_pos[1]+1)
		elif(action=="WEST" and curr_pos[1]>0):
			next_pos=(curr_pos[0],curr_pos[1]-1)
		for i in next_locations:
			if(next_pos[0]==i[0] and next_pos[1]==next_pos[1]):
				return None
		return next_pos
	def valid_stay(self,curr_pos,next_locations):
		stay = True
		valid = [1,1,1,1] #South, North, East, West
		if(curr_pos[0]<(BOARD_DIMS-1)):
			south_pos = (curr_pos[0]+1,curr_pos[1])
		else:
			valid[0] = 0

		if(curr_pos[0]>0):
			north_pos = (curr_pos[0]-1,curr_pos[1])
		else:
			valid[1]=0

		if(curr_pos[1]<(BOARD_DIMS-1)):
			east_pos = (curr_pos[0],curr_pos[1]+1)
		else:
			valid[2]=0

		if(curr_pos[1]>0):
			west_pos = (curr_pos[0],curr_pos[1]-1)
		else:
			valid[3] = 0

		for i in next_locations:
			if(stay and same_pos_2d(curr_pos,i)):
				stay = False
			if(valid[0]==1 and same_pos_2d(south_pos,i)):
				valid[0] = 0
			if(valid[1]==1 and same_pos_2d(north_pos,i)):
				valid[1] = 0
			if(valid[2]==1 and same_pos_2d(east_pos,i)):
				valid[2] = 0
			if(valid[3]==1 and same_pos_2d(west_pos,i)):
				valid[3] = 0
		if(stay):
			return (None,None)
		else:
			if(valid[0]==1):
				return ('SOUTH',south_pos)
			elif(valid[1]==1):
				return ('NORTH',north_pos)
			elif(valid[2]==1):
				return ('EAST',east_pos)
			elif(valid[3]==1):
				return ('WEST',west_pos)
			else:
				return (None,None)




	def get_closest_halite_blacklist(self, curr_pos, threshold,blacklist):
		#print(blacklist)
		halite_coords = self.get_halite_locations(threshold)
		distances = {}
		in_blacklist_count = 0
		for i in halite_coords:
			in_blacklist = False
			for j in blacklist:
				#print(j)
				if(i[0]==j[0] and i[1]==j[1]):
					in_blacklist = True
					in_blacklist_count+=1
			if(in_blacklist):
				distances[i] = (BOARD_DIMS**2)+BOARD_DIMS #Essentially Infinity
			else:
				# find euclidean distance, doesn't take into account wrap around
				dist = np.sqrt((i[0] - curr_pos[0])**2 + (i[1] - curr_pos[1])**2)
				distances[i] = dist
		# from the dict get the closest set of coords     
		closest_xy =  min(distances, key=distances.get)

		if(in_blacklist_count==len(halite_coords)):
			return None
		else:
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
lastHaliteSpawn = starting_halite
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
	global states,lastHaliteSpawn
	#index = obs.step
	#if(index==0 or index==1):
		#print(obs)
		#print(len(obs.halite))
	start = time.time()
	#print(obs.step)
	actions = {}
	destinations = {}

	halite, shipyards, ships = obs.players[obs.player]
	#opp_halite, opp_shipyards, opp_ships = obs.players[1]
	board = HaliteBoard(obs)
	next_locations = []
	if(obs.step==6): #DEBUG 
		print("Step: ", obs.step)
		#print(board.agent_board_1d)
		#print(board.agent_board_2d)
		#print(board.halite_board_1d)
		#print(board.halite_board_2d)
		#print(board.halite_board_2d[2][1])
	for uid, ship in ships.items():
		if(len(shipyards) == 0):
			states[uid] = CONVERT
			actions[uid] = CONVERT
			continue
	
	for uid, ship_info in ships.items():
		curr_ship = Agent(ship_info, uid)
		if(obs.step==6):#DEBUG 
			print("SHIP Position:", curr_ship.coords_2d)
		#print('STATE: ', curr_ship.agent_id, states[curr_ship.agent_id])
		if(uid not in states):
			states[uid] = COLLECT
			print(obs.step," COLLECT",uid)

		# collection logic: Move toward halite until storage is > 1000, at which point path to the closest shipyard
		if(states[uid] == COLLECT):
			#print(obs.step,int_to_coords(ship_info[0]),ship_info[1])
			if(ship_info[1] > 1000):
				states[uid] = DEPOSIT
				print(obs.step," DEPOSIT ",uid)
			else:

				a = board.get_closest_halite_blacklist(curr_ship.coords_2d, obs.step+10,destinations.values()) #The threshold should increase as time goes on because halite grows over time.
				if(a is not None):
					destinations[uid] = a
					action = curr_ship.move_to_target_location(a)#curr_ship.return_random_action()#curr_ship.move_to_target_location(a) #Move to the closest Halite Cell
					#if(obs.step==25): #Debug step
						#print(obs.step)
						#print("POSITION: ",curr_ship.coords_2d)
						#print("Closest Halite: ",a)
						#print("Halite: ", board.halite_board_2d[a[0]][a[1]])
						#print("Surrounding Halite: ",board.halite_board_2d[a[0]-1][a[1]],board.halite_board_2d[a[0]+1][a[1]],board.halite_board_2d[a[0]][a[1]-1],board.halite_board_2d[a[0]][a[1]+1]) #board.halite_board_2d[a[0]-1][a[1]]
				else:
					action = curr_ship.return_random_action()

				if(action is not None):
					curr_ship_next_loc = board.valid_action(curr_ship.coords_2d,action,next_locations)
					if(curr_ship_next_loc is not None):
						actions[uid] = action
						next_locations.append(curr_ship_next_loc)
					else:
						alt_act, alt_next_loc = board.valid_stay(curr_ship.coords_2d,next_locations) #Check if it is okay to stay in the same place
						print("collector ",obs.step)
						print("POSITION: ",curr_ship.coords_2d)
						print(alt_act, alt_next_loc)
						if(alt_act is None):
							next_locations.append(curr_ship.coords_2d)
						else: #Move somewhere else because staying is not safe
							actions[uid] = alt_act
							next_locations.append(alt_next_loc)
					# check if this action is valid, if it is add it to the next_location list
					#if not, make this ship stay still
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
			#print('DEPOSITING to ', curr_ship.coords_2d, closest_shipyard, len(board.get_shipyard_locations()))
			ship_action = curr_ship.move_to_target_location(closest_shipyard)
			#if moves == None:
			#	ship_action = None
			#else:
			#	ship_action = moves
			if ship_action is not None:
				curr_ship_next_loc = board.valid_action(curr_ship.coords_2d,ship_action,next_locations)
				if(curr_ship_next_loc is not None):
					actions[uid] = ship_action
					next_locations.append(curr_ship_next_loc)
					#if(curr_ship_next_loc[0]==closest_shipyard[0] and curr_ship_next_loc[1]==closest_shipyard[1]):
					#	states[uid] = COLLECT #Once deposited, go back and collect
				else:
					alt_act, alt_next_loc = board.valid_stay(curr_ship.coords_2d,next_locations)
					print("depositor ",obs.step)
					print("POSITION: ",curr_ship.coords_2d)
					print(alt_act, alt_next_loc)
					if(alt_act is None):
						next_locations.append(curr_ship.coords_2d)
					else:
						actions[uid] = alt_act
						next_locations.append(alt_next_loc)
					#if(curr_ship.coords_2d[0]==closest_shipyard[0] and curr_ship.coords_2d[1]==closest_shipyard[1]):
					#	states[uid] = COLLECT #Once deposited, go back and collect
			else:
				states[uid] = COLLECT #Once deposited, go back and collect	
	for uid, shipyard in shipyards.items():
		curr_yard = Yard(shipyard, uid)
		if(len(ships) == 0):
			actions[uid] = SPAWN
		if(halite-lastHaliteSpawn>=1000):
			spawn = True
			for n in next_locations:
				if(same_pos_2d(n,curr_yard.coords_2d)):
					spawn = False
			if(spawn): #If there is room to spawn a new ship
				actions[uid] = SPAWN
				lastHaliteSpawn = halite
				next_locations.append(curr_yard.coords_2d)
		if(obs.step==6):#DEBUG 
			print("Yard Position:", curr_yard.coords_2d)
	end = time.time()
	#print(end-start)
	return actions


import random
from kaggle_environments.envs.halite.halite import get_to_pos
import logging
import numpy as np
import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)


BOARD_DIMS = 21
BOARD_HALF = BOARD_DIMS//2
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

def manhattan_distance(p1, p2):
	return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])

class MapAnalysis():
	def __init__(self, board):
		self.board_2d = board
		row = np.concatenate((board,board,board), axis=0)
		self.board_2d_concat = np.concatenate((row,row,row), axis=1)
		self.board_2d_wrap = self.board_2d_concat[BOARD_DIMS-1:(2*BOARD_DIMS)+1,BOARD_DIMS-1:(2*BOARD_DIMS)+1]

	def get(self):
		return self.board_2d, self.board_2d_concat, self.board_2d_wrap
	
	def gauss_blur_thresh(self, sigma, threshold):
		g = scipy.ndimage.filters.gaussian_filter(self.board_2d_wrap,1.0)
		thresh = (g>threshold)*1
		return thresh, thresh[1:1+BOARD_DIMS,1:1+BOARD_DIMS]

	def smart_gauss_blur_thresh(self, sigma, miners, step):
		g = scipy.ndimage.filters.gaussian_filter(self.board_2d_wrap,1.0)
		threshold = min(miners*0.1*step+200,350)
		thresh = (g>threshold)*1
		return g, thresh[1:1+BOARD_DIMS,1:1+BOARD_DIMS]
	def create_cluster(self, cluster, q, threshold, count, sum_halite):
		if(len(q) > 0):
			#print("A")
			a = q.pop(0)
			i = a[0]
			j = a[1]
			#print(i,j)
			i = (i%BOARD_DIMS)
			j = (j%BOARD_DIMS)
			#print(i,j)
			if(sum_halite < threshold):
				if(cluster[i][j]==0):
					cluster[i][j]=count
					#print("BOARD: ",self.board_2d[i][j])
					sum_halite += self.board_2d[i][j]
					#print("SUM: ",sum_halite)
					q.append((i+1,j))
					q.append((i-1,j))
					q.append((i,j+1))
					q.append((i,j-1))
			s = self.create_cluster(cluster, q, threshold, count, sum_halite)
			return s
		else:
			#print("A")
			return sum_halite
		#print(sum_halite)

	def halite_cluster(self, group_threshold):
		cluster = np.zeros(np.shape(self.board_2d))
		count = 1
		for i in range(0, BOARD_DIMS):
			for j in range(0, BOARD_DIMS):
				if(cluster[i][j]==0):
					q = [(i,j)]
					s = self.create_cluster(cluster, q, group_threshold, count, 0.0)
					print(f'Count: {count}, Sum: {s}, Center: {i},{j}')
					count+=1
		#print(cluster)
		#print(self.board_2d)
		return cluster



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
				print("NO OPTION") #If this is printed, a crash is likely to occur
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
	def checkStay(self,board,next_locations,actions,uid):
		alt_act, alt_next_loc = board.valid_stay(self.coords_2d,next_locations) #Check if it is okay to stay in the same place
		if(alt_act is None):
			#print("Staying Still at ",self.coords_2d)
			next_locations.append(self.coords_2d)
		else:
			#print("Fleeing to ", alt_next_loc)
			actions[uid] = alt_act
			next_locations.append(alt_next_loc)
	def checkAction(self,action,board,next_locations,actions,uid,**kwargs):
		if(action is not None):
			curr_ship_next_loc = board.valid_action(self.coords_2d,action,next_locations)
			if(curr_ship_next_loc is not None):
				actions[uid] = action
				next_locations.append(curr_ship_next_loc)
				#moveTarget = kwargs.get('moveTarget', None)
				#if(moveTarget is not None):
				#	if(moveTarget):
				#		print("Moving toward ",curr_ship_next_loc)
				#	else:
				#		print("Moving randomly to ",curr_ship_next_loc)
			else:
				self.checkStay(board,next_locations,actions,uid)
			return True
		else:
			self.checkStay(board,next_locations,actions,uid)
			return False #To tell depositors to change into collectors
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
	#start = time.time()
	actions = {}
	#destinations = {}
	#print(obs.step)
	halite, shipyards, ships = obs.players[obs.player]
	opp_halite, opp_shipyards, opp_ships = obs.players[1]
	board = HaliteBoard(obs)
	mapA = MapAnalysis(board.halite_board_2d)

	b, b_concat, b_wrap = mapA.get()
	#thresh_wrap,thresh = mapA.smart_gauss_blur_thresh(1.0, 4, obs.step)
	dominance = np.full_like(b, 0)
	enemy_radius = 2
	for uid, ship_info in opp_ships.items():
		curr_ship = Agent(ship_info, uid)
		for y in range(-enemy_radius,enemy_radius+1):
			for x in range(-enemy_radius,enemy_radius+1):
				newY = curr_ship.coords_2d[0]+y
				newX = curr_ship.coords_2d[1]+x
				if(newY>0 and newY<BOARD_DIMS and newX>0 and newX<BOARD_DIMS):
					if(manhattan_distance(curr_ship.coords_2d,(newY,newX))<=3):
						dominance[newY][newX] = 1
	if(obs.step==10):
		c = mapA.halite_cluster(1000)
		#print(np.shape(mapA.board_2d))
	else:
		c = np.zeros(np.shape(mapA.board_2d))
	#	print(obs.step)
	#	print(board.halite_board_2d)
	#print("B")
	next_locations = []
	shipsSorted = []
	uidSorted = []


	end = time.time()
	return actions,mapA.board_2d,c


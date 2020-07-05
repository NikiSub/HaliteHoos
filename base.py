import random
from kaggle_environments.envs.halite.halite import get_to_pos
#from map_analysis import MapAnalysis
import logging
import numpy as np
import time
import math
import scipy.ndimage
logging.basicConfig(filename='stdout.log',level=logging.DEBUG)

# TODO: Parameters to change: CLUSTER_SUM GOAL, CLUSTER_NUMBER_GOAL, formula for choosingCluster, how clusters are formed, number of bots to have 
BOARD_DIMS = 21
BOARD_HALF = BOARD_DIMS//2
starting_halite = 5000
DIRS = ["NORTH", "SOUTH", "EAST", "WEST", None]
DIRS_TO_NUM = {"NORTH":-BOARD_DIMS, "SOUTH":BOARD_DIMS, "EAST":1, "WEST":-1, None:0}
CLUSTER_SUM_GOAL = 1000
CLUSTER_NUMBER_GOAL = 25
PROTECT_NUM = 2 #Number of miners each defender protects.
THREAT_NEAR = 4
THREAT_AWAY = 7

#ship states
COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"
DEFEND = "DEFEND"

#ship states with an enemy nearby
COLLECT_THREAT = "COLLECT_T"
DEPOSIT_THREAT = "DEPOSIT_T"
DEFEND_THREAT = "DEFEND_T"

CONVERT = "CONVERT"
SPAWN = "SPAWN"

#All coordinates will be in (row, column) form (y,x) because that is how the 2d Halite board is laid out

#Current best agent
def is_threat_mode(state): #returns true if the state is a threat
	if state == COLLECT_THREAT or state == DEPOSIT_THREAT or state == DEFEND_THREAT:
		return True
	return False
def toggle_threat_mode(state):
	if state == COLLECT:
		return COLLECT_THREAT
	elif state == COLLECT_THREAT:
		return COLLECT
	elif state == DEPOSIT:
		return DEPOSIT_THREAT
	elif state == DEPOSIT_THREAT:
		return DEPOSIT
	elif state == DEFEND:
		return DEFEND_THREAT
	elif state == DEFEND_THREAT:
		return DEFEND
	return None
def int_to_coords(num):
	return (int(num//BOARD_DIMS),num%BOARD_DIMS) #(row,column)


def same_pos_2d(p1,p2):
	if(p1[0]==p2[0] and p1[1]==p2[1]):
		return True
	return False

def manhattan_distance(p1, p2):
	row_difference = min(abs(p1[0]-p2[0]),abs(p1[0]-p2[0]-BOARD_DIMS),abs(p1[0]-p2[0]+BOARD_DIMS))
	col_difference = min(abs(p1[1]-p2[1]),abs(p1[1]-p2[1]-BOARD_DIMS),abs(p1[1]-p2[1]+BOARD_DIMS))
	return row_difference + col_difference

def chooseCluster(mapA, destination_cluster, location_2d, step): #choose cluster to mine based on distance, sum, and cell size TODO: incorporate enemies into logic
	cluster_num = len(mapA.cluster_centers)
	max_score = 0
	max_cluster_id = 0
	for cluster_id, v in mapA.cluster_centers.items():
		halite_sum = v[0]
		cluster_center_pos = v[1]
		cell_count = v[2]
		dist = manhattan_distance(location_2d, cluster_center_pos)
		max_dist = BOARD_DIMS-1
		average_cluster_size = (BOARD_DIMS**2)//cluster_num
		#Maximize score want short distance, large sum, and large cell_count (In the future, get big clusters)
		#With current parameters Max score is about: (20) + (20+) + (~15)
		if step<100: 
			score = (max_dist-dist)  # get closest cluster 
		elif step<200:
			score = (max_dist-dist)*5+(halite_sum/CLUSTER_NUMBER_GOAL)*40+(cell_count*2)
		elif step<300:
			score = (max_dist-dist)*2+(halite_sum/CLUSTER_NUMBER_GOAL)*60+(cell_count*2) # start focusing more on sums
		else:
			score = (max_dist-dist)+(halite_sum/CLUSTER_NUMBER_GOAL)*40+(cell_count*2)
		if score >= max_score and cluster_id not in destination_cluster.values():  
			max_score = score
			max_cluster_id = cluster_id
	return max_cluster_id

class MapAnalysis():
	def __init__(self, board):
		self.board_2d = board
		row = np.concatenate((board,board,board), axis=0)
		self.board_2d_concat = np.concatenate((row,row,row), axis=1)
		self.board_2d_wrap = self.board_2d_concat[BOARD_DIMS-1:(2*BOARD_DIMS)+1,BOARD_DIMS-1:(2*BOARD_DIMS)+1]
		self.cluster = np.zeros(np.shape(self.board_2d))
		self.cluster_centers = {}
		self.halite_regen_rate = 1.02
		self.halite_mining_rate = 0.25

	def get(self):
		return self.board_2d, self.board_2d_concat, self.board_2d_wrap
	def get_cluster(self):
		return self.cluster

	def get_cluster_center(self):
		return self.cluster_centers

	def set_board(self, board):
		self.board_2d = board
		row = np.concatenate((board,board,board), axis=0)
		self.board_2d_concat = np.concatenate((row,row,row), axis=1)
		self.board_2d_wrap = self.board_2d_concat[BOARD_DIMS-1:(2*BOARD_DIMS)+1,BOARD_DIMS-1:(2*BOARD_DIMS)+1]

	def sorted_map(self,index_num):
		a = np.argsort(self.board_2d, axis=None)
		loc = []
		for i in range(0,min(index_num,BOARD_DIMS**2)):
			b = a[len(a)-1-i]
			loc.append(int_to_coords(b))
		return loc

	def gauss_blur_thresh(self, sigma, threshold):
		g = scipy.ndimage.filters.gaussian_filter(self.board_2d_wrap,1.0)
		thresh = (g>threshold)*1
		return thresh, thresh[1:1+BOARD_DIMS,1:1+BOARD_DIMS]

	def smart_gauss_blur_thresh(self, sigma, miners, step):
		g = scipy.ndimage.filters.gaussian_filter(self.board_2d_wrap,1.0)
		threshold = min(miners*0.1*step+200,350)
		thresh = (g>threshold)*1
		return g, thresh[1:1+BOARD_DIMS,1:1+BOARD_DIMS]
	#def chooseCluster(self, ship_coords_2d, destination_cluster):
	# choose a cluster that is close to the location, far from enemies, and not too large in cell-size but large halite sum

	def optimal_halite_location(self, cluster_id, ship_coords_2d, location_blacklist): #given a ship's location and the cluster it is trying to go to, get the optimal cell that maximizes halite per step
		cell_loc = np.transpose(np.nonzero(self.cluster==cluster_id))
		#print("cluster ID: ", cluster_id)
		#print(cell_loc)
		if(len(cell_loc)>0):
			optimal_location = (cell_loc[0][0],cell_loc[0][1])
			optimal_halite_per_step = 0
			for loc in cell_loc:
				valid_loc = True
				for loc_2 in location_blacklist:
					if(same_pos_2d((loc[0],loc[1]),loc_2)):
						valid_loc=False
				if valid_loc:
					halite = self.board_2d[loc[0]][loc[1]]
					dist = manhattan_distance(ship_coords_2d,(loc[0],loc[1]))
					halite_per_step = math.floor(self.halite_mining_rate*(halite*(self.halite_regen_rate**dist)))/(1.0+dist)
					if halite_per_step > optimal_halite_per_step:
						optimal_halite_per_step = halite_per_step
						optimal_location = (loc[0],loc[1])
			return optimal_location
		else:
			return None
	def create_cluster(self, q, threshold, cluster_id, sum_halite, cell_count):
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
				if(self.cluster[i][j]==0):
					self.cluster[i][j]=cluster_id
					#print("BOARD: ",self.board_2d[i][j])
					sum_halite += self.board_2d[i][j]
					cell_count += 1
					#print("SUM: ",sum_halite)
					q.append((i+1,j))
					q.append((i-1,j))
					q.append((i,j+1))
					q.append((i,j-1))
			s, c = self.create_cluster(q, threshold, cluster_id, sum_halite, cell_count)
			return s, c
		else:
			#print("A")
			return sum_halite, cell_count
		#print(sum_halite)

	def halite_cluster_update(self,group_threshold, cluster_id):
		for i in range(0, BOARD_DIMS):
			for j in range(0, BOARD_DIMS):
				if(self.cluster[i][j]==0):
					q = [(i,j)]
					s, cell_count = self.create_cluster(q, group_threshold, cluster_id, 0.0, 0)
					#print(f'cluster_id: {cluster_id}, Sum: {s}, Center: {i},{j}')
					self.cluster_centers[cluster_id] = (s, (i,j), cell_count)
					cluster_id+=1
		#print(cluster)
		#print(self.board_2d)
		return self.cluster

	def halite_cluster_specific(self, group_threshold, centers): #centers is a list of location tuples (y,x)
		self.cluster_centers = {}
		self.cluster = np.zeros(np.shape(self.board_2d))
		cluster_id = 1
		for c in centers:
			q = [c]
			s, cell_count = self.create_cluster(q, group_threshold, cluster_id, 0.0, 0)
			self.cluster_centers[cluster_id] = (s, c, cell_count)
			cluster_id += 1
		self.halite_cluster_update(group_threshold, cluster_id)
		return self.cluster

	def halite_cluster_max(self, group_threshold, index_num):
		loc = self.sorted_map(index_num)
		self.halite_cluster_specific(group_threshold, loc)
		return self.cluster

	def halite_cluster(self, group_threshold):
		self.cluster_centers = {}
		self.cluster = np.zeros(np.shape(self.board_2d))
		cluster_id = 1
		self.halite_cluster_update(group_threshold, cluster_id)
		return self.cluster

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
		try:
			closest_yx =  min(distances, key=distances.get)
			return closest_yx
		except:
			return None
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
		else:
			next_pos = curr_pos
		#print("planned next location: ", next_pos)
		for i in next_locations:
			if(next_pos[0]==i[0] and next_pos[1]==i[1]): #NOTE: THis used to be next_pos[1]==next_pos[1] <-- fix in master
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
				#print("NO OPTION") #If this is printed, a crash is likely to occur
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
	def move_to_target_wrapping_helper(self, position_yx, target_yx, index, pos_move, neg_move): #method that works with map wrapping
		if position_yx[index] != target_yx[index]:
			position_forward = position_yx[index]+BOARD_DIMS  # This will be greater than BOARD_DIMS
			position_backward = position_yx[index]-BOARD_DIMS  # This will be less than 0
			normal_distance = abs(position_yx[index]-target_yx[index])
			forward_distance = abs(position_forward-target_yx[index])
			backward_distance = abs(position_backward-target_yx[index]) #Since the map wraps, going to position_yx[index] is the same as going to the other 2 posiitons
			closest_position = position_yx[index]

			if normal_distance <= forward_distance and normal_distance <= backward_distance: # Use the shortest distance
				closest_position = position_yx[index]
			elif forward_distance <= normal_distance and forward_distance <= backward_distance:
				closest_position = position_forward
			elif backward_distance <= normal_distance and backward_distance <= forward_distance:
				closest_position = position_backward
			
			if closest_position < target_yx[index]: #based on the closest_position
				return pos_move
			else:
				return neg_move
		else:
			return None
	def move_to_target_location(self, target_yx):  #TODO fix to work with wrapping
		posistion_yx = self.coords_2d
		if(random.random()<0.5): #Move vertically first
			command = self.move_to_target_wrapping_helper(posistion_yx, target_yx, 0, 'SOUTH', 'NORTH')
			if command is None:
				return self.move_to_target_wrapping_helper(posistion_yx, target_yx, 1, 'EAST', 'WEST')
			return command
		else: #Move horizontally first
			command = self.move_to_target_wrapping_helper(posistion_yx, target_yx, 1, 'EAST', 'WEST')
			if command is None:
				return self.move_to_target_wrapping_helper(posistion_yx, target_yx, 0, 'SOUTH', 'NORTH')
			return command
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
			#print("planned action: ",action)
			curr_ship_next_loc = board.valid_action(self.coords_2d,action,next_locations)
			if(curr_ship_next_loc is not None):
				actions[uid] = action
				next_locations.append(curr_ship_next_loc)
				moveTarget = kwargs.get('moveTarget', None)
				#if(moveTarget is not None):
				#	if(moveTarget):
				#		print("Moving toward ",curr_ship_next_loc)
				#	else:
				#		print("Moving randomly to ",curr_ship_next_loc)
			else:
				#print("another ship will be where I want to go")
				self.checkStay(board,next_locations,actions,uid)
			return True
		else:
			#print("None action given to checkAction")
			self.checkStay(board,next_locations,actions,uid)
			return False #To tell depositors to change into collectors
class Yard():
	def __init__(self, yard, agent_id):
		self.agent_id = agent_id
		self.coords_1d = yard
		self.coords_2d = int_to_coords(self.coords_1d)

states = {}
collection_states = {} #Local or global harvesting, currently not used
destination_cluster = {} #which cluster a ship goes to
lastHaliteSpawn = starting_halite
mapA = 0
defender_to_miner_map = {}
miner_to_defender_map = {}
conflict = False
#OBS
#obs.player, obs.step (turn) 0-398, obs.halite (map), obs.players

#obs.players - array that can be indexed using obs.player. 
#Each element returns halite count, shipyard dictionary, and ship dictionary
#shipyard dictionary matches name to index in obs.halite map
#ship dictionary matches to an array. index in obs.halite map is the first element. Second element is cargo of that ship.
#Key values for the dictionary: "step_created - id" Ex: "2-1" is an object created on step 2 with an ID of 1.

#Halite Map Index maping (r,c) maps to c+r*(total_number_of_columns)
#Guess: total_number_of_columns = 21
def agent(obs):
	global states,lastHaliteSpawn, collection_states, mapA, destination_cluster, CLUSTER_SUM_GOAL, CLUSTER_NUMBER_GOAL, defender_to_miner_map, miner_to_defender_map
	start = time.time()
	actions = {}
	destinations = {}
	print("STEP: ", obs.step)
	halite, shipyards, ships = obs.players[obs.player]
	opp_halite_list = []
	opp_shipyards_list = []
	opp_ships_list = []
	for i in range(0, len(obs.players)):
		if i != obs.player:
			opp_halite, opp_shipyards, opp_ships = obs.players[1]
			opp_halite_list.append(opp_halite)
			opp_shipyards_list.append(opp_shipyards)
			opp_ships_list.append(opp_ships)
	board = HaliteBoard(obs)
	next_locations = []
	shipsSorted = []
	uidSorted = []
	yard_location = (0,0)
	newYard = False
	newYard_loc = (0,0)
	#CLUSTER_SUM_GOAL = 1000
	#CLUSTER_NUMBER_GOAL = 25
	if(obs.step%100==0):
		mapA = MapAnalysis(board.halite_board_2d)
		if(obs.step==0):
			CLUSTER_SUM_GOAL = 300
			CLUSTER_NUMBER_GOAL = 8
		elif(obs.step==100):
			CLUSTER_SUM_GOAL = 500
			CLUSTER_NUMBER_GOAL = 12
		elif(obs.step==200):
			CLUSTER_SUM_GOAL = 750
			CLUSTER_NUMBER_GOAL = 20
		elif(obs.step==300):
			CLUSTER_SUM_GOAL = 1000
			CLUSTER_NUMBER_GOAL = 25
		elif(obs.step==400):
			CLUSTER_SUM_GOAL = 1000
			CLUSTER_NUMBER_GOAL = 25
		mapA.halite_cluster_max(CLUSTER_SUM_GOAL,CLUSTER_NUMBER_GOAL)
		destination_cluster = {}
	else:
		mapA.set_board(board.halite_board_2d)
	deposit_count = 0
	collector_count = 0
	defender_count = 0
	opponent_distance = {}   # (Key: Friendly Ship ID) (Value: Opponent ID, Opponent Distance, number of enemies within THREAT_NEAR)
	for uid, ship in ships.items():  # look at DEFEND ships first and get enemy distances
		curr_ship = Agent(ship, uid)
		for opp_ships in opp_ships_list:
			for opp_uid, opp_ship_info in opp_ships.items(): #fill out dictionary opponent_distance
				opp_ship = Agent(opp_ship_info, opp_uid)
				if uid in opponent_distance.keys():
					dist = manhattan_distance(curr_ship.coords_2d, opp_ship.coords_2d)
					count = 0
					if dist <= THREAT_NEAR:
						count+=1
					opponent_distance[uid] = (opp_uid, dist, count)
					if uid in states:
						if is_threat_mode(states[uid]) and dist >= THREAT_AWAY:  # Ship is currently in threat mode and threat has left
							states[uid] = toggle_threat_mode(states[uid])
						elif not(is_threat_mode(states[uid])) and dist <= THREAT_NEAR:  # Ship is currently not in threat mode and threat has arrived
							states[uid] = toggle_threat_mode(states[uid])
				else:
					dist = opponent_distance[uid][1]
					newDist = manhattan_distance(curr_ship.coords_2d, opp_ship.coords_2d)
					if newDist <= THREAT_NEAR:
						opponent_distance[uid][2]+=1
					if newDist < dist:
						opponent_distance[uid][0] = opp_uid
						opponent_distance[uid][1] = newDist
		if (uid in states) and (states[uid] == DEFEND or states[uid]== DEFEND_THREAT): 
			#Check if corresponding miners are ok
			if uid in defender_to_miner_map:
				miner_list = defender_to_miner_map[uid]
				threat = False
				for miner_id in miner_list:
					if miner_id in states and is_threat_mode(states[miner_id]):
						threat = True
				if threat:
					states[uid] = DEFEND_THREAT
				else:
					states[uid] = DEFEND
			shipsSorted.append(ship)
			uidSorted.append(uid)
			defender_count+=1
	for uid, ship in ships.items(): #look at DEPOSIT ships second TODO: Modify these for loops to be a regular sorting algorithm instead
		if (uid in states) and (states[uid] == DEPOSIT or states[uid] == DEPOSIT_THREAT):
			shipsSorted.append(ship)
			uidSorted.append(uid)
			deposit_count+=1
	for uid, ship in ships.items(): #look at other ships
		if  (uid in states) and (states[uid] == COLLECT or states[uid] == COLLECT_THREAT):
			shipsSorted.append(ship)
			uidSorted.append(uid)
			collector_count+=1
		elif uid not in states:
			shipsSorted.append(ship)
			uidSorted.append(uid)
	print("Number of miners: ",len(ships))
	shipCount = 0
	#for uid, ship_info in ships.items():
	destroyed_ships = []
	for ship_id in destination_cluster.keys(): # Deletes destroyed ships from dictionary
		if ship_id not in uidSorted:
			destroyed_ships.append(ship_id)
	for d in destroyed_ships:
		destination_cluster.pop(d)
	for k in range(0,len(ships)):
		uid = uidSorted[k]
		ship_info = shipsSorted[k]
		#print("Info for Ship ",shipCount, "ID: ", uid)
		shipCount+=1
		curr_ship = Agent(ship_info, uid)
		#print("Ship location: ",curr_ship.coords_2d)
		if(len(shipyards) == 0 and halite >= 500 and newYard==False) or (obs.step>=398 and curr_ship.halite>=500): #TODO: Fix this so only 1 yard spawns
			states[uid] = CONVERT
			actions[uid] = CONVERT
			newYard=True
			newYard_loc = curr_ship.coords_2d
			#print("Converting ship")
		if(uid not in states):
			if defender_count<((collector_count + deposit_count)//2):
				states[uid] = DEFEND
				defender_to_miner_map[uid] = []
				for potential_id in states.keys():
					if potential_id not in miner_to_defender_map:
						miner_to_defender_map[potential_id] = uid
						defender_to_miner_map[uid].append(potential_id)
					if len(defender_to_miner_map[uid]) >= PROTECT_NUM:
						break
			else:
				states[uid] = COLLECT
			#print("becoming a collector")
		if states[uid] == DEFEND:
			print("Defender Ship location: ",curr_ship.coords_2d)
			print("Defender Halite: ",curr_ship.halite)
			for s_id in defender_to_miner_map.keys(): #Delete any destroyed ships
				alive_miners = []
				for miner_id in defender_to_miner_map[s_id]:
					if miner_id in uidSorted:
						alive_miners.append(miner_id)
				defender_to_miner_map[s_id] = alive_miners
			
			if len(defender_to_miner_map[uid]) < PROTECT_NUM:
				for potential_id in states.keys():
					if potential_id not in miner_to_defender_map:
						miner_to_defender_map[potential_id] = uid
						defender_to_miner_map[uid].append(potential_id)
					if len(defender_to_miner_map[uid]) >= PROTECT_NUM:
						break
			print("Protecting: ", defender_to_miner_map[uid])
			action = curr_ship.return_random_action()
			curr_ship.checkAction(action,board,next_locations,actions,uid)
		# collection logic: Move toward halite until storage is > 1000, at which point path to the closest shipyard
		if(states[uid] == COLLECT):
			#print(obs.step,int_to_coords(ship_info[0]),ship_info[1])
			#print("COLLECT")
			if(ship_info[1] >= CLUSTER_SUM_GOAL):
				states[uid] = DEPOSIT
				#print("STEP:", obs.step)
				#print("DELETEING from dest_cluster: ", uid, destination_cluster[uid],  mapA.cluster_centers[destination_cluster[uid]][1])
				if uid in destination_cluster:
					destination_cluster.pop(uid)

				#print("Becoming a depositor")
				# For now, stay still when transitioning states: TODO move back toward shipyard instead
				curr_ship.checkStay(board,next_locations,actions,uid)
			else:
				if(uid not in destination_cluster): #Add to dictionary, set collection_state as going to cluster
					#print("STEP:", obs.step)
					cluster_id = chooseCluster(mapA, destination_cluster, curr_ship.coords_2d, obs.step)
					#if cluster_id==0:
						#print("ALL CLUSTERS ARE TAKEN",obs.step, uid)
					destination_cluster[uid] = cluster_id
					#print("ADDING to dest_cluster: ",uid, cluster_id)
				#print("Cluster Center: ",mapA.cluster_centers[destination_cluster[uid]][1])	
				#Go to cluster, or explore cluster
				#Check if in cluster, if not, go toward center. else: explore
				#explore: look
				moveTarget = True
				#a = board.get_closest_halite_blacklist(curr_ship.coords_2d, obs.step+10,destinations.values()) #The threshold should increase as time goes on because halite grows over time.
				a = mapA.optimal_halite_location(destination_cluster[uid], curr_ship.coords_2d, [yard_location])
				#print("optimal halite location: ", a)
				if(a is not None):
					#destinations[uid] = a
					action = curr_ship.move_to_target_location(a)#curr_ship.return_random_action()#curr_ship.move_to_target_location(a) #Move to the closest Halite Cell
				else:
					#print("OPTIMAL halite is None")
					action = curr_ship.return_random_action()
					moveTarget = False

				curr_ship.checkAction(action,board,next_locations,actions,uid,moveTarget=moveTarget)


		# deposit logic: path naively back to the closest shipyard
		if(states[uid] == DEPOSIT):
			#print("DEPOSIT")
			closest_shipyard = board.get_closest_shipyard(curr_ship.coords_2d)
			if closest_shipyard is None and newYard==False:
				newYard=True
				states[uid] = CONVERT
				actions[uid] = CONVERT
				newYard_loc = curr_ship.coords_2d
			#print('DEPOSITING to ', curr_ship.coords_2d, closest_shipyard, len(board.get_shipyard_locations()))
			else:
				if closest_shipyard is None:
					closest_shipyard = newYard_loc
				ship_action = curr_ship.move_to_target_location(closest_shipyard)
				action_not_none = curr_ship.checkAction(ship_action,board,next_locations,actions,uid)
				if(not(action_not_none)):#TODO: Ships currently waste 1 turn staying at shipyard
					states[uid] = COLLECT #Once deposited, go back and collect 
					#TODO: Maybe Add reclustering bc a depositor becomes a collector. run halite_cluster_specific?
	for uid, shipyard in shipyards.items():
		curr_yard = Yard(shipyard, uid)
		if(len(ships) == 0):
			actions[uid] = SPAWN
		if (obs.step <150 and halite>=1000 and collector_count<8) or (obs.step <300 and halite>=3000 and collector_count<6) or (obs.step <400 and halite>=5000 and collector_count<4) or (halite>=1000 and defender_count<((collector_count + deposit_count)//2)):
			spawn = True
			for n in next_locations:
				if(same_pos_2d(n,curr_yard.coords_2d)):
					spawn = False
			if(spawn): #If there is room to spawn a new ship
				actions[uid] = SPAWN
				lastHaliteSpawn = halite
				next_locations.append(curr_yard.coords_2d)
		#if(obs.step<2):#DEBUG 
		yard_location = curr_yard.coords_2d
		#print("Yard Position:", curr_yard.coords_2d)
	end = time.time()
	#if(obs.step>390):
	#	print(obs.halite)
	return actions#, board.halite_board_2d, mapA.cluster, mapA.cluster_centers


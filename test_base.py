from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import random
import logging
import numpy as np
import time


DIRS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST, None]

COLLECT = "COLLECT"
DEPOSIT = "DEPOSIT"

CONVERT = ShipAction.CONVERT #"CONVERT"
SPAWN = ShipyardAction.SPAWN #"SPAWN"



class HaliteBoard(Board): # initialized the same way as the builtin Halite board with some extra methods attached

	def __init__(self, obs, config, *args):
		super().__init__(obs, config, *args)
		''' 
		defined by super:

		cells -> Dict[Point, Cell]
		ships -> Dict[ShipId, Ship]
		shipyards -> Dict[ShipyardId, Shipyard]
		players -> Dict[PlayerId, Player]

		current_player_id -> PlayerId
		current_player -> Player
		opponents -> List[Player]

		configuration -> Configuration
		observation -> Dict[str, Any]
		step -> int

		
		inherited methods:
		next() -> Board

		__deepcopy__(_) -> Board
		__getitem__(point: Union[Tuple[int, int], Point]) -> Cell
		__str__() -> str
		'''

		self.height = self.width = self.configuration["size"]
		self.halite_board_1d = np.array(obs.halite)
		self.halite_board_2d = np.array(obs.halite).reshape(self.width,self.height) # [r][c]

	# returns an array of Points ((x, y) tuples) of halite locations that have halite deposits above a certain threshold
	def get_halite_locations(self, threshold):
		hy, hx = np.where(self.halite_board_2d > threshold)
		halite_coords = list(Point(x, y) for x, y in zip(hx, hy))
		return halite_coords

	# returns ship locations given a player argument (player can be int or player object)
	# to get all ally ship locations, pass in HaliteBoardObject.current_player into the player argument when you call this.
	def get_all_ship_locations_for_player(self, player):
		if(isinstance(player, Player)):
			return [ship.position for ship in player.ships]
		if(isinstance(player, int)):
			return [ship.position for ship in self.players[player].ships]

	# returns all points which have enemy ships
	def get_all_enemy_ship_locations(self):
		locs = []
		for player in opponents:
			if(player != self.current_player):
				locs.extend(self.get_all_ship_locations_for_player(player))
		return locs

	# returns all ally ship locations
	def get_ship_locations(self):
		return self.get_all_ship_locations_for_player(self.current_player)

	# gets all shipyard points for a given player
	def get_all_shipyard_locations_for_player(self, player):
		if(isinstance(player, Player)):
			return [ship.position for ship in player.shipyards]
		if(isinstance(player, int)):
			return [ship.position for ship in self.players[player].shipyards]

	# returns all points which have enemy shipyards
	def get_all_enemy_shipyard_locations(self):
		locs = []
		for player in opponents:
			if(player != self.current_player):
				locs.extend(self.get_all_shipyard_locations_for_player(player))
		return locs

	# returns all ally shipyard locations
	def get_shipyard_locations(self):
		return self.get_all_shipyard_locations_for_player(self.current_player)


# class wrapper for ships that provides pathing abstractions
class Agent():
	def __init__(self, ship):
		'''
		id -> ShipId
		halite -> int

		position -> Point
		cell -> Cell

		player_id -> PlayerId
		player -> Player

		next_action -> Optional[ShipAction]
		'''
		self.ship = ship
		self.id = ship.id
		self.halite = ship.halite
		self.position = ship.position
		self.cell = ship.cell
		self.player_id = ship.player_id
		self.player = ship.player
		self.next_action = ship.next_action

	# returns a random action amongst the cardinal directions and None
	def return_random_action(self):
		action = random.choice(DIRS)
		return action

	# stays put
	def stall(self):
		return None



class Yard():
	def __init__(self, shipyard):
		'''
		id -> ShipyardId

		position -> Point
		cell -> Cell

		player_id -> PlayerId
		player -> Player

		next_action -> Optional[ShipyardAct
		'''
		self.shipyard = shipyard
		self.id = shipyard.id
		self.position = shipyard.position
		self.cell = shipyard.cell
		self.player_id = shipyard.player_id
		self.player = shipyard.player
		self.next_action = shipyard.next_action


	# a basic spawner strategy for a shipyard
	def spawn_strategy_1(self, step, halite):
		if(step < 100):
			return SPAWN
		if(step >= 100 and halite > 3000):
			return SPAWN




print_time = False
global states

def agent(obs, config):
	start = time.time()
	board = HaliteBoard(obs, config)
	me = board.current_player


	# at a minimum maintain 1 shipyard and 1 ship
	for ship in me.ships:
		if(len(me.shipyards) == 0):
			ship.next_action = CONVERT
		else:
			# cast the current ship to an "Agent," so that we can use movement methods and other easy abstractions
			curr_ship = Agent(ship)
			ship.next_action = curr_ship.return_random_action()


	for shipyard in me.shipyards:
		if(len(me.ships) == 0):
			shipyard.next_action = SPAWN
		else:
			# cast the current shupyard to a "Yard," so that we can easily create "strategies" in the form of methods
			curr_yard = Yard(shipyard)
			shipyard.next_action = curr_yard.spawn_strategy_1(board.step, me.halite)






	end = time.time()
	if(print_time):
		print(end-start)
	return me.next_actions
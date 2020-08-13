import copy
BOARD_DIMS = 21

firstShipyard = (9, 9)
secondShipyard = (20, 20)

print("first shipyard", firstShipyard)
print("second shipyard", secondShipyard)

def manhattan_distance(p1, p2):
	row_difference = min(abs(p1[0]-p2[0]),abs(p1[0]-p2[0]-BOARD_DIMS),abs(p1[0]-p2[0]+BOARD_DIMS))
	col_difference = min(abs(p1[1]-p2[1]),abs(p1[1]-p2[1]-BOARD_DIMS),abs(p1[1]-p2[1]+BOARD_DIMS))
	return row_difference + col_difference

print("Shipyard distance: ",manhattan_distance(firstShipyard, secondShipyard))

sum = 0
newSum = 0
for i in range(0,21):
    for j in range(0,21):
        firstDist = manhattan_distance(firstShipyard, (i,j))
        secondDist = manhattan_distance(secondShipyard, (i,j))
        sum += firstDist
        newSum += min(firstDist, secondDist)


print("Minimum Distance to first shipyard", sum)
print("Minimum Distance to both shipyard", newSum)


a = [1,2,3,4,5,6]
b = copy.copy(a)
b.pop(1)
a[0] = 10
print(a)
print(b)


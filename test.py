import numpy as np

def hashit(board):
    base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
    return int(base3)


def features(board):
	singlets = 0
	doublets = 0

	verticals = [board[0:3], board[3:6], board[6:9]]

	horizontal = [[board[0], board[3], board[6]], [board[1], board[4], board[7]], [board[2], board[5], board[8]]]

	diagonal = [[board[0], board[4], board[8]], [board[2], board[4], board[6]]]

	for direction in [verticals, horizontal, diagonal]:
		for lines in direction:
			if (0 in lines):
				if (sum(lines) == 1):
					singlets += 1
				elif (sum(lines) == 2):
					doublets += 1
	
	# returns [singles, doubles, board[0], ..., board[8]]
	return [singlets, doublets] + [int(i) for i in board]

board = np.zeros(9)
print(board[10000])

''' board[2] = 1
board[1] = 2
board[0] = 1

feature_board = board
feature_board[board == 2] = -1

fet = features(feature_board)

a = {}
a[1] = {'f':[1,2], 'v':100}
a[2] = {'f':[3,4], 'v':200}

f = [a[x]['f'] for x in a]
v = [a[x]['v'] for x in a]


sol1 = np.linalg.solve(f, v)

print(sol1) # fæ [0, 50]

parameter1 = np.linalg.inv(np.matmul(np.transpose(f) , f)) # (f.T * f)^-1

test1 = np.transpose(f) #f.T
#print(test1)

test2 = np.matmul(test1, f) #f.T * f
#print(test2)

test3 = np.linalg.inv(test2) # (f.T * f)^-1
#print(test3)

test4 = np.matmul(test3, test1) # (f.T * f)^-1 * f.T
#print(test4)

sol2 = np.matmul(np.around(test4, decimals=1), np.transpose(v))

print(sol2) # fæ [0, 50]


#print(sol2) # fæ [-200, 150] '''
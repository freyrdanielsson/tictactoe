import numpy as np

def hashit(board):
	base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
	return int(base3)

def legal_moves(board):
	return np.where(board == 0)[0]

def epsilongreedy(board, player, epsilon, debug = False):
	moves = legal_moves(board)
	if (np.random.uniform() < epsilon):
		if (1 == debug):
			print("explorative move")
		return np.random.choice(moves, 1)
	na = np.size(moves)
	va = np.zeros(na)
	for i in range(0, na):
		board[moves[i]] = player
		va[i] = value[hashit(board)]
		board[moves[i]] = 0  # undo
	return moves[np.argmax(va)]

def approximate(board, player):
	moves = legal_moves(board)
	na = np.size(moves)
	va = np.zeros(na)
	for i in range(0, na):
		board[moves[i]] = player
		va[i] = v_approx.get(hashit(board))
		board[moves[i]] = 0  # undo
	return moves[np.argmax(va)]


def iswin(board, m):
	if np.all(board[[0, 1, 2]] == m) | np.all(board[[3, 4, 5]] == m):
		return 1
	if np.all(board[[6, 7, 8]] == m) | np.all(board[[0, 3, 6]] == m):
		return 1
	if np.all(board[[1, 4, 7]] == m) | np.all(board[[2, 5, 8]] == m):
		return 1
	if np.all(board[[0, 4, 8]] == m) | np.all(board[[2, 4, 6]] == m):
		return 1
	return 0

def getotherplayer(player):
	if (player == 1):
		return 2
	return 1


# Calculate features for board
def features(board):
	singletsX = 0
	doubletsX = 0

	singletsO = 0
	doubletsO = 0

	verticals = [board[0:3], board[3:6], board[6:9]]

	horizontal = [[board[0], board[3], board[6]], [board[1], board[4], board[7]], [board[2], board[5], board[8]]]

	diagonal = [[board[0], board[4], board[8]], [board[2], board[4], board[6]]]

	for direction in [verticals, horizontal, diagonal]:
		for lines in direction:
			if (0 in lines):
				if (sum(lines) == 1):
					singletsX += 1
				elif (sum(lines) == 2):
					doubletsX += 1

# uncomment and fix return value if we want these fetures as well
	''' for direction in [verticals, horizontal, diagonal]:
		for lines in direction:
			if (0 in lines):
				if (sum(lines) == -1):
					singletsO += 1
				elif (sum(lines) == -2):
					doubletsO += 1 '''

	# returns [1, singles, doubles, actual_board[0], ..., actual_board[8]]
	# Tried removing the first feature (1) which I only have because the book has it
	# i got some estimate values greater then 1.0 which did not make sense so I put it there again
	actual_board = board
	actual_board[board == -1] = 2
	return [singletsX, doubletsX] + [int(i) for i in actual_board]

def learnit(numgames, epsilon, alpha, debug = False):
	# play games for training
	for games in range(0, numgames):
		board = np.zeros(9)          # initialize the board
		# This will be the board player one sees before making an action, have to store it to be able to
		# keep the p1_features object clean and only containing features and values for player 1
		old_board_view_p1 = []
		sold = [0, hashit(board), 0] # first element not used
		# player to start is "1" the other player is "2"
		player = 1
		# start turn playing game, maximum 9 moves
		for move in range(0, 9):
			# store the features and value for this state for player 1
			if (player == 1):
				updateFeatureValue(np.copy(board)) # update features for the board before taking action
			
			# use a policy to find action
			action = epsilongreedy(np.copy(board), player, epsilon, debug)

			# perform move and update board (for other player)

			if (player == 1):
				updateFeatureValue(np.copy(board)) # update features for the board after taking action
			board[action] = player
			if debug: # print the board, when in debug mode
				symbols = np.array([" ", "X", "O"])
				print("player ", symbols[player], ", move number ", move+1, ":")
				print(symbols[board.astype(int)].reshape(3,3))
			if (1 == iswin(board, player)): # has this player won?
				value[sold[player]] = value[sold[player]] + alpha * (1.0 - value[sold[player]]) 
				sold[player] = hashit(board) # index to winning state
				value[sold[player]] = 1.0 # winner (reward one)								
				value[sold[getotherplayer(player)]] = 0.0 # looser (reward zero)
				break
			# do a temporal difference update, once both players have made at least one move
			if (1 < move):
				value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
			sold[player] = hashit(board) # store this new state for player
			# check if we have a draw, then set the final states for both players to 0.5
			if (8 == move):
				value[sold] = 0.5 # draw (equal reward for both)
			player = getotherplayer(player) # swap players
			

def compete(numgames, debug = False, featuresOn = False):
	# play games for competition
	wins = {'p1':0, 'p2':0, 'tie':0}
	for games in range(0, numgames):
		board = np.zeros(9)          # initialize the board
		
		player = 1
		for move in range(0, 9):
			if (player == 1 and featuresOn):
				action = approximate(np.copy(board), player)
			else:
				action = epsilongreedy(np.copy(board), player, 0, debug)

			board[action] = player

			if debug: # print the board, when in debug mode
				symbols = np.array([" ", "X", "O"])
				print("player ", symbols[player], ", move number ", move+1, ":")
				print(symbols[board.astype(int)].reshape(3,3))

			if (1 == iswin(board, player)): # has this player won?
				if (player == 1):
					wins['p1'] += 1
				else:
					wins['p2'] += 1
				break
			if (8 == move):
				wins['tie'] += 1
			player = getotherplayer(player) # swap players

	return wins

# parameters:
# board: board for which features will be calulated
# sold: list containing pointers to old states of the board
# player=1: the player we would like to update for

# inserts a key-value object into p1_features if key does not exist, otherwise updates the value for given key
# how the object will look
# {123: {feature: [array], value: float}, ...} = {key: {value}, ...} = {state: {feature and true value for that state}, ...}
def updateFeatureValue(board):
	feature_board = board
	feature_board[board == 2] = -1
	# for state s = hashit(board), store features and true value
	p1_features[hashit(board)] = features(feature_board)

# global after-state value function, note this table is too big and contrains states that
# will never be used, also each state is unique to the player (no one after-state seen by both players) 
value = np.ones(hashit(2 * np.ones(9))) / 1.0
p1_features = {}
alpha = 0.1 # step size
epsilon = 0.1 # exploration parameter
# train the value function using 10000 games
learnit(10000, epsilon, alpha)

features = [p1_features[x] for x in p1_features]

# get values for all states that have features
true_values = [value[x] for x in p1_features]

ftrans = np.transpose(features) # f.T
fmul = np.matmul(ftrans, features) # f.T * f
finv = np.linalg.inv(fmul) # (f.T * f)^-1
finv_mul = np.matmul(finv, ftrans) # (f.T * f)^-1 * f.T
weigts = np.matmul(finv_mul, true_values) # (f.T * f)^-1 * f.T * v

v_approx = {}
lo, hi = 100, -100
count = 0

for i in p1_features:
	# this gives a value function of
	# v_e = x1w1 + x2w2 + ... +x12w12
	x = np.matmul(p1_features[i], weigts)
	lo, hi = min(x, lo), max(x, hi) # just checking if there is anything interesting with max, min in v_approx
	v_approx[i] = x


#print(len(p1_features)) # for how many states have we collected features?
#print(lo, hi) # anything interesting with the lo, hi?
#print(len(v_approx))

# play games using TD(0) for both players
print("Play games 100 using TD(0) ")
tdGames = compete(100, False, False)
print("Player 1 - wins: ", tdGames.get('p1'))
print("Player 2 - wins: ", tdGames.get('p2'))
print("Ties: ", tdGames.get('tie'))

print("Play games 1000 using TD(0) ")
tdGames = compete(1000, False, False)
print("Player 1 - wins: ", tdGames.get('p1'))
print("Player 2 - wins: ", tdGames.get('p2'))
print("Ties: ", tdGames.get('tie'))

# play games where player 1 users value function approximation
print("Play games 100 using value function approximation for player 1 ")
featurGames = compete(100, False, True)
print("Player 1 - wins: ", featurGames.get('p1'))
print("Player 2 - wins: ", featurGames.get('p2'))
print("Ties: ", featurGames.get('tie'))

print("Play games 1000 using value function approximation for player 1 ")
featurGames = compete(1000, False, True)
print("Player 1 - wins: ", featurGames.get('p1'))
print("Player 2 - wins: ", featurGames.get('p2'))
print("Ties: ", featurGames.get('tie'))

print('Weights used to evaluate the value function approximation: ', weigts)

# Results: I need to figure out where to place the updateFeatures function in the learnit
# player 1 allways ends up doing the same moves... his first move is always the same as well, maybe the features are bad
# because they don't have any information on how player 2 is doing (in terms of singlets, doublets)
# or maybe i'm not using the features for the board smart enough...

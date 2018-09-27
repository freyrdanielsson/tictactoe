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

def learnit(numgames, epsilon, alpha, startingPlayer = 1, debug = False, vsPlayer = False):
		# play games for training
		for games in range(0, numgames):
				board = np.zeros(9)          # initialize the board
				sold = [0, hashit(board), 0] # first element not used
				# player to start is "1" the other player is "2"
				player = startingPlayer
				# start turn playing game, maximum 9 moves
				for move in range(0, 9):
						if vsPlayer:
							if (player == 1):
									# use a policy to find action
									action = epsilongreedy(np.copy(board), player, epsilon, debug)
									print("We make a move on " + str(action))
									# perform move and update board (for other player)
									board[action] = player
							if (player == 2 and debug):
									legal = False
									moves = legal_moves(board)
									while not legal:
										action = int(input("make a move: "))
										if (action in moves):
											legal = True
										else:
											print('That\'s not a legal move ' + u'👾 ')
									board[action] = player
						else:
									#if training use this for both players
									action = epsilongreedy(np.copy(board), player, epsilon, debug)
									#perform move and update board (for other player)
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
								if(vsPlayer and (player == 2)):
										print('smart cookie ' + u'🍪')
								break
						
						# do a temporal difference update, once both players have made at least one move
						if (1 < move):
								value[sold[player]] = value[sold[player]] + alpha * (value[hashit(board)] - value[sold[player]])
						sold[player] = hashit(board) # store this new state for player
						# check if we have a draw, then set the final states for both players to 0.5
						if (8 == move):
								value[sold] = 0.5 # draw (equal reward for both)
						player = getotherplayer(player) # swap players


value = np.ones(hashit(2 * np.ones(9))) / 1.0
alpha = 0.1 # step size
epsilon = 0.1 # exploration parameter
print("Give us a sec while we learn tic tac toe " + u'🧠  🧠')
print("Here are the rules while you wait")
print("This is what the board looks like in terms of moves you can make")
print(np.array(range(0,9)).reshape(3,3))
print("simply select the number on the board where you want to make your move")

# train the value function using 10000 games
learnit(20000, epsilon, alpha)

# compete with self
learnit(20000, 0, 0)

print("We have now mastered tic tac toe, play us " + u'🐍')
# play a game agains a human
learnit(1, 0, 0, 1, True, True)
while True:
	if(input("O boy, want to play again ?"+u'😈'+" (yes/no): ") == 'yes'):
			if( input('Do you want to have the first move? (yes/no): ') == 'yes'):
					learnit(1, 0, 0, 2, True, True)
			else:
					learnit(1, 0, 0, 1, True, True)
	else:
			print('see you later aligator')
			break


# Todo: make agent better when he doesn't have the first move
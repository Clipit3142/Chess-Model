import chess.pgn
import chess
import chess.engine
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

NUM_MOVES = 4672
INPUT_SIZE = 64

def fetch_games(pgn_folder):
    games = []
    for filename in os.listdir(pgn_folder):
        if filename.endswith(".pgn"):
            with open(os.path.join(pgn_folder, filename), "r", encoding="utf-8") as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    games.append(game)
    return games

def piece_to_int(piece):
    if piece is None:
        return 0
    value = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6
    }[piece.piece_type]
    return value if piece.color == chess.WHITE else -value

def board_to_vector(board):
    return [piece_to_int(board.piece_at(square)) for square in chess.SQUARES]

def extract_positions_from_games(games_list):
    data = []
    for game in games_list:
        board = game.board()
        for move in game.mainline_moves():
            position_vector = board_to_vector(board)
            move_uci = move.uci()
            data.append((position_vector, move_uci))
            board.push(move)
    random.shuffle(data)
    print("Total Positions:",len(data))
    return data

def create_train_data(data, max_positions):
    data = data[:max_positions]
    x_train = []
    y_train = []
    for position_vector, move_uci in data:
        board = chess.Board(list_to_fen(position_vector))
        move = chess.Move.from_uci(move_uci)

        if move not in board.legal_moves:
            continue

        move_index = move_to_index(move)
        if move_index >= NUM_MOVES:
            continue  # skip moves that exceed the encoding

        x_train.append(position_vector)

        label = np.zeros(NUM_MOVES, dtype=np.float32)
        label[move_index] = 1.0
        y_train.append(label)

    return np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

def list_to_fen(board_list):
    piece_map = {
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
        -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
        0: ''
    }
    fen = ""
    for rank in range(7, -1, -1):  # top to bottom
        empty = 0
        for file in range(8):
            piece = board_list[rank * 8 + file]
            if piece == 0:
                empty += 1
            else:
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += piece_map[piece]
        if empty > 0:
            fen += str(empty)
        if rank > 0:
            fen += "/"
    return fen + " w - - 0 1"

def move_to_index(move: chess.Move) -> int:
    # from_square: 0–63, to_square: 0–63
    base = move.from_square * 64 + move.to_square
    if move.promotion:
        # promotion offset: 0 = None, 1 = N, 2 = B, 3 = R, 4 = Q
        # N=0, B=1, R=2, Q=3 -> add 4096 + offset
        promo_offset = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}[move.promotion]
        return 4096 + promo_offset * 64 * 64 + base
    return base

def index_to_move(index: int) -> chess.Move:
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)
    else:
        base = index % 4096
        from_sq = base // 64
        to_sq = base % 64
        promo_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        promo = promo_map[(index - 4096) // 4096]
        return chess.Move(from_sq, to_sq, promotion=promo)

def evaluate_fen(fen):
    board = chess.Board(fen)
    try:
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white().score(mate_score=10000)
        return score
    except Exception as e:
        return f"Eval error: {e}"

def print_position(position, use_unicode=True):
    piece_unicode = {
        1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',
        -1: '♟', -2: '♞', -3: '♝', -4: '♜', -5: '♛', -6: '♚',
        0: '.'
    }
    fen = list_to_fen(position)
    score = evaluate_fen(fen)
    print("Eval:", score)
    for rank in range(7, -1, -1):  # top to bottom
        row = ""
        for file in range(8):
            idx = rank * 8 + file
            val = position[idx]
            row += piece_unicode.get(val, '?') + " " if use_unicode else f"{val:2} "
        print(row)

def predict_best_move(model, position_vector):
    input_data = np.array(position_vector).reshape(1, -1)
    probs = model.predict(input_data)[0]
    best_move_index = np.argmax(probs)
    return best_move_index, probs[best_move_index]



engine_path = "stockfish"
if not os.path.isfile(engine_path):
    raise FileNotFoundError("Stockfish executable not found.")
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

pgn_folder = "pgn games"



games = fetch_games(pgn_folder)
data = extract_positions_from_games(games)
x_train, y_train = create_train_data(data, 100000)

model = keras.Sequential()
model.add(Input(shape=(INPUT_SIZE,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu')) # placeholder model
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_MOVES, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=64)

model.save("chess_model.h5")
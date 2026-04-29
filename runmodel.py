import chess.pgn
import chess
import chess.engine
import os
import random
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.models import load_model


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
        # We'll say N=0, B=1, R=2, Q=3 → add 4096 + offset
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

def board_to_input_vector(board: chess.Board):
    # Converts current board to flat 64-length input vector with same mapping you used during training
    vec = [0] * 64
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        offset = square
        piece_value = piece.piece_type
        vec[offset] = piece_value if piece.color == chess.WHITE else -piece_value
    return vec

def input_vector_to_board(vec):
    return vec

def play_self_game(model):
    board = chess.Board()
    while not board.is_game_over():
        position_vector = board_to_input_vector(board)
        best_idx, confidence = predict_best_move(model, position_vector)
        best_move = index_to_move(best_idx)

        if best_move not in board.legal_moves:
            # If predicted move is illegal, pick a random legal one
            print(f"Illegal predicted move: {best_move}, picking random move.")
            best_move = np.random.choice(list(board.legal_moves))

        print_position(input_vector_to_board(position_vector))
        print(f"\n{board.fullmove_number}. {'White' if board.turn else 'Black'} plays {best_move.uci()} with confidence {confidence:.2f}\n")
        board.push(best_move)

    print("Game over:", board.result())

def play_against_bot(model):
    board = chess.Board()
    
    while not board.is_game_over():
        print("\nCurrent position:")
        print(board)

        if board.turn == chess.WHITE:
            user_move_str = input("Your move: ").strip()
            try:
                user_move = chess.Move.from_uci(user_move_str)
                if user_move in board.legal_moves:
                    board.push(user_move)
                else:
                    print("Illegal move. Try again.")
                    continue
            except:
                print("Invalid notation. Try again.")
                continue
        else:
            position_vector = board_to_input_vector(board)
            best_idx, confidence = predict_best_move(model, position_vector)
            best_move = index_to_move(best_idx)

            if best_move not in board.legal_moves:
                # If predicted move is illegal, pick a random legal one
                print(f"Illegal predicted move: {best_move}, picking random move.")
                best_move = np.random.choice(list(board.legal_moves))

            print_position(input_vector_to_board(position_vector))
            print(f"\n{board.fullmove_number}. {'White' if board.turn else 'Black'} plays {best_move.uci()} with confidence {confidence:.2f}\n")
            board.push(best_move)

    print("\nGame over!")
    print(f"Result: {board.result()}")

engine_path = "stockfish"
if not os.path.isfile(engine_path):
    raise FileNotFoundError("Stockfish executable not found.")
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

model = load_model("chess_model.h5")

play_against_bot(model)
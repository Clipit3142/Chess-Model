import chess.pgn

def extract_endgames_from_pgn(pgn_path, max_positions=30000):
    endgame_positions = []
    with open(pgn_path, 'r', encoding='utf-8') as f:
        while len(endgame_positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                # Check for endgame criteria
                piece_counts = sum(1 for piece in board.piece_map().values() if piece.piece_type != chess.PAWN)
                if piece_counts <= 6:
                    endgame_positions.append(board.fen())
                    if len(endgame_positions) >= max_positions:
                        break
    return endgame_positions
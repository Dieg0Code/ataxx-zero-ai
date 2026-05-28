"""Game constants for Ataxx."""

BOARD_SIZE = 7
# v15 final: 11 canales legacy + 4 nuevos = 15.
#   0-10  : legacy (own/opp/empty, half_moves, repetition, mobility planes)
#   11    : own_captures_potential (max conversions si jugamos a esa celda)
#   12    : opp_captures_potential (idem desde la perspectiva del rival)
#   13    : last_move_dst (1.0 en el destino del ultimo move; resto 0)
#   14    : prev_move_dst (idem del penultimo)
# Checkpoints pre-v15-final cargan con num_input_channels=11 via backfill
# en checkpoint_compat; el board les pasa solo los primeros 11 canales.
OBSERVATION_CHANNELS = 15
OBSERVATION_CHANNELS_LEGACY = 11

EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = -1

WIN_P1 = 1
WIN_P2 = -1
DRAW = 0

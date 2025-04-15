# load handranks.dat

import array

def load_handranks(path= "TwoPlusTwoHandEvaluator\HandRanks.dat"):
    hand_ranks = array.array("I")
    with open(path, 'rb') as f:
        hand_ranks.fromfile(f, 32487834)

    return hand_ranks

hr = load_handranks()

def evaluate_5_card(hand: list, hand_ranks):
    id = 53
    for card in hand:
        id = hand_ranks[id + card]

    return hand_ranks[id]


RANKS = '23456789TJQKA'
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
SUITS = {'c': 0x8000, 'd': 0x4000, 'h': 0x2000, 's': 0x1000}

# Maps card string to rank and prime
RANK_TO_INT = {rank: i for i, rank in enumerate(RANKS)}
RANK_TO_PRIME = {rank: prime for rank, prime in zip(RANKS, PRIMES)}

def encode_card(rank: str, suit: str) -> int:
    r = RANK_TO_INT[rank]
    prime = RANK_TO_PRIME[rank]
    s = SUITS[suit]
    rank_bit = 1 << r

    return (rank_bit << 16) | (s) | (r << 8) | prime

def deck_builder():
    deck = []

    for rank in RANKS:
        for suit in SUITS:
            card = encode_card(rank, suit)
            deck.append(card)

    return deck

print(deck_builder())
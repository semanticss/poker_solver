import seaborn as sns
import matplotlib as plt
import math
from itertools import combinations
import random as rand
import math
from collections import Counter
import os
import tqdm as tqdm
from multiprocessing import Pool, cpu_count


class DeckManager():
    def __init__(self, shuffle_on_reset = True, stage = 'preflop'):
        self.stage = stage
        self.stage_counts = {
            "preflop": 0,
            "flop": 3,
            "turn": 4,
            "river": 5
        }
        self.shuffle_on_reset = shuffle_on_reset
        self.full_deck = list(range(52))
        rand.shuffle(self.full_deck)
        self.reset()

    def reset(self):
        self.remaining_cards = self.full_deck.copy()

    def remove_used(self, used_cards):
        self.remaining_cards = [card for card in self.remaining_cards if card not in used_cards]
    
    def deal(self, count: int):
        dealt_cards = self.remaining_cards[:count]
        self.remaining_cards = self.remaining_cards[count:]
        return dealt_cards
    
    def stage_setter(self, game_stage):
        assert game_stage in ['preflop', 'flop', 'turn', 'river']
        self.game_stage = game_stage


    def cards_needed_for_stage_calc(self):
        self.cards_needed_for_stage = 5 - self.stage_counts[self.stage]
        return self.cards_needed_for_stage
    
    def deal_community_to_stage(self, current_community: list):
        needed = self.cards_needed_for_stage_calc()
        new_cards = self.deal(needed)
        return current_community + new_cards


hands_score_dict = { # <-- quantified hand rankings
        "royal_flush": 9,
        "straight_flush": 8,
        "quads": 7,
        "full_house": 6,
        "flush": 5,
        "straight": 4,
        "three_of_a_kind": 3,
        "two_pair": 2,
        "pair": 1,
        "high_card": 0
    }

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['s', 'h', 'c', 'd']

def card_str_to_int(card_str):
        
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                    '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        rank_part = card_str[:-1]
        suit_part = card_str[-1].lower()
        return 13 * suit_map[suit_part] + rank_map[rank_part.upper()]

def get_rank(card: int):
    assert type(card) == int
    return card % 13

def get_suit(card: int):
    assert type(card) == int
    return card // 13

def evaluate_five_card_hand(cards: list):
    
    ranks = [get_rank(c) for c in cards] 
    suits = [get_suit(c) for c in cards] 
    sorted_ranks = sorted(ranks) 
    rank_appearances = Counter(ranks)
    suit_appearances = Counter(suits)

    flush = straight = False
    
    if len(set(suits)) == 1:
        flush = True

    if len(set(ranks)) == 5 and (sorted_ranks[-1] - sorted_ranks[0] == 4 or set([0, 1, 2, 3, 12]).issubset(set(ranks))): # <-- i think this logic is correct
        straight = True

    # returns highest score
    if flush and sorted_ranks == [8,9,10,11,12]:
        return (hands_score_dict["royal_flush"], [])
    
    elif flush and straight:
        return (hands_score_dict["straight_flush"], [max(sorted_ranks)])
    
    elif 4 in rank_appearances.values():
        quads_rank = [rank for rank, count in rank_appearances.items() if count == 4][0]
        kicker = max(r for r in ranks if r != quads_rank) # check logic
        return (hands_score_dict["quads"], [quads_rank, kicker])
    
    elif 3 in rank_appearances.values() and 2 in rank_appearances.values():

        trips_rank = max(rank for rank, count in rank_appearances.items() if count == 3)
        pair_rank = max(rank for rank, count in rank_appearances.items() if count == 2)
        return (hands_score_dict["full_house"], [trips_rank, pair_rank])

    elif flush:
        return (hands_score_dict["flush"], sorted_ranks)
    
    elif straight:
        return (hands_score_dict["straight"], [max(sorted_ranks)])
    
    elif 3 in rank_appearances.values():
        trips_rank = [rank for rank, count in rank_appearances.items() if count == 3][0]
        kicker = max(r for r in ranks if r != trips_rank) # check logic
        return (hands_score_dict["three_of_a_kind"], [trips_rank] + [kicker])
    
    elif list(rank_appearances.values()).count(2) == 2:
        pair_ranks = sorted([rank for rank, count in rank_appearances.items() if count == 2], reverse=True)
        kicker = [rank for rank in reversed(sorted_ranks) if rank not in pair_ranks][0]
        return (hands_score_dict["two_pair"], pair_ranks[:2] + [kicker])
    
    elif 2 in rank_appearances.values():
        pair_rank = [rank for rank, count in rank_appearances.items() if count == 2][0]
        kickers = [rank for rank in reversed(sorted_ranks) if rank != pair_rank][:3]
        return (hands_score_dict["pair"], [pair_rank] + kickers)
    
    else:
        high_cards = list(reversed(sorted_ranks))[:5]
        return (hands_score_dict["high_card"], high_cards)
    
def best_seven_card_hand(cards: list):
    assert len(cards) == 7
    return max(evaluate_five_card_hand(list(combo)) for combo in combinations(cards, 5))

def monte_sim(hero_hole: tuple, community_cards: list, stage = 'preflop', num_simulations: int = 1000000, vills: int = 4)-> float:

    wins, ties, losses = 0,0,0

    for test in tqdm.tqdm(range(num_simulations)):

        deck = DeckManager(stage = stage)
        used = set(hero_hole + tuple(community_cards))
        deck.remove_used(used)

        vill_cards = deck.deal(2 * vills)
        vill_hands = [tuple(vill_cards[i:i + 2]) for i in range(0, 2 * vills, 2)]

        board = deck.deal_community_to_stage(community_cards)
        
        hero_best = best_seven_card_hand(list(hero_hole) + board)
        vill_best = [best_seven_card_hand(list(opp) + board) for opp in vill_hands]

        if any(hero_best < i for i in vill_best):
            losses += 1
        elif any(hero_best == i for i in vill_best):
            ties += 1
        else:
            wins += 1

    return (wins + ties) / num_simulations


def ev_calc(pot: float, call: float, winp: float, tiep: float, oppnum: int) -> float:

    expected_val = (winp * pot) + (tiep * pot / (1 + oppnum)) - call

    return expected_val

def best_choice(pot: float, call: float, raise_amnt: float, winp: float, tiep: float, oppnum: int) -> str:

    ev_call = ev_calc(pot, call, winp, tiep, oppnum)
    ev_raise = ev_calc(pot + raise_amnt, raise_amnt, winp, tiep, oppnum)

    if ev_call > 0 and ev_call >= ev_raise:
        return 'call'
    elif ev_raise > ev_call and ev_raise > 0:
        return 'raise'
    else:
        return 'fold'

def poker_solver(hero_hole: tuple, community_cards: list, # return after 100, 1000, 10000, etc.
                 stage: str = 'preflop',
                 num_simulations: int = 1000, num_opponents: int = 4) -> dict:

    chances = monte_sim(hero_hole, community_cards)

    return chances


def card_str_to_int(card_str):
        
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                    '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        rank_part = card_str[:-1]
        suit_part = card_str[-1].lower()
        return 13 * suit_map[suit_part] + rank_map[rank_part.upper()]


def generate_preflop_int_combos():
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    combos = {}

    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            if i < j:
                label_suited = f"{r2}{r1}s"
                label_offsuit = f"{r2}{r1}o"

                # One suited example (same suit)
                c1 = f"{r2}s"
                c2 = f"{r1}s"
                combos[label_suited] = (card_str_to_int(c1), card_str_to_int(c2))

                # One offsuit example (diff suits)
                c1 = f"{r2}s"
                c2 = f"{r1}h"
                combos[label_offsuit] = (card_str_to_int(c1), card_str_to_int(c2))

            elif i == j:
                label_pair = f"{r1}{r1}"
                c1 = f"{r1}s"
                c2 = f"{r1}h"
                combos[label_pair] = (card_str_to_int(c1), card_str_to_int(c2))

    return combos

def hand_simulation(args):
    hand_label, hero_hole = args

    return (hand_label, poker_solver(
        hero_hole,
        []
    ))


hands = list(generate_preflop_int_combos().items()) # returns the tuple with hand name and int tuple

if __name__ == "__main__": # does NOT work --> python switches cores and doesn't have good multithreading
    with Pool(18) as pool:
        results = pool.map(hand_simulation, hands)
    
    equity_map = dict(results)

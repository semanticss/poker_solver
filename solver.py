import random as rand
import math
from collections import Counter
import math
import os
import streamlit as st
from itertools import combinations
import tqdm as tqdm

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

    assert len(cards) == 5 # 
    
    ranks = [get_rank(c) for c in cards] 
    suits = [get_suit(c) for c in cards] 
    sorted_ranks = sorted(ranks) 
    rank_appearances = Counter(ranks)
    suit_appearances = Counter(suits)

    flush = straight = False
    
    # Flush checker
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

def monte_sim(hero_hole: tuple, vills: int, community_cards: list, stage = 'preflop', num_simulations: int = 1000)-> float:

    wins, ties, losses = 0,0,0

    for test in tqdm.tqdm(range(num_simulations)):

        deck = DeckManager(stage = stage)
        used = set(hero_hole + tuple(community_cards))
        deck.remove_used(used)

        vill_cards = deck.deal(2 * vills)
        vill_hands = [tuple(vill_cards[i:i + 2]) for i in range(0, 2 * vills, 2)]

        board = deck.deal_community_to_stage(community_cards)

        if len(board) != 5:
            raise ValueError(f"Board has {len(board)} cards instead of 5: {board}")
        
        hero_best = best_seven_card_hand(list(hero_hole) + board)
        vill_best = [best_seven_card_hand(list(opp) + board) for opp in vill_hands]

        if any(hero_best < i for i in vill_best):
            losses += 1
        elif any(hero_best == i for i in vill_best):
            ties += 1
        else:
            wins += 1

        
        if test > 0 and math.log10(test).is_integer():
            print(f'{wins} {ties} {losses}')
            print(f'wins + ties: {wins + ties}')
            print({

        "win_probability": wins / test,
        "tie_probability": ties / test,
        "loss_probability": losses / test,
        "win + ties / tests": (wins + ties) / test
    })

    return {
        "win_probability": wins / num_simulations,
        "tie_probability": ties / num_simulations,
        "loss_probability": losses / num_simulations,
        "win_ties_probability": (wins + ties) / num_simulations
    }

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

def poker_solver(hero_hole: tuple, num_opponents: int, community_cards: list, # return after 100, 1000, 10000, etc.
                 stage: str = 'preflop',
                 num_simulations: int = 1000) -> dict:

    chances = monte_sim(hero_hole, num_opponents, community_cards, stage, num_simulations)

    # win_probability = chances["win_probability"]
    # tie_probability = chances["tie_probability"]
    # decision = best_choice(pot_size, call_amount, raise_amount, win_probability, tie_probability, num_opponents)

    return {
        "probabilities": chances,
        "optimal_decision": "not implemented right now."
    }


def is_valid_card_str(card_str):
    valid_ranks = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}
    valid_suits = {'s', 'h', 'd', 'c'}

    rank_part = card_str[:-1].upper()
    suit_part = card_str[-1].lower()

    return rank_part in valid_ranks and suit_part in valid_suits


def make_sure_cards_and_stuff_isnt_super_cooked_holy_cow(card_strs: list, already_used = None):

    try:
        for card_str in card_strs:
            if not is_valid_card_str(card_str):
                raise ValueError(f"'{card_str}' is not a valid card.")

        cards = [card_str_to_int(card) for card in card_strs]

        if len(set(cards)) != len(cards):
            raise ValueError("Duplicate cards entered.")

        if already_used:
            if any(card in already_used for card in cards):
                raise ValueError("Card already used.")

        return cards

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":

    already_used = set()


    while True:
        hero_input = input("enter your hero hole cards (e.g. 'ace of spades = As king of spades = Ks'): ").strip().split()
        validated = make_sure_cards_and_stuff_isnt_super_cooked_holy_cow(hero_input)
        if validated and len(validated) == 2:
            hero_hole = tuple(validated)
            already_used.update(hero_hole)
            break
        else:
            print("Invalid hero cards. Try again.")


    stages = {'0': 'preflop', '1': 'flop', '2': 'turn', '3': 'river'}
    while True:
        print("choose a game stage:")
        print("0 - preflop (evaluate hand preflop)")
        print("1 - to flop (3 cards)")
        print("2 - to turn (4 cards)")
        print("3 - to river (5 cards)")
        stage_choice = input("type 0, 1, 2 or 3: ").strip()
        if stage_choice in stages:
            stage = stages[stage_choice]
            num_board_cards = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}[stage]
            break
        else:
            print("invalid pick")

    while True:

        mode = input("manually enter community cards? (y/n): ").strip().lower()
        if mode in {'y', 'n'}:
            manual = mode == 'y'
            break
        else:
            print("Please enter 'y' or 'n'.")

    if manual: # DUPILCATE CARDS BEING FLOPPED
        while True:
            board_input = input(f"Enter {num_board_cards} community cards (e.g., '9c Td Ah'): ").strip().split()
            validated = make_sure_cards_and_stuff_isnt_super_cooked_holy_cow(board_input, already_used)
            if validated and len(validated) == num_board_cards:
                community_cards = validated
                already_used.update(community_cards)
                break
            else:
                print("Invalid board. Try again.")
    else:
        deck = DeckManager(stage=stage)
        deck.remove_used(already_used)
        community_cards = deck.deal(num_board_cards)
        print(community_cards)
        already_used.update(community_cards)
        print("Generated board:")
        print(" ".join(f"{ranks[get_rank(c)]}{suits[get_suit(c)]}" for c in community_cards))


    num_opponents = int(input("Number of opponents: ").strip())
    # pot_size = float(input("Pot size: ").strip())
    # call_amount = float(input("Call amount: ").strip())
    # raise_amount = float(input("Raise amount: ").strip())
    num_simulations = int(input("Number of simulations: ").strip())


    result = poker_solver(
        hero_hole,
        num_opponents,
        community_cards,
        # pot_size,
        # call_amount,
        # raise_amount,
        stage,
        num_simulations
    )

    print("\n--- sim results ---")
    print("win % --> ", round(result['probabilities']['win_probability'] * 100, 2))
    print("tie % --> ", round(result['probabilities']['tie_probability'] * 100, 2))
    print("loss % --> ", round(result['probabilities']['loss_probability'] * 100, 2))
    print("win + tie % -->", round(result['probabilities']["win_ties_probability"] * 100, 2))
    print()
    print("Recommended action:", result['optimal_decision'].lower())
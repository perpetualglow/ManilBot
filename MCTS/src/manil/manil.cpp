#include "../include/manil.hpp"
#include "../include/sosolver.h"
#include "../include/solverbase.h"
#include "../include/config.h"

#include <ctime>
#include <chrono>
#include <algorithm>
#include <memory>
#include <numeric>
#include <iterator>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

typedef enum {SUIT, HIGHER, TRUMP, OTHER} FILTER;

namespace
{

    std::vector<Manil::Card> filter(
                    std::vector<Manil::Card> const& hand,
                    Manil::Card other,
                    int trump,
                    FILTER f) {
        std::vector<Manil::Card> res;
        if (f == SUIT) {
            for (auto& c : hand) {
                if (c.suit == other.suit)
                    res.push_back(c);
            }
        } else if (f == HIGHER) {
            for (auto& c : hand) {
                if (c.suit == other.suit && c > other)
                    res.push_back(c);
            }

        } else if (f == TRUMP) {
            for (auto& c : hand) {
                if (c.suit == trump)
                    res.push_back(c);
            }

        } else if (f == OTHER) {
            for (auto&c : hand) {
                if (c.suit != trump)
                    res.push_back(c);
            }
        }
        return res;
    }

    inline std::mt19937 &prng()
    {
        std::mt19937 static thread_local prng;
        return prng;
    }

}

ManilGame::ManilGame()
    :player_num{4}
    ,tricksleft{8}
{
    for (unsigned i {0}; i < 32; ++i) {
        or_deck[i] = {Manil::Card(i % 8, i / 8)};
    }
    std::srand(time(NULL));
    init();
}

ManilGame::ManilGame(unsigned players)
    :player_num{4}
    ,tricksleft{8}
{
    for (unsigned i {0}; i < 32; ++i) {
        or_deck[i] = {Manil::Card(i % 8, i / 8)};
    }
    std::srand(time(NULL));
    init();
}

void ManilGame::init() {
    for (unsigned i {0}; i < 32; ++i) {
        deck[i] = {Manil::Card(i % 8, i / 8)};
    }
    total_clones_called = 0;
    playerCards.resize(player_num);
    points.resize(player_num);
    m_players.resize(player_num);
    std::iota(points.begin(), points.end(), 0);
    std::iota(m_players.begin(), m_players.end(), 0);
    dealer.init();
    deal();
    current_player = std::rand() % 4;
    trump = std::rand() % 4;
    possCards.clear();
    initPossibleCards();
}

std::vector<double> ManilGame::run_game()
{
    for (int i=0; i<32; i++) {
        if (current_player % 2 == 0)
            doAIMove();
        else
            doRandomMove();
    }
    return points;
}

ManilGame::Clone ManilGame::cloneAndRandomise(Player observer) const
{
    auto clone = std::make_unique<ManilGame>(*this);
    // Hand unseenCards = unknownCards;
    // for (auto p : m_players) {
    //     if (p == observer)
    //         continue;
    //     auto const &hand = playerCards[p];
    //     unseenCards.insert(unseenCards.end(), hand.begin(), hand.end());
    // }
    // std::shuffle(unseenCards.begin(), unseenCards.end(), prng());
    // auto u = unseenCards.begin();
    // for (auto p : m_players) {
    //     if (p == observer)
    //         continue;
    //     auto &hand = clone->playerCards[p];
    //     std::copy_n(u, hand.size(), hand.begin());
    //     u += hand.size();
    // }
    std::vector<Manil::Card> card_delete;
    std::vector<Manil::Card> prevCards;
    for (auto p : m_players) {
        if (p == observer)
            continue;
        auto pos = possCards[observer][p].begin();
        std::copy_if(possCards[observer][p].begin(), possCards[observer][p].end(), std::back_inserter(prevCards), [&](auto const &c){
            auto const pos = std::find(card_delete.begin(), card_delete.end(), c);
            if (pos < card_delete.end() || c.rank < 0)
                return false;
            else
                return true;
            });
        std::shuffle(prevCards.begin(), prevCards.end(), prng());
        auto &hand = clone->playerCards[p];
        prevCards.resize(hand.size());
        std::copy_n(prevCards.begin(), hand.size(), hand.begin());
        card_delete.insert(card_delete.end(), prevCards.begin(), prevCards.end());
        prevCards.clear();
    }

    // Hand unseenCards = unknownCards;
    // for (auto p : m_players) {
    //     if (p == observer)
    //         continue;
    //     auto const &hand = playerCards[p];
    //     unseenCards.insert(unseenCards.end(), hand.begin(), hand.end());
    // }
    // std::shuffle(unseenCards.begin(), unseenCards.end(), prng());
    // auto u = unseenCards.begin();
    // for (auto p : m_players) {
    //     if (p == observer)
    //         continue;
    //     auto &hand = clone->playerCards[p];
    //     std::copy_n(u, hand.size(), hand.begin());
    //     u += hand.size();
    // }

    return clone;

        // std::shuffle(unseenCards.begin(), unseenCards.end(), prng());
        // std::cout << "unseen cards size 2 " << unseenCards.size() << std::endl;
        // auto &hand = clone->playerCards[p];

        // std::cout << "prev cards size " << prevCards.size() << std::endl;
        // std::cout << "prev cards size " << prevCards.size() << std::endl;
        // std::copy_n(unseenCards.begin(), hand.size(), hand.begin());
        // prevCards.insert(prevCards.end(), hand.begin(), hand.end());
        // std::cout << "copied " << std::endl;
}

ManilGame::Player ManilGame::currentPlayer() const
{
    return current_player;
}

std::vector<ManilGame::Player> ManilGame::players() const
{
    return m_players;
}

std::vector<Manil::Card> ManilGame::validMoves() const
{   
    std::vector<Manil::Card> res;
    std::vector<Manil::Card> hand = playerCards[current_player];
    Hand suit_cards;
    Hand other_cards;
    Hand higher_cards;
    Hand trump_cards;
    Hand higher_trump_cards;

    if (currentTrick.size() == 0) {
        return playerCards[current_player];
    }
    else if (currentTrick.size() == 1) {
        auto const table_card = currentTrick.front().second;
        std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == table_card.suit;});
        if (suit_cards.size() > 0) {
            std::copy_if(hand.begin(), hand.end(), std::back_inserter(higher_cards), [&](auto const &c){return c >= table_card;});
            if (higher_cards.size() > 0)
                return higher_cards;
            else
                return suit_cards;
        } else {
            std::copy_if(hand.begin(), hand.end(), std::back_inserter(trump_cards), [&](auto const &c){return c.suit == trump;});
            if (trump_cards.size() > 0)
                return trump_cards;
            else
                return hand;
            
        }
    }
    else if (currentTrick.size() == 2) {
        auto winner = currentTrick.begin();
        int ind = 0;
        int ind_winner = 0;
        bool bought = false;
        for (auto p = winner; p < currentTrick.end(); ++p) {
            if (winner->second.suit == trump){
                if (p->second >= winner->second) {
                    winner = p;
                    ind_winner = ind;
                    bought = true;
                }

            } else {
                if ((p->second >= winner->second) || (p->second.suit == trump))
                    winner = p;
                    ind_winner = ind;
            }
            ind += 1;
        }
        if (ind_winner == 0) {
            auto const table_card = currentTrick.front().second;
            std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == table_card.suit;});
            if (suit_cards.size() > 0)
                return suit_cards;
            else 
                return hand;
        } else if (ind_winner == 1) {
            auto const first_card = currentTrick.front().second;
            auto const sec_card = currentTrick[1].second;
            if (first_card.suit != trump) {
                if (sec_card.suit == trump) {
                    std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
                    if (suit_cards.size() > 0) {
                        return suit_cards;
                    } else {
                        std::copy_if(hand.begin(), hand.end(), std::back_inserter(higher_trump_cards), [&](auto const &c){return c >= sec_card;});
                        if (higher_trump_cards.size() > 0)
                            return higher_trump_cards;
                        else {
                            std::copy_if(hand.begin(), hand.end(), std::back_inserter(other_cards), [&](auto const &c){return c.suit != trump;});
                            if (other_cards.size() > 0)
                                return other_cards;
                            else
                                return hand;
                        }
                    }
                } else {
                    std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
                    if (suit_cards.size() > 0) {
                        std::copy_if(suit_cards.begin(), suit_cards.end(), std::back_inserter(higher_cards), [&](auto const &c){return c >= sec_card;});                       
                        if (higher_cards.size() > 0)
                            return higher_cards;
                        else
                            return suit_cards;
                        
                    } else {
                        std::copy_if(hand.begin(), hand.end(), std::back_inserter(trump_cards), [&](auto const &c){return c.suit == trump;});
                        if (trump_cards.size() > 0)
                            return trump_cards;
                        else
                            return hand;
                    }
                }
            } else {
                std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
                if (suit_cards.size() > 0){
                    std::copy_if(suit_cards.begin(), suit_cards.end(), std::back_inserter(higher_cards), [&](auto const &c){return c >= sec_card;});
                    if (higher_cards.size() > 0)
                        return higher_cards;
                    else 
                        return suit_cards;
                } else {
                    return hand;
                }
            }
        }

    } else if (currentTrick.size() == 3){
        auto winner = currentTrick.begin();
        int ind = 0;
        int ind_winner = 0;
        bool bought = false;
        for (auto p = winner; p < currentTrick.end(); ++p) {
            if (winner->second.suit == trump){
                if (p->second >= winner->second) {
                    winner = p;
                    ind_winner = ind;
                    bought = true;
                }

            } else {
                if ((p->second >= winner->second) || (p->second.suit == trump))
                    winner = p;
                    ind_winner = ind;
            }
            ind += 1;
        }
        auto const first_card = currentTrick.front().second;
        auto const winner_card = currentTrick[ind_winner].second;
        if (winner_card.suit != trump) {
            std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
            if (suit_cards.size() > 0){
                std::copy_if(suit_cards.begin(), suit_cards.end(), std::back_inserter(higher_cards), [&](auto const &c){return c >= winner_card;});
                if (higher_cards.size() > 0)
                    return higher_cards;
                else
                    return suit_cards;
            } else {
                if (ind_winner == 0 || ind_winner == 2) {
                    std::copy_if(hand.begin(), hand.end(), std::back_inserter(trump_cards), [&](auto const &c){return c.suit == trump;});
                    if (trump_cards.size() > 0)
                        return trump_cards;
                    else 
                        return hand;
                } else {
                    return hand;
                }
            }
        } 
        if (first_card.suit == trump) {
            std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
            if (suit_cards.size() > 0) {
                if (ind_winner == 0 || ind_winner == 2) {
                    std::copy_if(suit_cards.begin(), suit_cards.end(), std::back_inserter(higher_cards), [&](auto const &c){return c >= winner_card;});
                    if (higher_cards.size() > 0)
                        return higher_cards;
                    else 
                        return suit_cards;
                } else {
                    return suit_cards;
                }
            } else {
                return hand;
            }
        }
        if (winner_card.suit == trump) {
            if (ind_winner == 0 || ind_winner == 2) {
                std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
                if (suit_cards.size() > 0)
                    return suit_cards;
                else
                {
                    std::copy_if(hand.begin(), hand.end(), std::back_inserter(higher_trump_cards), [&](auto const &c){return c >= winner_card;});
                    if (higher_trump_cards.size() > 0)
                        return higher_trump_cards;
                    else {
                        std::copy_if(hand.begin(), hand.end(), std::back_inserter(other_cards), [&](auto const &c){return c.suit != trump;});
                        if (other_cards.size() > 0)
                            return other_cards;
                        else
                            return hand;
                    }
                }
                
            } else {
                std::copy_if(hand.begin(), hand.end(), std::back_inserter(suit_cards), [&](auto const &c){return c.suit == first_card.suit;});
                if (suit_cards.size() > 0)
                    return suit_cards;
                else 
                    return hand;                
            }
        }
    }
    return playerCards[current_player];

}

void ManilGame::doMove(Manil::Card const move)
{
    updatePossibleCards(move);
    currentTrick.emplace_back(current_player, move);
    auto &hand = playerCards[current_player];
    auto const pos = std::find(hand.begin(), hand.end(), move);
    if (pos < hand.end())
        hand.erase(pos);
    else 
        throw std::out_of_range("pos");
    current_player = (current_player + 1) % player_num;
    if (currentTrick.size() == player_num)
        finishTrick();
    if (playerCards[current_player].empty())
        finishGame();
    return;
}

void ManilGame::doAIMove() {
    std::size_t x = 1000;
    ISMCTS::SOSolver<Manil::Card, ISMCTS::RootParallel> solver {20000, 12};
    this->doMove(solver(*this));
}

void ManilGame::doRandomMove() {
    std::vector<Manil::Card> valids = validMoves();
    int ind = std::rand() % valids.size();
    doMove(valids[ind]);
}

double ManilGame::getResult(ManilGame::Player p) const
{
    return points[p];
}

void ManilGame::deal() {
    dealer.shuffle();
    auto pos = dealer.deck.begin();
    for (auto p: m_players) {
        auto &hand = playerCards[p];
        std::copy_n(pos, 8, std::back_inserter(hand));
        pos += 8;
    }
    unknownCards.clear();
}

void ManilGame::finishTrick() {
    double point = 0;
    ManilGame::Play winner = judge();
    for (ManilGame::Play p : currentTrick) {
        point += p.second.get_value();
    }

    points[winner.first] += point;
    points[(winner.first + 2) % player_num] += point;
    currentTrick.clear();
    current_player = winner.first;
    return;
}

void ManilGame::finishGame() {
    possCards.clear();
}

void ManilGame::initPossibleCards() {
    possCards.resize(4, std::vector<std::vector<Manil::Card>>(4, std::vector<Manil::Card>(32)));
    for (Player p: m_players){
        for (Player q: m_players) {
            auto pos = or_deck.begin();
            auto cards = possCards[p][q].begin();
            std::copy_n(pos, 32, cards);
            auto &hand = playerCards[p];
            for (Manil::Card c : hand) {
                auto const pos = std::find(cards, cards+32, c);
                possCards[p][q].erase(pos);
            }
        }
    }
}

void ManilGame::updatePossibleCards(Manil::Card card) {
    unsigned int player = current_player;
    std::vector<Manil::Card> to_remove;
    to_remove.reserve(32);
    Hand suit_cards;
    Hand higher_cards;
    if (currentTrick.size() == 0) {
        to_remove.push_back(card);
    } else if (currentTrick.size() == 1) {
        to_remove.push_back(card);
        Manil::Card first_card = currentTrick[0].second;
        if (card.suit != first_card.suit){
            std::copy_if(or_deck.begin(), or_deck.end(), std::back_inserter(to_remove), [&](auto const &c){return c.suit == first_card.suit;});
        } else{
            if (card <= first_card) {
                std::copy_if(or_deck.begin(), or_deck.end(), std::back_inserter(to_remove), [&](auto const &c){return c >= first_card;});
            }
        }
    } else if (currentTrick.size() == 2) {
        to_remove.push_back(card);
        Manil::Card first_card = currentTrick[0].second;
        Manil::Card second_card = currentTrick[1].second;
        if (card.suit != first_card.suit){
            std::copy_if(or_deck.begin(), or_deck.end(), std::back_inserter(to_remove), [&](auto const &c){return c.suit == first_card.suit;});
        } else {

        }
    } else if (currentTrick.size() == 3){
        to_remove.push_back(card);
        Manil::Card first_card = currentTrick[0].second;
        if (card.suit != first_card.suit){
            std::copy_if(or_deck.begin(), or_deck.end(), std::back_inserter(to_remove), [&](auto const &c){return c.suit == first_card.suit;});
        } else {

        }
    }
    for (Player p: m_players) {
        for (Player q: m_players) {
            if (p == q)
                continue;
            auto const pos2 = std::find(possCards[p][q].begin(), possCards[p][q].end(), card);
            if (pos2 < possCards[p][q].end())
                possCards[p][q].erase(pos2);
        }
    }
    for (Player p: m_players){
        if (p == player)
            continue;
        for (Manil::Card c : to_remove) {
            auto const pos = std::find(possCards[p][player].begin(), possCards[p][player].end(), c);
            if (pos < possCards[p][player].end())
                possCards[p][player].erase(pos);
        }
    }
    to_remove.clear();
}

ManilGame::Play ManilGame::judge() {
    auto winner = currentTrick.begin();
    for (auto p = winner + 1; p < currentTrick.end(); ++p) {
        if (winner->second.suit == trump){
            if (p->second >= winner->second)
                winner = p;
        } else {
            if ((p->second >= winner->second) || (p->second.suit == trump))
                winner = p;
        }
    }
    return *winner;
}



ManilGame::~ManilGame() {};
    
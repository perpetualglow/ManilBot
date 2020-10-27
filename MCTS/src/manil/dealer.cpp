#include <iostream>
#include <algorithm>
#include <random>
#include "../include/dealer.hpp"
#include "../include/card.hpp"
#include "../include/utility.h"


namespace Manil {
    std::vector<Card> Dealer::init() {
        deck.clear();
        for (int i=0; i < 4; i++) {
            for (int j=0; j < 8; j++) {
                deck.push_back(Card(j,i));
            }
        }
        return deck;
    }

    std::vector<Card> Dealer::shuffle() {
        std::shuffle(deck.begin(), deck.end(), ISMCTS::prng());
        return deck;
    }
}
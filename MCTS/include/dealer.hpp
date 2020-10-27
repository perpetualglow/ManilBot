#ifndef DEALER_H
#define DEALER_H
#include "card.hpp"
#include <vector>


namespace Manil {
    class Dealer {
        public:
            std::vector<Card> deck;
            std::vector<Card> init();
            std::vector<Card> shuffle();
    };
}


#endif
#ifndef PLAYER_H
#define PLAYER_H

#include <vector>
#include "card.hpp"

namespace Manil {
    class Player{
        protected:
            std::vector<Card> hand;
        public:
            int id;
            void deal(std::vector<Card> hand);
            std::vector<Card> get_hand();
            void play_card(Manil::Card);
    };
}

#endif
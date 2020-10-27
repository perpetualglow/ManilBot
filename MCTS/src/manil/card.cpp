#include <iostream>
#include "../include/card.hpp"

int SUITS [] = {0,1,2,3};
int NUMBERS [] = {0,1,2,3,4,5,6,7};
int VALUES [] = {0,0,0,1,2,3,4,5};

std::string SUITS_TEXT [] = {"spades", "hearts", "diamons", "clubs"};
std::string LETTERS_TEXT [] = {"7", "8", "9", "J", "Q", "K", "A", "10"};

namespace Manil {
    int Card:: get_value() {
        return VALUES[rank];
    }

    std::ostream& operator<<(std::ostream& output, const Card& c){
        if (c.rank == -1)
            output << "NULL";
        else
            output << LETTERS_TEXT[c.rank] << ' ' << SUITS_TEXT[c.suit];
        return output;
    }
}


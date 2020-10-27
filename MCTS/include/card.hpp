#ifndef CARD_H
#define CARD_H

#include <ostream>

namespace Manil {
    struct Card{
        int suit, rank;
        constexpr Card()
            : rank(-1), suit(-1)
        {}
        constexpr Card(int r, int s)
            : rank(r), suit(s) 
        {}
        int get_rank() { return rank; }
        int get_suit() { return suit; }
        int get_value();
        explicit operator int() const {
            return rank + 8 * suit;
        }

        bool operator==(Card const &other) const
        {
            return rank == other.rank && suit == other.suit;
        }

        bool operator != (Card const &other) const
        {
            return !(*this==other);
        }

        bool operator > (Card const& other) const
        {
            return rank > other.rank;
        }

        bool operator >= (Card const& other) const
        {
            return rank >= other.rank && suit == other.suit;
        }

        bool operator <= (Card const& other) const
        {
            return rank <= other.rank && suit == other.suit;
        }

        operator std::string() const
        {
            return "ahaa";
        }
        
        friend std::ostream& operator<<(std::ostream& os, const Card& x);

    };
inline bool operator<(Card const &a, Card const &b) 
{
    return int(a) < int(b);
}
}


#endif
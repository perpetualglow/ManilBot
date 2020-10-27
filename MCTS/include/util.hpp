#include <random>

#include "card.hpp"

Manil::Card index_to_card(int ind) {
    return Manil::Card(ind%8, ind/8);
}
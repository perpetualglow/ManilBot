#include <iostream>
#include "include/card.hpp"
#include "include/dealer.hpp"
#include "include/util.hpp"
#include "include/sosolver.h"
#include "include/manil.hpp"

using namespace Manil;

int main() 
{
    unsigned int uns = 4;
    double avg = 0;
    std::vector<double> points;
    std::vector<double> total;
    for (int i =0; i < 100; i++) {
        ManilGame *y = new ManilGame(uns);
        points = y->run_game();
        total.push_back(points[0]);
        for (double p : total)
            avg += p;
        avg /= total.size();
        std::cout << "Run Game "<< i << " points: " << avg << std::endl;
        avg = 0;
        delete y;
    }
    
    std::cout << avg << std::endl;
    return 0;
}
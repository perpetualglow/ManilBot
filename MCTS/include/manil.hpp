#ifndef MANILGAME_H
#define MANILGAME_H

#include "game.h"
#include <vector>
#include "card.hpp"
#include "dealer.hpp"
#include "sosolver.h"
#include "execution.h"

class ManilGame : public ISMCTS::POMGame<Manil::Card>
{
    public:
        explicit ManilGame();
        explicit ManilGame(unsigned players = 4);
        virtual Clone cloneAndRandomise(Player observer) const override;
        virtual Player currentPlayer() const override;
        virtual std::vector<Player> players() const override;
        virtual std::vector<Manil::Card> validMoves() const override;
        virtual void doMove(Manil::Card const move) override;
        virtual double getResult(Player player) const override;
        virtual ~ManilGame();
        std::vector<double> run_game();
        void init();

    public:
        using Hand = std::vector<Manil::Card>;
        using Play = std::pair<Player, Manil::Card>;
        std::vector<Player> m_players;
        std::vector<Manil::Card> unknownCards;
        std::vector<Hand> playerCards;
        std::vector<Play> currentTrick;
        std::vector<Manil::Card> deck {32};
        std::vector<Manil::Card> or_deck {32};
        std::vector<double> points;
        std::vector<std::vector<std::vector<Manil::Card>>> possCards;
        Manil::Dealer dealer;
        int current_player;
        int player_num;
        int tricksleft;
        int trump;
        int total_clones_called;

        void finishTrick();
        void finishGame();
        void deal();
        void doAIMove();
        void doRandomMove();
        void initPossibleCards();
        void updatePossibleCards(Manil::Card card);
        Play judge();
};


#endif
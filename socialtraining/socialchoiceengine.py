__author__ = 'Matheus Alves'

from ballotbox.ballot import BallotBox
from ballotbox.singlewinner.preferential.borda import BordaVoting
from ballotbox.singlewinner.plurality import FirstPastPostVoting
from ballotbox.singlewinner.preferential.condorcet import KemenyYoungVoting
from ballotbox.singlewinner.preferential.condorcet import CopelandVoting

import socialtraining

class Borda:
    def __init__(self):

        self.ballot_box = BallotBox(method=BordaVoting, mode="standard")

    def get_winner(self):

        return self.ballot_box.get_winner()[0]

    def add_vote(self, preferences):

        self.ballot_box.add_vote(preferences)

class Plurality:
    def __init__(self):

        self.ballot_box = BallotBox(method=FirstPastPostVoting)

    def get_winner(self):

        winner = self.ballot_box.get_winner()

        ranking = eval (winner[0][1])

        list_ranking = sorted([(position, name) for name, position in
                               ranking.items()], reverse=True)

        return list_ranking

    def add_vote(self, preferences):

        self.ballot_box.add_vote(preferences)

class Copeland:
    def __init__(self):

        self.ballot_box = BallotBox(method=CopelandVoting)
        self.rounds = list()

    def get_winner(self):

        return self.ballot_box.get_winner(self.rounds)[0]

    def add_vote(self, preferences):

        round = BallotBox(method=FirstPastPostVoting)

        list_preferences = [(str(name), position) for name, position in
                            preferences.items()]

        round.batch_votes(list_preferences)

        self.rounds.append(round)

class _SocialChoiceFactory:
    def create_engine(social_choice_function):
        if social_choice_function == \
                socialtraining.SocialChoiceFunctionTypes.borda:
            return (Borda(), 'Borda Count')
        elif social_choice_function == \
                socialtraining.SocialChoiceFunctionTypes.plurality:
            return (Plurality(), 'Plurality')
        elif social_choice_function == \
                socialtraining.SocialChoiceFunctionTypes.copeland:
            return (Copeland(), 'Copeland')

class SocialChoiceEngine:
    def __init__(self, social_choice_function):

        factory_result = _SocialChoiceFactory.create_engine(
            social_choice_function)

        self._engine = factory_result[0]
        self._method = factory_result[1]

    def apply_social_choice_function(self, rankings):

        self._extract_votes_from_rankings(rankings)

        return self._engine.get_winner()

    def _extract_votes_from_rankings(self, rankings):

        for classifier, ranking in rankings.items():
            preferences = dict()

            preference = 1
            for position in range(0, len(ranking)):
                preferences[ranking[position][0]] = preference
                preference += 1

            self._engine.add_vote(preferences)

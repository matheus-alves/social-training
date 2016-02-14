__author__ = 'Matheus Alves'

from ballotbox.ballot import BallotBox
from ballotbox.singlewinner.preferential.borda import BordaVoting

import socialtraining

class _SocialChoiceFactory:
    def create_engine(social_choice_function):
        if social_choice_function == \
                socialtraining.SocialChoiceFunctionTypes.borda:
            return (BallotBox(method=BordaVoting, mode="standard"),
                    'Borda Count')

class SocialChoiceEngine:
    def __init__(self, social_choice_function):

        factory_result = _SocialChoiceFactory.create_engine(
            social_choice_function)

        self._engine = factory_result[0]
        self._method = factory_result[1]

    def apply_social_choice_function(self, rankings):

        self._extract_votes_from_rankings(rankings)

        return self._engine.get_winner()[0]

    def _extract_votes_from_rankings(self, rankings):

        for classifier in rankings:
            ranking = rankings[classifier]
            preferences = dict()

            preference = 1
            for position in range(0, len(ranking)):
                preferences[ranking[position][0]] = preference
                preference += 1

            self._engine.add_vote(preferences)

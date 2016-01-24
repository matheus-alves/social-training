__author__ = 'Matheus Alves'

class Classifier:
    def __init__(self, id, algorithm):
        self._id = id
        self._algorithm = algorithm

    def __str__(self):
        return self._id

    def train(self, instances, labels):
        self._algorithm.fit(instances, labels)

    def classify(self, instances):
        return self._algorithm.predict(instances)

    def define_ranking(self, instances):
        ranking = []
        for position in range(0, len(instances)):
            prob = self._algorithm.predict_proba(instances[position])
            ranking.append((position, prob[0][0]))

        ranking.sort(key=lambda tup: tup[1], reverse=True)

        return ranking

    def define_class_ranking(self, instances):
        # TODO
        pass

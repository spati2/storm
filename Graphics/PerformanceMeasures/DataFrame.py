class ProblemFrame():
    def __init__(self, problem, algorithms):
        self.problem = problem
        self.algorithms = algorithms
        self.data = []
        self.get_data()

    def get_data(self):
        self.data = [AlgorithmFrame(self.problem, algorithm) for algorithm in self.algorithms]

    def get_frontiers(self, number=-1):
        """
        :param number: Front Number
        :return: List of all the solutions across all algorithms and repeats
        """
        assert(len(self.data) != 0), "The frame was not initialized properly"
        return [item for d in self.data for item in d.get_frontiers(number)]

    def get_extreme_points(self, number=-1):
        points = self.get_frontiers(number)
        objectives = [point.objectives for point in points]
        need to implement the furthest points
        import pdb
        pdb.set_trace()



class AlgorithmFrame():
    def __init__(self, problem, algorithm):
        self.problem = problem
        self.algorithm = algorithm
        self.foldername =  "./Population_Archives/" + algorithm.name + "_" + problem.name + "/"
        self.repeats = None
        self.get_repeat_data()

    def get_repeat_data(self):
        import os
        subdirs = [self.foldername + d for d in os.listdir(self.foldername) if os.path.isdir(self.foldername + d)]
        self.repeats = [RepeatFrame(self.problem, subdir) for subdir in subdirs]

    def get_frontiers(self, number):
        return [item for repeat in self.repeats for item in repeat.get_frontier(number)]


class RepeatFrame():
    def __init__(self, problem, folder_name):
        self.problem = problem
        self.foldername = folder_name
        self.generations = []
        self.get_generation_data()

    def get_generation_data(self):
        from os import listdir
        from os.path import isfile, join
        files = [join(self.foldername,f) for f in listdir(self.foldername) if isfile(join(self.foldername,f))]
        self.generations = [GenerationFrame(self.problem, file) for file in files]

    def get_frontier(self, number):
        return self.generations[number].solutions


class GenerationFrame():
    def __init__(self, problem, filename):
        self.generation_number = filename.split("/")[-1]
        self.filename = filename
        self.problem = problem
        self.solutions = []
        self.get_data()

    def get_data(self):
        number_of_decisions = len(self.problem.decisions)
        for line in open(self.filename).readlines():
            content = map(float, line.split(","))
            self.solutions.append(SolutionFrame(content[:number_of_decisions], content[number_of_decisions:]))



class SolutionFrame():
    def __init__(self, decisions, objectives):
        self.decisions = decisions
        self.objectives = objectives

    def __repr__(self):
        return "|".join(map(str, self.decisions + self.objectives))

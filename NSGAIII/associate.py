from __future__ import division

def perpendicular_distance(pointa, pointb):
    """

    :param pointa:  the line from the origin passes through point a
    :param pointb:
    :return:
    """
    def dotproduct(pointa, pointb):
        ret = 0
        for i, j in zip(pointa, pointb):
            ret += (i*j)
        return ret
    def magnitude(pointa):
        sum = 0
        for i in pointa:
            sum += i ** 2
        return sum ** 0.5
    mag = dotproduct(pointa, pointb)/(magnitude(pointa))
    lengthb = magnitude(pointb) # hypotenuse
    base = mag
    return (lengthb ** 2 - base ** 2) ** 0.5

"""
Perpendicular distance between point a and point b, Adapted from Dr. Chiang's code
"""
def perpendicular_distance2(pointa, pointb):
    numerator = 0
    denominator = 0
    for d, p in zip(pointa, pointb):
        numerator += d * p
        denominator += d ** 2
    k = numerator/denominator
    d = sum([((k * pointa[i]) - pointb[i]) ** 2 for i in xrange(len(pointa))])
    return d ** 0.5


def associate(population, reference_points):
    """
    :param population:  list of jmoo_individuals
    :param reference_points: list of reference_points
    :return:
    """
    for individual in population:
        temp = []
        for point in reference_points:
            temp.append([point.id, perpendicular_distance(point.coordinates, individual.normalized)])

        # closed ref point
        nearest_ref_point = sorted(temp, key=lambda x: x[1])[0]

        individual.closest_ref = nearest_ref_point[0]
        individual.closest_ref_dist = nearest_ref_point[1]
    return population


if __name__ == "__main__":
    a = [1, -2, 3]
    b = [2, 4, 3]
    print perpendicular_distance(a, b)
    print perpendicular_distance2(a, b)
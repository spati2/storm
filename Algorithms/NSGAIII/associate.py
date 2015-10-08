from __future__ import division

# def perpendicular_distance(pointa, pointb):
#     """
#     Perpendicular distance between point a and point b
#     :param pointa:  the line from the origin passes through point a
#     :param pointb:
#     :return:
#     """
#     def dotproduct(pointa, pointb):
#         ret = 0
#         for i, j in zip(pointa, pointb):
#             ret += (i*j)
#         return ret
#     def magnitude(pointa):
#         sum = 0
#         for i in pointa:
#             sum += i ** 2
#         return sum ** 0.5
#     mag = dotproduct(pointa, pointb)/(magnitude(pointa))
#     lengthb = magnitude(pointb) # hypotenuse
#     if (lengthb ** 2 - mag ** 2) <= 0:
#         print
#         print pointa
#         print pointb
#         print lengthb, mag
#     assert((lengthb ** 2 - mag ** 2) > 0), "Something is wrong!"
#     base = mag
#     return (lengthb ** 2 - base ** 2) ** 0.5

def perpendicular_distance(d, z):
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

    temp = (dotproduct(d, z) / (magnitude(d) * magnitude(z))) ** 2  # Dr. Deb's formula
    distance = magnitude(z) * ((1 - temp) ** 0.5)
    if distance < 0:
        print d, magnitude(d)
        print z, magnitude(z)
        print "dot product: ", dotproduct(d, z)
        print "blah: ", (magnitude(d) * magnitude(z))
        print "temp: ", (dotproduct(d, z) / (magnitude(d) * magnitude(z))) ** 2
        print distance
    assert(distance >= 0), "Distance can't be less than 0"
    return distance


def perpendicular_distance2(pointa, pointb):
    """
    Perpendicular distance between point a and point b, Adapted from Dr. Chiang's code
    """
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
    temp_ref_points = []
    temp_population = []
    for rf in reference_points: temp_ref_points.extend(rf.coordinates)
    for indi in population: temp_population.extend(indi.normalized)

    # Check if the reference coordinates are between 0 and 1
    assert(reduce(lambda x, y: (x and y), map(lambda x: (x >= 0.0) and (x <= 1.0), temp_ref_points)) is True), "Inequality not satisfied for reference points"
    # Check if the normalized points are between 0 and 1
    assert(reduce(lambda x, y: (x and y), map(lambda x: (x >= 0.0) and (x <= 1.0), temp_population))
           is True), "Inequality not satisfied for normalized fitness"

    for individual in population:
        temp_perpendicular_distance = []
        for point in reference_points:
            temp_perpendicular_distance.append([point.id,
                                                perpendicular_distance(point.coordinates, individual.normalized)])
        temp_perpendicular_distance = sorted(temp_perpendicular_distance, key=lambda x: x[1])

        assert(temp_perpendicular_distance[0][1] <= temp_perpendicular_distance[0][1]), \
            "Perpendicular distance is wrong"

        # closed ref point
        nearest_ref_point = temp_perpendicular_distance[0]
        individual.closest_ref = nearest_ref_point[0]
        individual.closest_ref_dist = nearest_ref_point[1]

    return population


if __name__ == "__main__":
    a = [1, -2, 3]
    b = [2, 4, 3]
    print perpendicular_distance(a, b)
    print perpendicular_distance2(a, b)
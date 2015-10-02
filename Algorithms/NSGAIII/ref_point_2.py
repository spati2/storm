from __future__ import division


def generate_recursive(rps, point, num_of_obj, left, total, element):
    if element == num_of_obj -1:
        point[element] = round(left/total, 2)
        print point, sum(point)
        rps.append(point)
        return rps
    else:
        for i in xrange(left + 1):
            # print point
            point[element] = round(i/total, 2)
            rps = generate_recursive(rps, point, num_of_obj, left - i, total, element + 1)
        return rps
    return rps


def generate_reference_points(problem):
    if len(problem.objectives) > 5:
        p = [3, 2]
    else:
        p = [12, 6]
    point = [0 for _ in problem.objectives]
    rps = []
    rps = generate_recursive(rps, point, len(problem.objectives), p[0], p[1], 0)
    f = open("generated_points_2.txt", "w")
    for r in rps:
        for i in r:
            f.write(str(i) + " ")
        f.write("\n")
    f.close()
    print len(rps)
    exit()

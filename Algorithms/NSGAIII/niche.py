from __future__ import division
from random import choice

class NicheContainer:
    def __init__(self, id, count=0):
        self.id = id
        self.count = count
    def __str__(self):
        return str("id: " + str(self.id) + " count: " + str(self.count))

def niche_counting(count, population):
    niche_count = [NicheContainer(c) for c in xrange(count)]
    for individual in population:
        contained = niche_count[individual.closest_ref]
        assert (contained.id == individual.closest_ref), "They should be the same"
        contained.count += 1
    return niche_count

def niching(k, no_ref_points, population, last_level):
    # import time
    # file_name = "./tmp4/nicheb4" + str(time.clock()) + ".txt"
    # f = open(file_name, "w")
    # for pop in population:
    #     for a in pop.decisionValues:
    #         f.write(str(a) + " ")
    #     f.write("\n")
    # f.close()

    blah = len(population)
    remove_candidates = []
    while len(population) < blah + k:
        niche_count = niche_counting(no_ref_points, population)

        for i in remove_candidates:
            assert(niche_count[i].id == i), "They should be same"
            niche_count[i].id = i
            niche_count[i].count = 1e32  # so that they never are selected

        count_last_level = niche_counting(no_ref_points, last_level)

        temp_sort = sorted(niche_count, key=lambda z: z.count)

        candidates = [x for x in temp_sort if x.count == temp_sort[0].count]  # candidates with minimum count

        candidate = choice(candidates)
        assert(candidate.count == temp_sort[0].count), "They should be the same`"
        # print count_last_level[candidate.id]
        if count_last_level[candidate.id].count != 0:
            assert(count_last_level[candidate.id].id == candidate.id), "Something's wrong"
            closest_f_l_members = [can for can in last_level if can.closest_ref == candidate.id]
            # print len(closest_f_l_members), candidate.id
            if niche_count[candidate.id].count == 0:
                closest_f_l_member = sorted(closest_f_l_members, key=lambda x: x.closest_ref_dist)[0]
                population.append(closest_f_l_member)
                last_level.remove(closest_f_l_member)
            else:
                # print "#"
                closest_f_l_member = choice(closest_f_l_members)
                population.append(closest_f_l_member)
                last_level.remove(closest_f_l_member)
            niche_count[candidate.id].count += 1
        else:
            # print "$"
            remove_candidates.append(candidate.id)

    return population
#
# def niching(k, no_ref_points, population, last_level):
#
#

from __future__ import division
from random import choice

class NicheContainer:
    def __init__(self, id, count=0):
        self.id = id
        self.count = count
    def __str__(self):
        return str("id: " + str(self.id) + " count: " + str(self.count))

def niche_counting(count, population):
    """
    :param count: Number of reference points
    :param population: Length of the population excluding the final level (non dominated sorting)
    :return: List of reference points along with the count of points associated with them
    """
    niche_count = [NicheContainer(c) for c in xrange(count)]
    for individual in population:
        contained = niche_count[individual.closest_ref]
        assert (contained.id == individual.closest_ref), "They should be the same"
        contained.count += 1
    return niche_count


def niching(k, no_ref_points, population, last_level):
    """
    :param k: points to be generated using the niching operation
    :param no_ref_points: Number of reference points
    :param population: population excluding the final level (non dominated sorting)
    :param last_level: final level (non dominated sorting)
    :return: final population
    """

    blah = len(population)
    remove_candidates = []
    niche_count = niche_counting(no_ref_points, population)
    count_last_level = niche_counting(no_ref_points, last_level)
    while len(population) < blah + k:
        temp_sort = sorted(niche_count, key=lambda z: z.count)
        candidates = filter(lambda x: x.count == temp_sort[0].count, temp_sort)
        candidate = choice(candidates)
        assert(candidate.count == temp_sort[0].count), \
            "They should be the same, since the candidate count is supposed to have the minimum count value"
        import pdb
        # pdb.set_trace()
        # If the count(meaning that there is no associated P_(t+1) member to the reference point j)
        if candidate.count == 0:
            # There exists one or more members in front of F_l that are associated with  the reference point j.
            # In this case, the one having the shortest perpendicular distance from the reference line is added to P_(t+1)
            if count_last_level[candidate.id].count != 0:
                closest_f_l_members = [can for can in last_level if can.closest_ref == candidate.id]
                # pdb.set_trace()
                closest_f_l_member = sorted(closest_f_l_members, key=lambda x: x.closest_ref_dist)[0]
                # pdb.set_trace()
                population.append(closest_f_l_member)
                last_level.remove(closest_f_l_member)
                niche_count[candidate.id].count += 1
                count_last_level[candidate.id].count -= 1
            else: # count_last_level[candidate.id].count == 0
                niche_count[candidate.id].count = 1e32

        # In the event the minimum count is greater than equal to 1 (meaning that already one member associated with the
        # reference point exists in S_t\F_l), a randomly chosen member, if exists , from front F_l that is associated
        # with the reference point is added to the population
        else:
            closest_f_l_members = [can for can in last_level if can.closest_ref == candidate.id]
            if len(closest_f_l_members) != 0:
                closest_f_l_member = sorted(closest_f_l_members, key=lambda x: x.closest_ref_dist)[0]
                population.append(closest_f_l_member)
                last_level.remove(closest_f_l_member)
                niche_count[candidate.id].count += 1
            else:
                niche_count[candidate.id].count = 1e32


        # if count_last_level[candidate.id].count != 0:
        #     assert(count_last_level[candidate.id].id == candidate.id), "Something's wrong"
        #     closest_f_l_members = [can for can in last_level if can.closest_ref == candidate.id]
        #     # print len(closest_f_l_members), candidate.id
        #     if niche_count[candidate.id].count == 0:
        #         closest_f_l_member = sorted(closest_f_l_members, key=lambda x: x.closest_ref_dist)[0]
        #         population.append(closest_f_l_member)
        #         last_level.remove(closest_f_l_member)
        #     else:
        #         # print "#"
        #         closest_f_l_member = choice(closest_f_l_members)
        #         population.append(closest_f_l_member)
        #         last_level.remove(closest_f_l_member)
        #     niche_count[candidate.id].count += 1
        # else:
        #     # print "$"
        #     remove_candidates.append(candidate.id)

    return population


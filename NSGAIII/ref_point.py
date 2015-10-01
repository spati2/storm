from __future__ import division
class reference_point:
    def __init__(self, id, coordinates):
        self.id = id
        self.coordinates = coordinates

    def __str__(self):
        s = "id: " + str(self.id) + "\n"
        s += "coordinates: " + str(self.coordinates) + "\n"
        return s

def get_ref_points(root):
    ref_points = []
    assert(root.data == -1 and root.level == 0), "Supplied node is not root"
    visited, stack = set(), [root]
    count = 0
    while len(stack) != 0:
        vertex = stack.pop()
        if vertex not in visited:
            if len(vertex.children) == 0:
                temp = vertex
                points = []
                while temp is not None:
                    points = [round(temp.data, 5)] + points
                    temp = temp.parent
                ref_points.append(reference_point(count, points))
                count += 1
            stack.extend(vertex.children)
            visited.add(vertex)
    return ref_points

class Node(object):
    def __init__(self, data=-1, level=0, parent = None):
        self.data = data
        self.level = level
        self.children = []
        self.parent = parent

    def add_child(self, obj):
        self.children.append(obj)

def tree(node, n, p, level = 0):
    if level == 0:
        from numpy import arange
        for i in [j for j in arange(0, 1 + 10e-10, 1/p)]:
            node.add_child(Node(i, level + 1))
        for child in node.children:
            tree(child, n, p, level + 1)
    elif level < (n - 1):
        other_beta = 0

        # Traversing up the tree to get other values of beta
        temp = node
        while temp is not None:
            other_beta += temp.data
            temp = temp.parent

        k = (1 - other_beta)/(1/p)
        from numpy import arange
        for i in [j for j in arange(0, k * (1/p) + 10e-10, (1/p))]:
            node.add_child(Node(i, level + 1, node))
        for child in node.children:
            tree(child, n, p, level + 1)
    elif level == (n -1):
        other_beta = 0
        # Traversing up the tree to get other values of beta
        temp = node
        while temp is not None:
            other_beta += temp.data
            temp = temp.parent
        node.add_child(Node(1-other_beta, level + 1, node))

    else:
        return

def transform_points_id(start, population):
    for i, point in enumerate(population):
        point.id = start + i
    return population

def cover(n):
    from scipy.misc import comb
    if n <= 5:
        if n == 3: p = 12
        elif n == 5: p = 6
        root = Node(-1)
        tree(root, n, p)
        lst = get_ref_points(root)
        assert(len(lst) == comb(n + p - 1, p)), "Length of the lst should be equal to combination"
    else:
        lst = []
        combination = 0
        if n == 8: P = [3,2]
        elif n == 10: P = [3, 2]
        elif n == 15: P = [2, 1]
        else:
            print "Not ready to handle this"
            exit()

        # generate outer reference points
        p = P[0]
        root = Node(-1)
        tree(root, n, p)
        lst.extend(get_ref_points(root))
        combination = int(comb(n + p - 1, p))
        assert(len(lst) == combination), "Length of the lst should be equal to combination"

        temp = []
        p = P[1]
        root = Node(-1)
        tree(root, n, p)
        temp.extend(get_ref_points(root))
        temp = transform_points_id(len(lst), temp)
        # checkout the new kdebs paper to change the reference point selection
        combination = int(comb(n + p - 1, p))
        assert(len(temp) == combination), "Length of the temp should be equal to combination"
        # center = 1/n
        points_inner_layer = len(temp)
        tau = 0.5
        for j, point in enumerate(temp):
            for i, obj in enumerate(point.coordinates):
                old = obj
                temp[j].coordinates[i] = round(((1 - tau)/points_inner_layer) + tau * obj, 3)
                assert(old != point.coordinates[i]), "something's wrong"
        lst.extend(temp)
    f = open("generation.txt", "w")
    for l in lst:
        for ll in l.coordinates:
            f.write(str(ll) + " ")
        f.write("\n")
    f.close()
    return lst

# -------------------------- Testing -------------------------- #

def _get_ref_points():
    for i in cover(15):
        for xx in i.coordinates:
            print xx,
        print


if __name__ == "__main__":
     _get_ref_points()
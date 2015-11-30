#    Copyright (C) 2010 Simon Wessing
#    TU Dortmund University
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Simon Wessing"


class HyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []


    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """

        def weaklyDominates(point, other):
            for i in xrange(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        for point in front:
            # only consider points that dominate the reference point
            if weaklyDominates(point, referencePoint):
                relevantPoints.append(point)
        print relevantPoints
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in xrange(len(relevantPoints)):
                relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in xrange(dimensions)]
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return round(hyperVolume, 4)


    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo != None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in xrange(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol


    def preProcess(self, front):
        """Sets up the list Data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in xrange(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList


    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort()
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]
            
            
            
class MultiList: 
    """A special Data structure needed by FonsecaHyperVolume.
    
    It consists of several doubly linked lists that share common nodes. So, 
    every node has multiple predecessors and successors, one in every list.

    """

    class Node: 

        def __init__(self, numberLists, cargo=None): 
            self.cargo = cargo 
            self.next  = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists
    
        def __str__(self): 
            return str(self.cargo)
        
        
    def __init__(self, numberLists):  
        """Constructor. 
        
        Builds 'numberLists' doubly linked lists.
        Vivek: numberLists means number of dimensions

        """
        self.numberLists = numberLists
        self.sentinel = MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists  
        
        
    def __str__(self):
        strings = []
        for i in xrange(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr
    
    
    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.numberLists
    
    
    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length
            
            
    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node
        
        
    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node
        
        
    def remove(self, node, index, bounds): 
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in xrange(index): 
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor  
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous 
        nodes of the node that is reinserted are in the list.

        """
        for i in xrange(index): 
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


def get_hyper_volume(reference_point, results):
    """Receives list of  lists"""

    HV = HyperVolume(reference_point)
    return HV.compute(results)

if __name__ == "__main__":

    # Example:
    referencePoint = [9999999929.03, 9999999900.0, 9999999902.56]
    hv = HyperVolume(referencePoint)
    front = [[9999999941.94, 9999999926.09, 9999999906.84], [9999999948.39, 9999999913.04, 9999999905.13], [9999999948.39, 9999999930.43, 9999999904.84], [9999999946.77, 9999999900.0, 9999999904.56], [9999999949.46, 9999999917.39, 9999999904.56], [9999999948.92, 9999999926.09, 9999999906.55], [9999999941.94, 9999999926.09, 9999999904.27], [9999999946.24, 9999999921.74, 9999999905.13], [9999999945.7, 9999999930.43, 9999999905.41], [9999999942.47, 9999999952.17, 9999999904.56], [9999999947.31, 9999999926.09, 9999999907.12], [9999999947.31, 9999999904.35, 9999999905.7], [9999999951.08, 9999999917.39, 9999999905.7], [9999999944.62, 9999999917.39, 9999999905.98], [9999999954.3, 9999999943.48, 9999999907.12], [9999999949.46, 9999999913.04, 9999999903.99], [9999999938.17, 9999999917.39, 9999999903.7], [9999999951.08, 9999999956.52, 9999999905.41], [9999999945.16, 9999999921.74, 9999999906.27], [9999999950.0, 9999999956.52, 9999999903.99], [9999999947.85, 9999999956.52, 9999999904.84], [9999999944.62, 9999999921.74, 9999999905.98], [9999999941.4, 9999999926.09, 9999999905.13], [9999999943.55, 9999999926.09, 9999999907.41], [9999999952.15, 9999999956.52, 9999999905.98], [9999999959.68, 9999999926.09, 9999999905.13], [9999999950.54, 9999999956.52, 9999999906.55], [9999999941.94, 9999999926.09, 9999999906.84], [9999999947.85, 9999999930.43, 9999999906.55], [9999999946.24, 9999999926.09, 9999999907.41], [9999999948.39, 9999999952.17, 9999999904.27], [9999999944.62, 9999999926.09, 9999999903.99], [9999999945.16, 9999999926.09, 9999999905.7], [9999999958.06, 9999999926.09, 9999999906.55], [9999999951.61, 9999999921.74, 9999999906.27], [9999999946.24, 9999999908.7, 9999999903.99], [9999999952.69, 9999999952.17, 9999999908.26], [9999999943.55, 9999999926.09, 9999999906.84], [9999999951.61, 9999999921.74, 9999999904.27], [9999999946.77, 9999999921.74, 9999999906.84], [9999999946.24, 9999999917.39, 9999999906.84], [9999999945.7, 9999999956.52, 9999999906.84], [9999999941.4, 9999999908.7, 9999999904.84], [9999999950.0, 9999999952.17, 9999999907.12], [9999999959.14, 9999999908.7, 9999999906.55], [9999999939.78, 9999999930.43, 9999999905.98], [9999999953.76, 9999999952.17, 9999999904.84], [9999999940.32, 9999999921.74, 9999999905.13], [9999999940.32, 9999999904.35, 9999999905.41], [9999999953.76, 9999999947.83, 9999999905.98], [9999999942.47, 9999999921.74, 9999999905.98], [9999999942.47, 9999999930.43, 9999999904.84], [9999999938.17, 9999999926.09, 9999999905.7], [9999999955.38, 9999999956.52, 9999999905.13], [9999999951.61, 9999999921.74, 9999999906.55], [9999999952.15, 9999999921.74, 9999999907.12], [9999999945.7, 9999999921.74, 9999999906.84], [9999999938.71, 9999999926.09, 9999999905.13]]
    volume = hv.compute(front)
    print volume






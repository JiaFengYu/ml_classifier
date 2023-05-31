from Geometry import Point, Rect

class kdtree(object):
    '''A K-dimensional tree or K-d tree is a structure used to
    speed up searches for points in a k-dimensional space. It
    is like a binary tree, except that each 'level' of the tree
    splits on a different dimension.
    '''
    
    class node(object):
        '''A simple node of K-dimensional points for a K-d tree.
        Contains the following attributes:
        pt - a Point object
        rc_left - Rect object that defines an axis-aligned rectangle
        enclosing the 'left' child nodes.
        rc_right - Rect object that defines an axis-aligned rectangle
        enclosing the 'right' child nodes.'
        nd_left - Reference to the left child.
        nd_right - Reference to the right child.
        count - The number of nodes in this subtree.
        '''
        def __init__(self, pt, rc, dim):
            '''Initialize a K-d tree node.'''
            self.pt = pt
            pt_left = rc.pt_max[:dim] + (pt[dim],) + rc.pt_max[dim+1:]
            pt_right = rc.pt_min[:dim] + (pt[dim],) + rc.pt_min[dim+1:]
            self.rc_left = Rect(rc.pt_min, pt_left)
            self.rc_right = Rect(pt_right, rc.pt_max)
            self.nd_left = None
            self.nd_right = None
            self.count = 1
        
    def __init__(self, k):
        self.K = k
        self.nd_root = None
        self.rc_root = Rect(Point((0,) * k), Point((1,) * k))

    def __bool__(self):
        return self.nd_root != None

    def __len__(self):
        return self.__size(self.nd_root)

    def __size(self, node):
        return node.count if node else 0

    def insert(self, pt):
        '''Inserts the point into the K-d tree.
        Uses a recursive algorithm very similar to that used by
        the BST. However, it adds the detail that it "splits" by
        either the x or y coordinate depending on the depth of the
        tree.
        In either case, if the coordinate value is less, the insertion
        moves to the left. If it is not equal to the current point,
        it moves to the right.'''

        def __insert(node, pt, rc, depth):
            # Compute the dimension we will use
            # 0 -> x, 1 -> y
            dim = depth % self.K
            if node == None:
                return kdtree.node(pt, rc, dim)
            if pt[dim] < node.pt[dim]:
                node.nd_left = __insert(node.nd_left, pt, node.rc_left, depth + 1)
            elif pt != node.pt:
                node.nd_right = __insert(node.nd_right, pt, node.rc_right, depth + 1)
            node.count = 1 + self.__size(node.nd_left) + self.__size(node.nd_right)
            return node

        if pt == None:
            raise ValueError("Argument is null.")
        self.nd_root = __insert(self.nd_root, pt, self.rc_root, 0)

    def contains(self, pt):
        '''Determine if the point 'pt' is contained within the
        K-d tree. Uses an iterative approach.'''
        if pt == None:
            raise ValueError("Illegal argument")
        depth = 0
        node = self.nd_root
        while node != None:
            dim = depth % self.K
            if pt[dim] < node.pt[dim]:
                node = node.nd_left
            elif pt != node.pt:
                node = node.nd_right
            else:
                return True
            depth += 1
        return False

    def range(self, rect):
        '''Return a list of all of the Point objects inside
       the rectangle.'''
        def __range(node, rect, result):
            if not node:
                return
            if rect.intersects(node.rc_right):
                __range(node.nd_right, rect, result)
            if rect.contains(node.pt):
                result.append(node.pt)
            if rect.intersects(node.rc_left):
                __range(node.nd_left, rect, result)
        if not rect:
            raise ValueError("Illegal argument")
        result = []
        __range(self.nd_root, rect, result)
        return result

    def nearest(self, pt):
        '''Find the saved point nearest the query point 'pt'.'''
        def search(node, depth):
            if node == None:
                return
            
            nonlocal search_point, search_distance

            distance = node.pt.distance(pt)
            if distance < search_distance:
                search_point = node.pt
                search_distance = distance

            dim = depth % self.K
            if pt[dim] < node.pt[dim]:
                search(node.nd_left, depth + 1)
                distance = node.rc_right.distance(pt)
                if distance < search_distance:
                    search(node.nd_right, depth + 1)
            else:
                search(node.nd_right, depth + 1)
                distance = node.rc_left.distance(pt)
                if distance < search_distance:
                    search(node.nd_left, depth + 1)
                    
        if not pt:
            raise ValueError("Illegal argument")
        search_point = None
        search_distance = float('inf')
        search(self.nd_root, 0)
        return search_point
        
    def k_nearest(self, pt, k):
        '''Find the 'k' saved points nearest the query point 'pt'.'''
        def search(node, depth):
            if node == None:
                return
            
            nonlocal search_radius

            if node.rc_left.distance(pt) < search_radius or len(search_points) < k:
                search(node.nd_left, depth + 1)
            
            distance = node.pt.distance(pt)
            if distance < search_radius or len(search_points) < k:
                i = 0
                for i in range(len(search_points)):
                    if distance < search_points[i].distance(pt):
                        break
                search_points.insert(i, node.pt)
                if len(search_points) > k:
                    search_points.pop()
                search_radius = search_points[-1].distance(pt)

            if node.rc_right.distance(pt) < search_radius or len(search_points) < k:
                search(node.nd_right, depth + 1)

        if not pt:
            raise ValueError("Illegal argument")
        if k == 1:
            return [self.nearest(pt)]
        search_points = []
        search_radius = float('inf')
        search(self.nd_root, 0)
        return search_points
        
    def draw(self, canvas):
        def __draw(canvas, node, depth):
            if node == None:
                return
            __draw(canvas, node.nd_left, depth + 1)
            node.pt.draw(canvas)
            if (depth % self.K) == 0:
                canvas.create_line(node.pt[0], node.rc_left.pt_min[1],
                                   node.pt[0], node.rc_left.pt_max[1],
                                   fill="red")
            else:
                canvas.create_line(node.rc_right.pt_min[0], node.pt[1],
                                   node.rc_right.pt_max[0], node.pt[1],
                                   fill="blue")

            __draw(canvas, node.nd_right, depth + 1)

        __draw(canvas, self.nd_root, 0)


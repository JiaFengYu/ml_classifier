from tkinter import Canvas
class ScaledCanvas(Canvas):
    '''This is a simple modification to the Tk Canvas widget that
    allows us to set a scaling parameter, so that we can use a simpler
    coordinate system (e.g. x and y range [0, 1] with the lower left
    corner corresponding to the origin).'''
    def __init__(self, master=None, cnf={}, **kwargs):
        super().__init__(master, cnf, **kwargs)
    
    def set_scaling(self, xscale, yscale):
        self.__xscale = xscale
        self.__yscale = yscale

    @staticmethod
    def scale(c0, c1, scale):
        if scale > 0:
            return (c0 * scale, c1 * scale)
        else:
            return ((c0 - 1.0) * scale, (c1 - 1.0) * scale)
        
    def create_oval(self, x0, y0, x1, y1, **kwargs):
        x0, x1 = ScaledCanvas.scale(x0, x1, self.__xscale)
        y0, y1 = ScaledCanvas.scale(y0, y1, self.__yscale)
        return super().create_oval(x0, y0, x1, y1, **kwargs)

    def create_line(self, x0, y0, x1, y1, **kwargs):
        x0, x1 = ScaledCanvas.scale(x0, x1, self.__xscale)
        y0, y1 = ScaledCanvas.scale(y0, y1, self.__yscale)
        return super().create_line(x0, y0, x1, y1, **kwargs)

class Point(object):
    def __init__(self, iterable):
        self.data = tuple(iterable)

    def __getitem__(self, index):
        '''Support indexing for points, so can write:
        p = Point((0, 1))
        x = p[0]
        '''
        return self.data[index]
    
    def draw(self, canvas, d=0.005):
        return canvas.create_oval(self[0] - d, self[1] - d,
                                  self[0] + d, self[1] + d, fill="black")
        
    def distance(self, other):
        '''Get the distance (actually the squared distance) from this
        point to another point.'''
        r = 0
        for x1, x2 in zip(self, other):
            delta = x1 - x2
            r += delta * delta
        return r
    
    def __repr__(self):
        return str(self.data)

class Rect(object):
    '''Class that represents an n-dimensional, axis-aligned rectangle.'''
    def __init__(self, pt_min, pt_max):
        self.pt_min = pt_min
        self.pt_max = pt_max

    def contains(self, pt):
        '''Returns True if this rectangle contains this point.'''
        return (all(x >= y for x,y in zip(pt, self.pt_min)) and 
                all(x <= y for x,y in zip(pt, self.pt_max)))
        
    def intersects(self, other):
        '''Returns True if this rectangle overlaps other.'''
        return (all(x >= y for x, y in zip(self.pt_max, other.pt_min)) and
                all(x >= y for x, y in zip(other.pt_max, self.pt_min)))
    
    def distance(self, pt):
        '''Compute the shortest distance from the Point 'pt' to
        this rectangle. We use a common trick, which is to compute
        the distance squared, which we can treat just like the "real"
        distance, but it saves the time of computing the square root.'''
        r = 0
        for p, n, x in zip(pt, self.pt_min, self.pt_max):
            delta = max(n - p, 0, p - x)
            r += delta * delta
        return r
        

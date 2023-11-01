#!/usr/bin/python

class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y



def orientation(p, q, r):
    ## See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
    ## for details of below formula. 
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
  
    if val == 0: return 0  ## colinear :
    elif val > 0: return 1 #clock wise
    else: return 2          # counterclock wise 


#def intersect(A,B,C,D):
#	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def intersect(p1,q1,p2,q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    ##General case
    if (o1 != o2 and o3 != o4):
        return True; 
    
    onSegment = lambda p, q, r: q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)
    
    ## Special Cases 
    ## p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    if o1 == 0 and onSegment(p1, p2, q1): return True
  
    ## p1, q1 and q2 are colinear and q2 lies on segment p1q1 
    if o2 == 0 and onSegment(p1, q2, q1): return True
  
    ## p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    if o3 == 0 and onSegment(p2, p1, q2): return True
  
    ## p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    if o4 == 0 and onSegment(p2, q1, q2): return True
  
    return False ## Doesn't fall in any of the above cases 

if __name__ == '__main__':
    
    ## Driver program to test above functions 

    
    a = Point(0,0)
    b = Point(0,1)
    c = Point(1,1)
    d = Point(1,0)
    
    
    print(intersect(a,b,c,d))
    print(intersect(a,c,b,d))
    print(intersect(a,d,b,c))
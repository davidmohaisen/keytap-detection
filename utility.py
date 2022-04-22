import math, json, scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression, Ridge, HuberRegressor

#####################################################################################################
# PLANE FITTING
#####################################################################################################

# return normalized normal
def fit_plane_normal(positions_list, weights):
    data = positions_list

    # Take x and y coordinates from the data and put 1s for z coordinates.
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]

    X = A
    y = -1 * data[:,2] 
    
    # RANSAC
    # model_ransac = RANSACRegressor(LinearRegression(), residual_threshold=10, max_trials=10000) # min_samples=0.2, max_trials=1000)
    # model_ransac.fit(X,y, sample_weight=weights)

    # coeffs = model_ransac.estimator_.coef_
    # inlier_mask = model_ransac.inlier_mask_


    # HuberRegressor
    # model_ransac = HuberRegressor()#LinearRegression())
    # model_ransac.fit(X,y)

    # coeffs = model_ransac.coef_
    #inlier_mask = model_ransac.inlier_mask_

    model = Ridge(fit_intercept=False, normalize=True, solver='lsqr')

    # model = LinearRegression(fit_intercept=False, normalize=True, n_jobs = 4)
    model.fit(X,y, sample_weight=weights)

    coeffs = model.coef_

    inlier_mask = None


    return coeffs, inlier_mask

    # fit a linear plane to [x,y,1] - [z] pairs by minimizing square distance.
    # C includes coefficients A, B, D of the plane equation z = Ax + By + D
    # C,_,_,_ = scipy.linalg.lstsq(A, -1 * data[:,2])
    
    #return C

    """
    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    """

"""
with open('positions', 'rb') as pickle_file:
    content = pickle.load(pickle_file)

fit_plane(content)
"""


# returns normal, point
def fit_plane(points, weights):
    # Fit plane - find normal
    n, inlier_mask = fit_plane_normal(points, weights)

    # The plane equation is z = Ax + By + D
    A, B, D = n
    
    # x and y coordinates can be decided randomly, we selected (1,1) to calculate z coordinates.
    x, y = 1, 1
    
    # Calculate the z coordinate fits to the plane equation.
    z = -1 * ( A*x + B*y + D )

    # Create the list that holds the coordinates of the point.
    point = [x, y, z]
    
    # Z component of the normal can be either -1 or 1, since the original plane equation should be look like below:
    # Ax + By + Cz + D = 0 --> -Cz = Ax + By + D = 0
    # If we select C (which is the z component of the normal) as 1, the calculated z coordinate should be multiplied by -1.
    #         since the equation becomes -z = Ax + By + D.
    # If we select C as -1, the z component of the normal should be -1 and the equation becomes z = Ax + By + D.
    normal = [A, B, 1.0]

    normal = get_normalized_vec(normal)
    
    return normal, point, inlier_mask
    
    # # Previous implementation.
    
    # # plane equation: (p - p0).n = 0
    # # line equation : p = d.l + p0

    # # ref point is used as the beginning of the normal line
    # refp = points[0]

    # # find t for each point wrt ref point
    # t = []
    # for p in points:
    #     t.append( find_t_btw_points(p, refp, n) )

    # # find mean t
    # mean = statistics.mean(t)

    # # extract mean from ref point to find the plane point
    # # TODO: This could be improved through investigation of distribution and rejection
    # # of outliers
    # displacement = [mean * x for x in n]
    # refp = vec_sum(refp, displacement)

    # return n, refp



################################################################################
# Find point projection on plane

# given a plane and a point, find the project of the point on the plane in the
# direction of the plane normal

# returns a point
def find_point_projection_on_plane(plane_point, plane_normal, point_to_project):
    t = find_t_btw_points(point_to_project, plane_point, plane_normal)

    return vec_sum(point_to_project, [-t * x for x in plane_normal])


#####################################################################################################
# BASIS 
#####################################################################################################

# given two points and a normal vector that defines two different planes with
# those points, find the distance between planes in terms of a factor t. factor t
# should be used as a multiplier to normal vector to find the displacement from one
# point to another. (from refp to p)
def find_t_btw_points(p, refp, normal):
    # equation: t = ( (p - refp).normal ) / (normal.normal)
    t = inner_product( vec_diff(p, refp), normal ) / inner_product(normal, normal)
    return t


# generate orthonormal basis
# https://github.com/necipfazil/raytracer/blob/06ed8b14319374707f0ecdd4202d3204cad99927/source/geometry/impl/vector3.cpp#L214
def generate_orth_basis(vec):
    r = get_normalized_vec(vec)
    rPrime = r[:]

    # set minimum component of rPrime to 1
    rPrime[rPrime.index(min(rPrime))] = 1

    # set 0 components of rPrime to 1
    rPrime = [1 if x == 0 else x for x in rPrime]

    rPrime = get_normalized_vec(rPrime)

    # compute u, v, w
    u = get_normalized_vec( cross_product(rPrime, r) )
    v = get_normalized_vec( cross_product(u, r) )
    return u, v, r


def get_coordinates_about_basis(coords, basis):
    a = np.array( [ [x[0] for x in basis], [x[1] for x in basis], [x[2] for x in basis] ] )
    b = np.array( coords )
    x = np.linalg.solve(a, b)
    return x


def get_coordinates_back_from_basis(basis_coords, basis):
    c = basis_coords

    c0 = [c[0] * o for o in basis[0]]
    c1 = [c[1] * o for o in basis[1]]
    c2 = [c[2] * o for o in basis[2]]

    return [x+y+z for x,y,z in zip(c0,c1,c2)]

# adapter to use in map func
class Basis:
    def __init__(self, basis):
        self.basis = basis
    def get_coordinates_about_basis(self, coords):
        return get_coordinates_about_basis(coords, self.basis)
    def get_coordinates_back_from_basis(self, basis_coords):
        return get_coordinates_back_from_basis(basis_coords, self.basis)

#####################################################################################################
# GTKeyPositions
#####################################################################################################

# TODO: Do dynamic programming here
class GTKeyPositions:
    def __init__(self, json_filepath):
        with open(json_filepath) as json_file:
            self.key_positions = json.load(json_file)
    
    def get_unity_pos(self, keyname):
        key = self.key_positions[keyname]
        return key['Unity_x'], key["Unity_y"], key["Unity_z"]
    
    def get_all_unity_pos(self):
        result = { }
        for k in self.key_positions:
            result[k] = self.get_unity_pos(k)
        return result

    def get_all_unity_pos_in_basis(self, basis):
        result = { }
        for k in self.key_positions:
            result[k] = basis.get_coordinates_about_basis( self.get_unity_pos(k) )
        return result
    
    # vec_diff(each, key) applied
    # simply add key to recover to basis pos
    def get_all_unity_pos_in_basis_wrt_key(self, basis, key): 
        key_pos = basis.get_coordinates_about_basis(self.get_unity_pos(key))
        result = { }
        for k in self.key_positions:
            result[k] = \
                vec_diff( \
                    basis.get_coordinates_about_basis( self.get_unity_pos(k) ), \
                    key_pos )
        return result


#####################################################################################################
# VECTOR OPERATIONS
#####################################################################################################
def vec_len(a):
    b = [ x*x for x in a]
    b = sum(b)
    b = math.sqrt(b)

    return b

def get_normalized_vec(vec):
    norm = np.linalg.norm(vec)
    return [c/norm for c in vec]


def vec_normalized(a):
    return get_normalized_vec(a)

# a - b
def vec_diff(a, b):
    return [x - y for x,y in zip(a, b)]

def vec_sum(a, b):
    return [x + y for x,y in zip(a, b)]

def vec_dist(a, b):
    return vec_len(vec_diff(a, b))

# if you take the sqrt of the result, it gives the distance btw a and b
def vec_dist_square(a, b):
    diff = vec_diff(a, b)
    diffsqr = [x*x for x in diff]
    return sum(diffsqr)

def weighted_vec_avg(vec1, weight1, vec2, weight2):
    vec1 = [v*weight1 for v in vec1]
    vec2 = [v*weight2 for v in vec2]
    result = [(v1+v2)/(weight1+weight2) for v1,v2 in zip(vec1, vec2)]
    return result

def vec_avg(list_of_vecs):
    if len(list_of_vecs) == 0:
        print("vec_avg(): empty list of vecs")
        return None
    
    count = 1
    total = list_of_vecs[0]

    for i in range(1, len(list_of_vecs)):
        total = [ x+y for x,y in zip(total, list_of_vecs[i]) ]
        count += 1
    
    # avg
    avg = [x/count for x in total]
    return avg
    

def inner_product(a, b):
    return sum([x*y for x,y in zip(a,b)])

def dot_product(a, b):
    return inner_product(a, b)

def cross_product(a, b):
    ax, ay, az = a
    bx, by, bz = b

    x = ay*bz - az*by
    y = az*bx - ax*bz
    z = ax*by - ay*bx

    return [x, y, z]

def translate_3d(vec, translation):
    return [x+y for x,y in zip(vec,translation)]

def scale_3d(vec, scale):
    return  [x*y for x,y in zip(vec, scale)]


def is_vec_equal(vec1, vec2, eps):
    if not (vec_len(vec1) - vec_len(vec2) <= eps):
        return False
    
    if not (sum([abs(x-y) for x,y in zip(vec1, vec2)]) <= eps):
        return False

    return True

def find_2d_rotation_angle(vec1, vec2):
    vec1 = vec1[:2]
    vec2 = vec2[:2]

    #vec1[2] = 0
    #vec2[2] = 0

    vec1 = get_normalized_vec(vec1)
    vec2 = get_normalized_vec(vec2)

    angle = math.acos( inner_product(vec1, vec2) )

    rotated = rotate_2d(vec1, angle)

    if is_vec_equal(vec2, rotated, 0.000001):
        return angle
    else:
        angle = -angle
        rotated = rotate_2d(vec1, angle)

        if is_vec_equal(vec2, rotated, 0.000001):
            return angle
        else:
            assert False

def rotate_2d(vec, theta):
    x, y = vec[:2]

    c = math.cos(theta)
    s = math.sin(theta)

    x2 = c * x - s * y
    y2 = s * x + c * y

    return [x2, y2]

def cosine_similarity_2d(vec1, vec2):
    vec1 = vec1[:2]
    vec2 = vec2[:2]

    vlen1 = vec_len(vec1)
    vlen2 = vec_len(vec2)

    return inner_product(vec1, vec2) / (vlen1 * vlen2)


def vec_scale(vec, scalar):
    return [scalar*x for x in vec]




# angle: in degrees
# axis_theta = (axis, theta)
def rotate_3d(vec, axis_theta):
    axis, theta = axis_theta
    theta = math.radians(theta)

    x, y, z = vec

    # axis should 'x', 'y', or 'z'
    sint = math.sin(theta)
    cost = math.cos(theta)

    s = sint
    c = cost

    if axis == 'x':
        return [ \
            x, \
            y * c + z * s, \
            y * -s + z * c
        ]
    elif axis == 'y':
        return [
            x * c + z * -s, \
            y, \
            x * s + z * c
        ]
    elif axis == 'z':
        return [
            x * c + y * s, \
            x * -s + y * s, \
            z
        ]
    else:
        assert False


# DEPRECATED ONES

################################################################################
# Finding plane normal and position ############################################
################################################################################
def _plot_t_vals(t_list):
    # enumerate (to identify points later) and sort values (according to t vals)
    # result = list(enumerate(t_list))
    # result.sort(key = operator.itemgetter(1))
    # print("\nBEFORE RESULT\n")
    # print(result)
    # print("\nAFTER RESULT\n")

    #t_list.sort()
    #print("Sorted list: ", t_list)
    #val = 0.
    #plt.plot( t_list, np.zeros_like(t_list) + val, '|', markersize='30')
    fig, axs = plt.subplots()

    axs.boxplot(t_list)
    axs.set_title("basic plot")

    plt.show()
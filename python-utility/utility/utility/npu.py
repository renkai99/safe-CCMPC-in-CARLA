"""Numpy array operations"""
import numpy as np
import scipy
import scipy.optimize
import scipy.spatial
import scipy.stats

from . import UtilityException
from . import pairwise

def unit_normal_2d(p1, p2):
    """Compute unit normal vector counter-clockwise of the
    direction from p1 to p2 in 2D"""
    p = p2 - p1
    return np.array([-p[1], p[0]])

def kronecker_add_vectors(a, b):
    """Kronecker addition of two vectors,
    treating a as a row vector and b as a column vector"""
    return a[None, :] + b[:, None]

def kronecker_mul_vectors(a, b):
    """Kronecker multiplication of two vectors,
    treating a as a row vector and b as a column vector"""
    return np.kron(a[None, :], b[:, None])

def is_positive_semidefinite(X):
    """Check that a matrix is positive semidefinite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        regularized_X = X + np.eye(X.shape[0]) * 1e-14
        np.linalg.cholesky(regularized_X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def is_positive_definite(X):
    """Check that a matrix is positive definite
    
    Based on:
    https://stackoverflow.com/a/63911811
    """
    if X.shape[0] != X.shape[1]:
        return False
    if not np.all( X - X.T == 0 ):
        return False
    try:
        np.linalg.cholesky(X)
    except np.linalg.linalg.LinAlgError as err:
        if "Matrix is not positive definite"  == str(err):
            return False
        raise err
    return True

def indices_to_selection_mask(indices, n):
    """Create a ndarray mask to select ndarray at locations specified by indices.

    Parameters
    ==========
    indices : iterable of int
        Indices to set mask to True.
    n : int
        Size of the mask
    
    Returns
    =======
    ndarray
        The ndarray mask.
    """
    mask = np.full(n, False)
    for idx in indices:
        mask[idx] = True
    return mask

def reflect_radians_about_x_axis(r):
    r = (-r) % (2*np.pi)
    return r

def reflect_radians_about_y_axis(r):
    r = (np.pi - r) % (2*np.pi)
    return r

def warp_radians_neg_pi_to_pi(radians):
    """Warps radians to (-pi, pi]."""
    return (radians + np.pi) % (2 * np.pi) - np.pi

def warp_radians_0_to_2pi(radians):
    """Warps radians to [0, 2pi)"""
    return radians % (2*np.pi)

def warp_radians_about_center(radians, c):
    """Warps radians about interval (c - pi, c + pi] with center c.
    
    Parameters
    ==========
    radians : float or ndarray of float
        The radians to wrap as 1D ndarray.
    c : float of ndarray of float
        Center of interval to wrap radians.
        If c is a 1D ndarray with the same shape as `radians`,
        then warping is performed elementwise:
        f(radians, c) -> array([f(radians[i], c[i]) for i in range(radians.size)])
    
    Returns
    =======
    float of ndarray of float
        The radians wrapped.
    """
    return (radians - c + np.pi) % (2 * np.pi) + c - np.pi

def determinant_2d(A):
    """Compute the determinant of a 2D matrix ndarray."""
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return a*d - b*c

def inverse_2d(A):
    """Compute the inverse of a 2D matrix ndarray."""
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    return 1. / (a*d - b*c) * np.array([[d, -b], [-c, a]])

def rotation_2d(theta):
    """2D rotation matrix. If x is a column vector of 2D points then
    `rotation_2d(theta) @ x` gives the rotated points.
    
    Parameters
    ==========
    theta : float
        Rotates points clockwise about origin if theta is positive.
        Rotates points counter-clockwise about origin if theta is negative
    
    Returns
    =======
    ndarray
        2D rotation matrix of shape (2, 2).
    """
    return np.array([
            [ np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])

def consecutive_points_distances(X):
    """Obtain distances of consecutive points.
    Example: [1,2,4,8,4,0,1] -> [1,2,4,4,4,1]

    Parameters
    ==========
    X : ndarray
        Has shape (N,D) where N is the number of points and D is the dimension.
    
    Returns
    =======
    ndarray
        Distances of consecutive points of shape (N-1).
    """
    d = np.diff(X, axis=0)**2
    return np.sqrt(d.sum(axis=1) if d.ndim > 1 else d)

def cumulative_points_distances(X):
    """Obtain cumulative distances between consecutive points.
    Calls `consecutive_points_distances()`."""
    return np.cumsum(consecutive_points_distances(X))

def kernel_1d_gaussian(N, d, *params):
    """1D Kernel for Gaussian blur.

    Parameters
    ==========
    N : int
        Specifies the length of the filter.
    d : float
        The extent to sample points.

    Returns
    =======
    ndarray
        The filter discretizing the Gaussian blur PDF over [-d, d] of shape (2*N + 1).
    """
    x = np.linspace(-d, d, 2*N + 1)
    z = scipy.stats.norm.pdf(x)
    return z / np.sum(z)

def apply_kernel_1d(x, kernel, *params):
    """Apply a convolution on input x with a 1D kernel.

    Parameters
    ==========
    x : ndarray
        The input to apply convolution to.
    kernel : function
        The function to compute kernel.
    params : tuple
        The parameters to pass to the kernel function.

    Returns
    =======
    ndarray
        x with with same shape as x.
    ndarray
        The convolution used to apply convolution.
    """
    z = kernel(*params)
    y = np.convolve(x, z, mode='full')
    l = len(z)
    N = (l - 1 if l % 2 == 1 else l) // 2
    return y[N:-N], z

def distances_from_line_2d(points, x_start, y_start, x_end, y_end):
    """Get the distances from each point to a line spanned by line segment from
    (x_start, y_start) to (x_end, y_end). Works for horizontal and vertical lines.
    
    Based on:
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameters
    ==========
    points : np.array or list
        One 2D point, or multiple 2D points of shape (n, 2).
    x_start : float
        Line segment component
    y_start : float
        Line segment component
    x_end : float
        Line segment component
    y_end : float
        Line segment component

    Returns
    =======
    float or np.array
        Distance of point to line, or array of distances from points to line.
    """
    points = np.array(points)
    if points.ndim == 1:
        return np.abs((x_end - x_start)*(y_start - points[1]) - (x_start - points[0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    elif points.ndim == 2:
        return np.abs((x_end - x_start)*(y_start - points[:, 1]) - (x_start - points[:, 0])*(y_end - y_start)) \
                / np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
    else:
        raise UtilityException(f"Points of dimension {points.ndim} are not 1 or 2")

def extend_by_points(p1, direction=None, p2=None, distance=1., n=1):
    """Extend points from p1 in the direction vector, or towards of p2.
    The points are the same dimension, in any dimension > 1.
    
    Parameters
    ==========
    p1 : np.array
        Origin point to extend with more points.
    direction : np.array
        Unnormalized direction vector.
    p2 : np.array
        Point giving the direction from p1 we want to extend with more points.
        At least one of direction or p2 should be given.
    distance : float
        The distance between each extended point.
    n : int
        Number of points to extend from p1.
    distance
    n : int
        Number of points
    
    Returns
    =======
    np.array
        The extending points.
    """
    if direction is None:
        direction = (p2 - p1) / np.linalg.norm(p2 - p1)
    else:
        direction = direction / np.linalg.norm(direction)
    return p1 + kronecker_mul_vectors(direction, np.linspace(distance, n*distance, n))

def order_polytope_vertices(vertices):
    """Reorder the polytope vertices so that they are clockwise order around the polytope.
    Used by methods such as `vertices_to_halfspace_representation()` where vertex order matters.

    Returns
    =======
    np.ndarray
        The vertices of polytope of shape (N,2).

    Returns
    =======
    np.ndarray
        The vertices reordered, of shape (N,2).
    """
    mean = np.mean(vertices, axis=0)
    direction = vertices - mean
    dot = np.dot(direction, direction[0])
    norm = np.linalg.norm(direction, axis=-1) * np.linalg.norm(direction[0])
    dot_norm = np.clip(dot / norm, -0.99999, 0.99999)
    cos = np.arccos(dot_norm)
    sgn = np.sign(np.cross(direction, direction[0]))
    sgn[sgn == 0] = 1
    cos = sgn*cos
    indices = np.argsort(cos)
    return vertices[indices]

def vertices_of_bboxes(centers, theta, lw):
    """Get the vertices of N rectanglar bounding boxes given the centers of the boxes,
    the thetas the boxes they are pointing at, and the length and width of the boxes,
    assuming the length and widths of all boxes are the same.

    Parameters
    ==========
    centers : np.ndarray
        The centers of the bounding boxes of shape (N, 2).
    theta : number or np.ndarray
        The direction of the boxes in radians. Theta can be a number specifying the
        direction of all boxes, or a ndarray that specifies the direction for each box
        with shape (N,).
    lw : list or np.ndarray
        The length and width of the box. It can have shape (2,) and applied to all boxes,
        or have shape (N, 2) to specify the dimensions of each box separately.

    Returns
    =======
    np.ndarray
        The vertices of the boxes of shape (N,4,2).
    """
    if isinstance(lw, (list, tuple)):
        lw = np.array(lw)
    lws = np.repeat(lw[None], centers.shape[0], axis=0) if lw.ndim == 1 else lw
    thetas = np.full(centers.shape[0], theta) if np.ndim(theta) == 0 else theta
    C = np.cos(thetas)
    S = np.sin(thetas)
    rot11 = np.stack((-C,  S), axis=-1)
    rot12 = np.stack((-S, -C), axis=-1)
    rot21 = np.stack((-C, -S), axis=-1)
    rot22 = np.stack((-S,  C), axis=-1)
    rot31 = np.stack(( C, -S), axis=-1) 
    rot32 = np.stack(( S,  C), axis=-1)
    rot41 = np.stack(( C,  S), axis=-1)
    rot42 = np.stack(( S, -C), axis=-1)
    # Rot has shape (N, 8, 2)
    Rot = np.stack((rot11, rot12, rot21, rot22, rot31, rot32, rot41, rot42), axis=1)
    # disp has shape (N, 8)
    disp = 0.5 * np.einsum("...jk, ...k ->...j", Rot, lws)
    # centers has shape (N, 8)
    centers = np.tile(centers, (4,))
    return np.reshape(centers + disp, (-1,4,2))

def vertices_from_bbox(center, theta, lw):
    return vertices_of_bboxes(np.array([center]), np.array([theta]), lw)[0]

def interp_spline(points, normalize_interp=True, tol=1e-08):
    """Interpolate a spline to the points.
    Removes duplicate points and points close together in Euclidean space.

    Parameters
    ==========
    points : ndarray
        Points to interpolate of shape (N, D).
    normalize_interp : bool
        Normalize parameterization to [0, 1].
    tol : float
        Remove consecutive points within tol distance close.
    
    Returns
    =======
    scipy.interpolate.CubicSpline
        A cubic spline.
    ndarray
        The cumulative distances used to parameterize the cubic spline.
    """
    distances = np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1))
    mask = ~np.isclose(distances, 0., atol=tol)
    mask = np.insert(mask, -1, True)
    points = points[mask]
    distances = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)
    if normalize_interp:
        distances = distances / distances[-1]
    return scipy.interpolate.CubicSpline(distances, points, axis=0), points, distances

def interp_and_sample(points, n, interpolation='quadratic'):
    """Interpolate a spline over points and sample n equally
    spaced out points from the spline.
    
    Parameters
    ==========
    points : ndarray
        Points to interpolate of shape (N, 2).
    n : int
        Number of evenly spaced points to sample.
    
    Returns
    =======
    ndarray
        Sampled points of shape (n, 2).
    """
    distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)/distance[-1]
    interpolator = scipy.interpolate.interp1d(distance, points, kind=interpolation, axis=0)
    return interpolator(np.linspace(0, 1, n))

def place_rectangles_on_intep_curve(points, n, lws, thetas=None, interpolation='quadratic'):
    """Interpolate a curve on points and then place boxes on the curve.
    Calls `scipy.interpolate.interp1d()` to do the interpolate.

    Parameters
    ==========
    points : ndarray
        Points to interpolate of shape (N, 2).
    n : int
        Number of boxes to place on the interpolated curve.
    lws : ndarray
        The length and width of the box. It can have shape (2,) and applied to all boxes,
        or have shape (N, 2) to specify the dimensions of each box separately.
    thetas : number or ndarray (optional)
        The direction of the boxes in radians. Theta can be a number specifying the
        direction of all boxes, or a ndarray that specifies the direction for each box
        with shape (N,). By default the boxes point in the direction of the curve.
    interpolation : str or int (optional)
        Specifies the kind of interpolation for `scipy.interpolate.interp1d()` call.
    
    Returns
    =======
    ndarray
        Vertices of rectangles of shape (n, 4, 2).
    """
    interp_points = interp_and_sample(points, 2*n - 1, interpolation=interpolation)
    if thetas is None:
        X = interp_points[:2*n-2].reshape(-1, 2, 2).astype(complex)
        X = X[:, 1, :] - X[:, 0, :]
        X = X[:, 0] + 1j*X[:, 1]
        thetas = np.angle(X)
        X = interp_points[-2] - interp_points[-1]
        thetas = np.concatenate((thetas, [np.angle(X[0] + 1j*X[1])]))
    centers = interp_points[::2]
    return vertices_of_bboxes(centers, thetas, lws)

def pairs2d_to_halfspace(p1, p2):
    """Get half-space representation dividing the left side and the right side
    of the line formed by two points in R^2. The points x where Ax <= b are on
    the right side of the arrow from p1 to p2.
    
     x_2
      ^
      |  \
      |   \  Ax > b
      |    p1
    --|-----\------> x_1
      |      p2
     Ax <= b  \
      |        \

    Parameters
    ==========
    p1 : ndarray
        First point
    p2 : ndarray
        Second point
    
    Returns
    =======
    np.array
        A where Ax <= b
    int
        b where Ax <= b
    
    """
    p11, p12 = p1
    p21, p22 = p2
    A = np.array([p12-p22, p21-p11])
    b = (p12 - p22)*p11 + (p21 - p11)*p12
    return A, b

def vertices_to_halfspace_representation(vertices):
    """Vertices of convex polytope to half-space representation (A, b).
    where points x, A x <= b are inside the polytope. 
    
    Parameters
    ==========
    vertices : np.array
        Vertices of convex polytope of shape (N, 2) where N is the number of vertices.
        The vertices are sorted in clockwise order along the first axis and N > 2.
        
    Returns
    =======
    np.array
        A where x, Ax <= b are the points of the polytope 
    np.array
        b where x, Ax <= b are the points of the polytope
    """
    vertices = np.concatenate((vertices, vertices[0][None],), axis=0)
    A = []; b = []
    for p1, p2 in pairwise(vertices):
        _A, _b = pairs2d_to_halfspace(p1, p2)
        A.append(_A); b.append(_b)
    A = np.stack(A); b = np.array(b)
    return A, b

####################
# Plotting functions
####################

def plot_h_polyhedron(ax, A, b, fc='none', ec='none', alpha=0.3, **kwargs):
    """Plot a convex polytope in H-representation A x < b.
    Note: [A; b], A x + b < 0 is the format for HalfspaceIntersection
    
    Parameters
    ==========
    ax : matplotlib.axes.Axes
    A : ndarray
    b : ndarray
    fc : str
    ec : str
    ls : float
        Line size.
    alpha : float
        Transparency.
    """
    Ab = np.concatenate((A, -b[...,None],), axis=-1)
    res = scipy.optimize.linprog([0, 0],
            A_ub=Ab[:,:2], b_ub=-Ab[:,2],
            bounds=(None, None))
    hs = scipy.spatial.HalfspaceIntersection(Ab, res.x)
    ch = scipy.spatial.ConvexHull(hs.intersections)
    x, y = zip(*hs.intersections[ch.vertices])
    ax.fill(x, y, fc=fc, ec=ec, alpha=alpha, **kwargs)

#####################################################################
# Sequential reimplementation of some Numpy functions for object type
# Used when types cannot be vectorized
#####################################################################

def obj_matmul(A, B):
    """Non-vectorized multiplication of arrays of object dtype.
    
    Sequential reimplementation of some Numpy functions for object type.
    Used when types cannot be vectorized."""
    if len(A.shape) == 1 and len(B.shape) == 1:
        C = 0
        for i in range(A.shape[0]):
            C += A[i]*B[i]
    elif len(B.shape) == 1:
        C = np.zeros((A.shape[0]), dtype=object)
        for i in range(A.shape[0]):
            for k in range(A.shape[1]):
                C[i] += A[i,k]*B[k]
    else:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=object)
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                for k in range(A.shape[1]):
                    C[i,j] += A[i,k]*B[k,j]
    return C

def obj_vectorize(f, A):
    """Non-vectorized emulation of tensor vectorization.

    Sequential reimplementation of some Numpy functions for object type.
    Used when types cannot be vectorized."""
    if A.ndim == 0:
        return f(A)
    elif A.ndim == 1:
        return np.array([f(a) for a in A])
    else:
        return np.stack([obj_vectorize(f, a) for a in A])

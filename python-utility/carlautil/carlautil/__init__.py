import os
import math
import copy
import json
import itertools
import functools

import numpy as np
import networkx as nx

import carla

import utility as util


class CARLAUtilException(Exception):
    pass


def make_client(host="127.0.0.1", port=2000):
    """Create a client. Useful for debugging in the Python interpreter."""
    return carla.Client(host, port)


def debug_point(
    client,
    l,
    t=1.0,
    c=carla.Color(r=255, g=0, b=0, a=100),
    label="o"
):
    """Draw a point in the simulator.

    Parameters
    ----------
    client : carla.Client or carla.World
        Client.
    l : carla.Transform or carla.Location
        Position in map to place the point.
    t : float, optional
        Life time of point.
    c : carla.Color, optional
        Color of the point.
    label : str
        Alternative string to label the point.
    """
    if isinstance(l, carla.Transform):
        l = l.location
    if isinstance(client, carla.Client):
        world = client.get_world()
    else:
        world = client
    world.debug.draw_string(l, label, color=c, life_time=t)


def debug_square(
    client,
    l,
    r,
    rotation=carla.Rotation(),
    t=1.0,
    c=carla.Color(r=255, g=0, b=0, a=100),
):
    """Draw a square centered on a point.

    Parameters
    ----------
    client : carla.Client or carla.World
        Client.
    l : carla.Transform or carla.Location
        Position in map to place the point.
    r : float
        Radius of the square from the center
    t : float, optional
        Life time of point.
    c : carla.Color, optional
        Color of the point.
    """
    if isinstance(l, carla.Transform):
        l = l.location
    if isinstance(client, carla.Client):
        world = client.get_world()
    else:
        world = client
    box = carla.BoundingBox(l, carla.Vector3D(r, r, r))
    world.debug.draw_box(box, rotation, thickness=0.5, color=c, life_time=t)


def carlacopy(x):
    """Copies objects defined in CARLA.
    Note: the function exists because copy.deepcopy() fails."""
    if isinstance(x, carla.Vector3D):
        return type(x)(x.x, x.y, x.z)
    elif isinstance(x, carla.Rotation):
        return carla.Rotation(x.pitch, x.yaw, x.roll)
    elif isinstance(x, carla.Transform):
        return carla.Transform(
            location=carlacopy(x.location), rotation=carlacopy(x.rotation)
        )
    else:
        raise NotImplementedError(f"Don't know how to copy carla.{type(x).__name__}")


#############################
# Object ndarray manipulation
#############################

def location_to_ndarray(l, flip_x=False, flip_y=False):
    """Converts carla.Location to ndarray [x, y, z]"""
    x_mult, y_mult = np.where([flip_x, flip_y], -1, 1)
    return np.array([x_mult * l.x, y_mult * l.y, l.z])


def rotation_to_ndarray(r, flip_x=False, flip_y=False):
    """Converts carla.Rotation to ndarray [pitch, yaw, roll] in radians."""
    if flip_x and flip_y:
        raise NotImplementedError()
    if flip_x:
        raise NotImplementedError()
    if flip_y:
        pitch, yaw, roll = np.deg2rad([r.pitch, r.yaw, r.roll])
        yaw = util.reflect_radians_about_x_axis(yaw)
        pitch = -pitch
        roll = -roll
        return np.array([pitch, yaw, roll])
    else:
        return np.deg2rad([r.pitch, r.yaw, r.roll])


def ndarray_to_location(v, flip_x=False, flip_y=False):
    """ndarray of form [x, y, z] to carla.Location."""
    x_mult, y_mult = np.where([flip_x, flip_y], -1, 1)
    return carla.Location(x=x_mult * v[0], y=y_mult * v[1], z=v[2])


def actor_to_location_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's location to ndarray [x, y, z]"""
    return location_to_ndarray(a.get_location(), flip_x=flip_x, flip_y=flip_y)


def to_location_ndarray(a, flip_x=False, flip_y=False):
    """Converts location of object of relevant carla class to ndarray [x, y, z]"""
    if isinstance(a, carla.Actor):
        return location_to_ndarray(a.get_location(), flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Junction):
        return location_to_ndarray(
            a.bounding_box.location, flip_x=flip_x, flip_y=flip_y
        )
    elif isinstance(a, carla.Waypoint):
        return location_to_ndarray(a.transform.location, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Transform):
        return location_to_ndarray(a.location, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Location):
        return location_to_ndarray(a, flip_x=flip_x, flip_y=flip_y)
    else:
        raise CARLAUtilException("Not relevant carla class.")


def to_locations_ndarray(l, flip_x=False, flip_y=False):
    """Converts list of object of relevant carla class to ndarray of size (len(l), 3)"""
    return util.map_to_ndarray(
        lambda a: to_location_ndarray(a, flip_x=flip_x, flip_y=flip_y), l
    )


def actor_to_velocity_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's component-wise velocity to ndarray
    [vel_x, vel_y, vel_z]"""
    x_mult, y_mult = np.where([flip_x, flip_y], -1, 1)
    v = a.get_velocity()
    return np.array([x_mult * v.x, y_mult * v.y, v.z])


def actor_to_speed(a):
    """Get speed of vector in m/s."""
    v = a.get_velocity()
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def actor_to_acceleration(a):
    """Get magnitude of acceleration in m/s^2"""
    _a = a.get_acceleration()
    return math.sqrt(_a.x**2 + _a.y**2 + _a.z**2)


def actor_to_bbox_ndarray(a):
    """Converts carla.Actor's bounding box dimensions to ndarray
    [bbox_x, bbox_y, box_z]. bbox_x is the length on the longitudinal axis,
    bbox_y is the length on the lateral axis."""
    bb = a.bounding_box.extent
    return np.array([2 * bb.x, 2 * bb.y, 2 * bb.z])


def actor_to_rotation_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Actor's component-wise velocity to ndarray
    [pitch, yaw, roll]"""
    t = a.get_transform()
    r = t.rotation
    return rotation_to_ndarray(r, flip_x=flip_x, flip_y=flip_y)


def actor_to_forward_vector_ndarray(a, flip_x=False, flip_y=False):
    """Gets carla.Actor's forward unit vector in world coordinates."""
    f = a.get_transform().get_forward_vector()
    x_mult, y_mult = np.where([flip_x, flip_y], -1, 1)
    return np.array([x_mult*f.x, y_mult*f.y, f.z])


def actor_to_forward_velocity(a):
    """Get magnitude of forward component of velocity vector."""
    f = actor_to_forward_vector_ndarray(a)
    v = actor_to_velocity_ndarray(a)
    return np.dot(f, v)


def to_rotation_ndarray(a, flip_x=False, flip_y=False):
    """Converts velocity of object of relevant carla class to ndarray
    [pitch, yaw, roll] in radians."""
    if isinstance(a, carla.Actor):
        return rotation_to_ndarray(
            a.get_transform().rotation, flip_x=flip_x, flip_y=flip_y
        )
    elif isinstance(a, carla.Waypoint):
        return rotation_to_ndarray(a.transform.rotation, flip_x=flip_x, flip_y=flip_y)
    elif isinstance(a, carla.Transform):
        return rotation_to_ndarray(a.rotation, flip_x=flip_x, flip_y=flip_y)
    else:
        raise CARLAUtilException("Not relevant carla class.")


def to_rotations_ndarray(l, flip_x=False, flip_y=False):
    """Converts list of object of relevant carla class to ndarray of size (len(l), 3)"""
    return util.map_to_ndarray(
        lambda a: to_rotation_ndarray(a, flip_x=flip_x, flip_y=flip_y), l
    )


def actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(a, flip_x=False, flip_y=False):
    """Converts carla.Vehicle
    to ndarray [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
    length, width, height, pitch, yaw, roll] where pitch, yaw, roll are in
    radians.
    TODO: naming is wrong. Should be actor_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray()"""
    x_mult, y_mult = np.where([flip_x, flip_y], -1, 1)
    bb = a.bounding_box.extent
    t = a.get_transform()
    v = a.get_velocity()
    a = a.get_acceleration()
    l = t.location
    r = t.rotation
    l = [x_mult * l.x, y_mult * l.y, l.z]
    v = [x_mult * v.x, y_mult * v.y, v.z]
    a = [x_mult * a.x, y_mult * a.y, a.z]
    bb = [2 * bb.x, 2 * bb.y, 2 * bb.z]
    r = rotation_to_ndarray(r, flip_x=flip_x, flip_y=flip_y)
    return np.concatenate((l, v, a, bb, r))

actor_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray = actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray

def actors_to_location_ndarray(alist, flip_x=False, flip_y=False):
    """Converts iterable of carla.Actor to a ndarray of size (len(alist), 3)"""
    return util.map_to_ndarray(
        lambda a: actor_to_location_ndarray(a, flip_x=flip_x, flip_y=flip_y), alist
    )


def actors_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(alist, flip_x=False, flip_y=False):
    """Converts iterable of carla.Actor transformation
    to an ndarray of size (len(alist), 15) where each row is
    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z,
    length, width, height, pitch, yaw, roll]
    where pitch, yaw, roll are in radians.
    TODO: naming is wrong. Should be actors_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray()"""
    return util.map_to_ndarray(
        lambda a: actor_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray(a, flip_x=flip_x, flip_y=flip_y),
        alist,
    )


actors_to_Lxyz_Vxyz_Axyz_Blwh_Rpyr_ndarray = actors_to_Lxyz_Vxyz_Axyz_Rpyr_ndarray

def transform_to_location_ndarray(t, flip_x=False, flip_y=False):
    """Converts carla.Transform to location ndarray [x, y, z]"""
    return location_to_ndarray(t.location, flip_x=flip_x, flip_y=flip_y)


def transforms_to_location_ndarray(ts, flip_x=False, flip_y=False):
    """Converts an iterable of carla.Transform to a ndarray of size (len(iterable), 3)"""
    return util.map_to_ndarray(
        lambda t: transform_to_location_ndarray(t, flip_x=flip_x, flip_y=flip_y), ts
    )


##########################
# Vehicle specific methods
##########################


def fix_throttle(world, vehicle, is_synchronous=True):
    """Used in some CARLA versions to prevent controlled vehicles to stall
    when throttle is first applied.

    See Github issues:
    - https://github.com/carla-simulator/carla/issues/1640
    - https://github.com/carla-simulator/carla/issues/3256
    """
    if is_synchronous:
        tick = lambda : world.tick()
    else:
        tick = lambda : world.wait_for_tick()
    vehicle.apply_control(
        carla.VehicleControl(throttle=0, brake=1, manual_gear_shift=True, gear=1)
    )
    tick()
    vehicle.apply_control(
        carla.VehicleControl(manual_gear_shift=False)
    )
    tick()


def create_gear_control(manual_gear_shift=True, gear=1, **kwargs):
    """Same as carla.VehicleControl, but by default sets manual gear
    to true and gear to 1."""
    return carla.VehicleControl(
            manual_gear_shift=manual_gear_shift,
            gear=gear, **kwargs)


def get_steering_angle(vehicle):
    """Get steering angle of front wheels.
    NOTE: only works for CARLA 0.9.12 and above.
    """
    fl_angle = vehicle.get_wheel_steer_angle(
            carla.VehicleWheelLocation.FL_Wheel)
    fr_angle = vehicle.get_wheel_steer_angle(
            carla.VehicleWheelLocation.FR_Wheel)
    return math.radians((fl_angle + fr_angle) / 2.)


#########################################
# Transform and waypoint specific methods
#########################################


def move_along_road(carla_map, transform, distance):
    """Given a transform on a road. Select another transform some distance
    forward or backwards on the road.

    Parameters
    ==========
    carla_map : carla.Map
        To query the road on the map.
    transform : carla.Transform
        The transform on the road to select another one relative to it.
    distance : float
        Distance in meters forward if distance > 0, or backward if distance > 0.

    Returns
    =======
    carla.Transform
        Transform shifted some distance along the road.
    """
    if distance == 0:
        return transform
    wp = carla_map.get_waypoint(
        transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if distance < 0:
        return wp.previous(abs(distance))[0].transform
    else:
        return wp.next(distance)[0].transform


def transform_to_origin(transform, origin):
    """Create an adjusted transformation relative to origin.
    Creates a new transformation (doesn't mutate the parameters).

    Parameters
    ----------
    transform : carla.Transform
        The transform we want to adjust
    origin : carla.Transform or np.array
        The origin we want to adjust the transform to

    Returns
    -------
    carla.Transform
        New transform with the origin as reference.
    """
    location = transform.location
    rotation = transform.rotation
    return carla.Transform(
        carla.Location(
            x=location.x - origin.location.x,
            y=location.y - origin.location.y,
            z=location.z - origin.location.z,
        ),
        carla.Rotation(
            pitch=rotation.pitch,
            yaw=rotation.yaw - origin.rotation.yaw,
            roll=rotation.roll,
        ),
    )


def get_junctions_from_topology_graph(topology):
    """Gets unique junctions from topology

    Parameters
    ----------
    topology : nx.Graph or list

    Returns
    -------
    list of carla.Junction
    """
    if isinstance(topology, list):
        G = nx.Graph()
        G.add_edges_from(topology)
        topology = G
    junctions = map(
        lambda v: v.get_junction(), filter(lambda v: v.is_junction, topology.nodes)
    )
    return list({j.id: j for j in junctions}.values())


def to_xy_point(wp, flip_x=False, flip_y=False):
    """Convert waypoint to a 2D point.

    Parameters
    ==========
    wp : carla.Waypoint

    Returns
    =======
    list of float
    """
    x, y, _ = to_location_ndarray(wp, flip_x=flip_x, flip_y=flip_y)
    return [x, y]


def collect_points_along_path(
    start_wp, choices, max_distance, precision=1.0, flip_x=False, flip_y=False
):
    """Collects points along path on a CARLA map, beginning with provided waypoint.
    
    Parameters
    ==========
    start_wp : carla.Waypoint
        Waypoint designating the start of the path.
    choices : list of int
        Indices of turns at each junction along the path from start_wp onwards.
        If there are more junctions than indices contained in choices, then 
        choose default turn. 
    max_distance : float
        Maximum distance of path from start_wp we want to sample from.
    precision : float
        Distance between consecutive waypoints on the path to sample. 
    
    Returns
    =======
    np.array of float
        Sampled points along path starting from waypoint start_wp
        of shape (N, 2).
    list of float
        Accumulated distance from the first point to the last on
        the path starting from distance 0. of the initial point.
        List has length N.
    float
        Cumulative distance from first to last point, equal
        to the accumulated distance of the last point.
    """
    _to_point = functools.partial(to_xy_point, flip_x=flip_x, flip_y=flip_y)
    wp = start_wp
    wps = [wp]
    points = [_to_point(wp)]
    distances = [0.]
    cum_distance = 0.
    iidx = 0
    while cum_distance < max_distance:
        next_wps = wp.next(precision)
        if len(next_wps) > 1:
            try:
                wp = next_wps[choices[iidx]]
            except IndexError:
                wp = next_wps[0]
            iidx += 1
        else:
            wp = next_wps[0]
        wps.append(wp)
        points.append(_to_point(wp))
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        cum_distance += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(cum_distance)
    points = np.array(points)
    return wps, points, distances


def cylindrical_to_camera_watcher_transform(r, theta, z, location=carla.Location()):
    """Get the transformation to define camera position and orientation looking directly
    at location given some cylindrical coordinates relative to given location.

    Parameters
    ==========
    r : float
        distance in meters to define cylindrical coordinates relative to location.
    theta : float
        The angle in radians to define cylindrical coordinates relative to location.
    z : float
        The relative height from location in meters to define cylindrical coordinates
        relative to location.
    location : Carla.Location
        The location the camera should look at. By default, the location is origin.

    Return
    ======
    carla.Transform
        Transformation to define camera position and orientation looking directly at car.
    """
    location = carla.Location(r * math.cos(theta), r * math.sin(theta), z) + location
    rotation = carla.Rotation(
        yaw=math.degrees(theta + math.pi), pitch=-math.degrees(math.atan2(z, r))
    )
    return carla.Transform(location, rotation)


def spherical_to_camera_watcher_transform(r, theta, phi, location=carla.Location()):
    """Get the transformation to define camera position and orientation looking directly
    at location given some spherical coordinates relative to given location.

    Parameters
    ==========
    r : float
        radial distance in meters to define spherical coordinates relative to location.
    theta : float
        The azimuthal angle in radians (about xy-plane) to define spherical
        relative to location.
    phi : float
        The polar angle in radians (from +z-axis) from location in meters to define spherical coordinates
        relative to location.
    location : Carla.Location
        The location the camera should look at. By default, the location is origin.

    Return
    ======
    carla.Transform
        Transformation to define camera position and orientation looking directly at car.
    """
    location = (
        carla.Location(
            r * math.sin(phi) * math.cos(theta),
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi),
        )
        + location
    )
    rotation = carla.Rotation(
        yaw=math.degrees(theta + math.pi),
        pitch=-math.degrees(math.atan(1/math.tan(phi))),
    )
    return carla.Transform(location, rotation)


def strafe_transform(transform, right=0, above=0):
    """Strafe transform (right,left) and (above,below)
    while maintaining orientation."""
    if above != 0 and right != 0:
        transform = carlacopy(transform)
    if above != 0:
        ptr = carla.Transform(rotation=carlacopy(transform.rotation))
        ptr.rotation.pitch += 90
        ptr = ptr.get_forward_vector()
        transform.location += above * ptr
    if right != 0:
        ptr = carla.Transform(rotation=carlacopy(transform.rotation))
        ptr.rotation.yaw += 90
        ptr = ptr.get_forward_vector()
        transform.location += right * ptr
    return transform


####################
# Deprecated methods
####################


def locations_to_ndarray(ls, flip_x=False, flip_y=False):
    """Converts list of carla.Location to ndarray of size (len(ls), 3).
    DEPRECATED: use to_locations_ndarray()"""
    return util.map_to_ndarray(
        lambda l: location_to_ndarray(l, flip_x=flip_x, flip_y=flip_y), ls
    )


def transform_to_yaw(t):
    """Converts carla.Transform to rotation yaw mod 360
    DEPRECATED"""
    return t.rotation.yaw % 360.0


def transform_points(transform, points):
    """Given a 4x4 transformation matrix, transform an array of 3D points.
    Expected point foramt: [[X0,Y0,Z0],..[Xn,Yn,Zn]].
    DEPRECATED"""
    # Needed format: [[X0,..Xn],[Z0,..Zn],[Z0,..Zn]]. So let's transpose
    # the point matrix.
    points = points.transpose()
    # Add 0s row: [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[0,..0]]
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # Point transformation
    # points = transform * points
    points = np.dot(transform, points)
    # Return all but last row
    return points[0:3].transpose()


##############
# Scratch work
##############


"""Creating internal library for similarity transformations
since CARLA transformation matrices are unreliable.

Based on
https://cs184.eecs.berkeley.edu/uploads/lectures/05_transforms-2/05_transforms-2_slides.pdf
"""


def create_translation_mtx(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def create_x_rotation_mtx(a):
    """Create elementary rotation matrix about global +x-axis"""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(a), -np.sin(a), 0],
            [0, np.sin(a), np.cos(a), 0],
            [0, 0, 0, 1],
        ]
    )


def create_y_rotation_mtx(b):
    """b is pitch"""
    return np.array(
        [
            [np.cos(b), 0, np.sin(b), 0],
            [0, 1, 0, 0],
            [-np.sin(b), 0, np.cos(b), 0],
            [0, 0, 0, 1],
        ]
    )


def create_z_rotation_mtx(c):
    """c is yaw"""
    return np.array(
        [
            [np.cos(c), -np.sin(c), 0, 0],
            [np.sin(c), np.cos(c), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transform_to_translation_mtx(transform):
    l = transform.location
    return create_translation_mtx(l.x, l.y, l.z)


class SimilarityTransform(object):
    @staticmethod
    def to_radians(d):
        return math.radians(d)

    @classmethod
    def from_transform(cls, transform):
        location = transform.location
        rotation = transform.rotation
        return cls(
            location.x,
            location.y,
            location.z,
            cls.to_radians(rotation.yaw),
            cls.to_radians(rotation.pitch),
            cls.to_radians(rotation.roll),
        )

    def __init__(self, x, y, z, yaw, pitch, roll):
        self.x = x
        self.y = y
        self.z = z
        self.translation = create_translation_mtx(x, y, z)
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

import json
import itertools
import functools
import numpy as np
import networkx as nx
import os
import carla

import utility as util
import carlautil

def show_sloped_wps(client, dest=4.0,
        pitch=6.0, t=1.0,
        c=carla.Color(r=255, g=0, b=0, a=100)):
    world = client.get_world()
    carla_map = world.get_map()
    wps = carla_map.generate_waypoints(dest)
    ppitch = (360. - pitch)
    def f(wp):
        wp_pitch = wp.transform.rotation.pitch % 360.
        return wp_pitch > pitch and wp_pitch < ppitch
    sloped_wps = util.filter_to_list(f, wps)
    for wp in sloped_wps:
        loc = wp.transform.location
        carlautil.debug_point(client, loc, t=t, c=c)

def show_square(client, x, y, z, r,
        rotation=carla.Rotation(), t=1.0,
        c=carla.Color(r=255, g=0, b=0, a=100)):
    """Create a square at a position with radius.
    
    Parameters
    ----------
    client : carla.Client or carla.World
    """
    l = carla.Location(x, y, z)
    carlautil.debug_point(client, l, t=t, c=c)
    carlautil.debug_square(client, l, r, t=t, c=c)

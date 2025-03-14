
import os
import copy
import random

import numpy as np
import pytest

import utility as util

random.seed(256)

def sort_list_of_list(ll):
    for l in ll:
        l.sort()

def test_id_maker():
    id_maker = util.IDMaker(
            'map_name/episode/agent/frame',
            prefixes={
                'episode':  'ep',
                'agent':    'agent',
                'frame':    'frame'},
            format_spec={
                'episode':  '03d',
                'agent':    '03d',
                'frame':    '08d'})

    id_preprocess = [
            util.AttrDict(map_name='Town01', episode=1, agent=1, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=2, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=3, frame=1000),
            util.AttrDict(map_name='Town01', episode=1, agent=1, frame=2000),
            util.AttrDict(map_name='Town01', episode=1, agent=2, frame=2000),
            util.AttrDict(map_name='Town01', episode=2, agent=1, frame=500),
            util.AttrDict(map_name='Town01', episode=2, agent=2, frame=500),
            util.AttrDict(map_name='Town01', episode=2, agent=3, frame=530),
            util.AttrDict(map_name='Town01', episode=2, agent=4, frame=530),
            util.AttrDict(map_name='Town02', episode=3, agent=1, frame=100),
            util.AttrDict(map_name='Town02', episode=3, agent=2, frame=100),
            util.AttrDict(map_name='Town02', episode=4, agent=1, frame=100),
            util.AttrDict(map_name='Town02', episode=4, agent=2, frame=100),]

    ids = ['Town01/ep001/agent001/frame00001000',
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent001/frame00002000',
            'Town01/ep001/agent002/frame00002000',
            'Town01/ep002/agent001/frame00000500',
            'Town01/ep002/agent002/frame00000500',
            'Town01/ep002/agent003/frame00000530',
            'Town01/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',]

    expected = {'map_name': 0, 'episode': 1, 'agent': 2, 'frame':3}
    actual = id_maker.sample_pattern
    assert actual == expected

    expected = '{map_name:}/ep{episode:03d}/agent{agent:03d}/frame{frame:08d}'
    actual = id_maker.fstring
    assert actual == expected

    def f(d):
        return id_maker.make_id(
                map_name=d.map_name, episode=d.episode,
                agent=d.agent, frame=d.frame)
    actual = util.map_to_list(f, id_preprocess)
    expected = ids
    assert actual == expected

def test_filter_ids():
    id_maker = util.IDMaker('map_name/episode/agent/frame')
    ids = ['Town01/ep001/agent001/frame00001000',
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent001/frame00002000',
            'Town01/ep001/agent002/frame00002000',
            'Town01/ep002/agent001/frame00000500',
            'Town01/ep002/agent002/frame00000500',
            'Town01/ep002/agent003/frame00000530',
            'Town02/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',
            'Town03/ep005/agent001/frame00001000',
            'Town03/ep005/agent001/frame00002000',]
    random.shuffle(ids)

    filter = {'map_name': 'Town02'}
    actual = id_maker.filter_ids(ids, filter)
    actual.sort()
    expected = [
            'Town02/ep002/agent004/frame00000530',
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent001/frame00000100',
            'Town02/ep004/agent002/frame00000100',]
    expected.sort()
    assert actual == expected

    filter = {'map_name': 'Town02', 'episode': 'ep003'}
    actual = id_maker.filter_ids(ids, filter)
    actual.sort()
    expected = [
            'Town02/ep003/agent001/frame00000100',
            'Town02/ep003/agent002/frame00000100',]
    expected.sort()
    assert actual == expected
    
    filter = {'agent': 'agent001', 'episode': 'ep002'}
    actual = id_maker.filter_ids(ids, filter, inclusive=False)
    actual.sort()
    expected = [
            'Town01/ep001/agent002/frame00001000',
            'Town01/ep001/agent003/frame00001000',
            'Town01/ep001/agent002/frame00002000',
            'Town02/ep003/agent002/frame00000100',
            'Town02/ep004/agent002/frame00000100',]
    expected.sort()
    assert actual == expected

def test_group_ids():
    id_maker = util.IDMaker('annotation/subtype/slide/patch_size/magnification/coordinate')
    patch_ids = [
        'Stroma/MMRd/VOA-1000A/512/20/0_0',
        'Stroma/MMRd/VOA-1000A/512/20/2_2',
        'Stroma/MMRd/VOA-1000A/512/10/0_0',
        'Stroma/MMRd/VOA-1000A/256/10/0_0',
        'Tumor/POLE/VOA-1000B/256/10/0_0']
    
    groups, labels = id_maker.group_ids(patch_ids, ['patch_size'])
    
    expected = {'patch_size': ['256', '512']}
    util.sort_nested_dict_of_list(expected)
    actual = labels
    util.sort_nested_dict_of_list(actual)
    assert actual == expected

    expected = {
        '512': [
            'Stroma/MMRd/VOA-1000A/512/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/2_2',
        ],
        '256': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    actual = groups
    util.sort_nested_dict_of_list(actual)
    assert actual == expected
    
    groups, labels = id_maker.group_ids(patch_ids, ['patch_size', 'magnification'])

    expected = {'patch_size': ['256', '512'], 'magnification': ['10', '20']}
    util.sort_nested_dict_of_list(expected)
    actual = labels
    util.sort_nested_dict_of_list(actual)
    assert actual == expected


    expected = {
        '512': {
            '20': [
                'Stroma/MMRd/VOA-1000A/512/20/0_0',
                'Stroma/MMRd/VOA-1000A/512/20/2_2',
            ],
            '10': [
                'Stroma/MMRd/VOA-1000A/512/10/0_0',
            ]
        },
        '256': {
            '20': [ ],
            '10': [
                'Stroma/MMRd/VOA-1000A/256/10/0_0',
                'Tumor/POLE/VOA-1000B/256/10/0_0'
            ]
        }
    }
    util.sort_nested_dict_of_list(expected)
    actual = groups
    util.sort_nested_dict_of_list(actual)
    assert actual == expected


def test_group_ids_by_index():
    id_maker = util.IDMaker('annotation/subtype/slide/patch_size/magnification/coordinate')
    patch_ids = [
        'Stroma/MMRd/VOA-1000A/512/20/0_0',
        'Stroma/MMRd/VOA-1000A/512/20/2_2',
        'Stroma/MMRd/VOA-1000A/512/10/0_0',
        'Stroma/MMRd/VOA-1000A/256/20/0_0',
        'Stroma/MMRd/VOA-1000A/256/10/0_0',
        'Tumor/POLE/VOA-1000B/256/10/0_0']
    
    actual = id_maker.group_ids_by_index(patch_ids,
            include=['annotation', 'magnification'])
    util.sort_nested_dict_of_list(actual)
    expected = {
        'Stroma/20': [
            'Stroma/MMRd/VOA-1000A/256/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/2_2'
        ],
        'Stroma/10': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/10/0_0'
        ],
        'Tumor/10': [
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    assert actual == expected
    
    actual = id_maker.group_ids_by_index(patch_ids,
            exclude=['slide', 'magnification'])
    util.sort_nested_dict_of_list(actual)
    expected = {
        'Stroma/MMRd/512/0_0': [
            'Stroma/MMRd/VOA-1000A/512/10/0_0',
            'Stroma/MMRd/VOA-1000A/512/20/0_0'
        ],
        'Stroma/MMRd/512/2_2': [
            'Stroma/MMRd/VOA-1000A/512/20/2_2'
        ],
        'Stroma/MMRd/256/0_0': [
            'Stroma/MMRd/VOA-1000A/256/10/0_0',
            'Stroma/MMRd/VOA-1000A/256/20/0_0'
        ],
        'Tumor/POLE/256/0_0': [
            'Tumor/POLE/VOA-1000B/256/10/0_0'
        ]
    }
    util.sort_nested_dict_of_list(expected)
    assert actual == expected

def test_extract_value():
    id_maker = util.IDMaker('map_name/episode/agent/frame')
    ids = ['Town01/ep001/agent001/frame00001000',
           'Town01/ep001/agent002/frame00001000',
           'Town01/ep001/agent003/frame00001000',
           'Town01/ep001/agent001/frame00002000',
           'Town01/ep001/agent002/frame00002000',
           'Town01/ep002/agent001/frame00000500',
           'Town01/ep002/agent002/frame00000500',
           'Town01/ep002/agent003/frame00000530',
           'Town02/ep002/agent004/frame00000530',
           'Town02/ep003/agent001/frame00000100',
           'Town02/ep003/agent002/frame00000100',
           'Town02/ep004/agent001/frame00000100',
           'Town02/ep004/agent002/frame00000100',
           'Town03/ep005/agent001/frame00001000',
           'Town03/ep005/agent001/frame00002000']

    expected = 'agent001'
    actual = id_maker.extract_value(ids[0], 'agent')
    assert actual == expected

    expected = ['Town01', 'Town01', 'Town01', 'Town01', 'Town01',
            'Town01', 'Town01', 'Town01', 'Town02', 'Town02',
            'Town02', 'Town02', 'Town02', 'Town03', 'Town03']
    actual = id_maker.extract_value(ids, 'map_name')
    assert actual == expected

    expected = ['ep001', 'ep001', 'ep001', 'ep001', 'ep001',
            'ep002', 'ep002', 'ep002', 'ep002', 'ep003', 'ep003',
            'ep004', 'ep004', 'ep005', 'ep005']
    actual = id_maker.extract_value(ids, 'episode')
    assert actual == expected
    
    expected = ['agent001', 'agent002', 'agent003', 'agent001',
            'agent002', 'agent001', 'agent002', 'agent003',
            'agent004', 'agent001', 'agent002', 'agent001',
            'agent002', 'agent001', 'agent001']
    actual = id_maker.extract_value(ids, 'agent')
    assert actual == expected
    
    expected = ['frame00001000', 'frame00001000', 'frame00001000',
            'frame00002000', 'frame00002000', 'frame00000500',
            'frame00000500', 'frame00000530', 'frame00000530',
            'frame00000100', 'frame00000100', 'frame00000100',
            'frame00000100', 'frame00001000', 'frame00002000']
    actual = id_maker.extract_value(ids, 'frame')
    assert actual == expected











import math
import numpy as np
import pytest

from ugv_self_supervised_traversability.terrain_analysis import (
    TerrainConfig, analyze_terrain, MotionEstimator, PASS, BLOCKED, UNKNOWN)


def cloud(height=lambda x,y: 0*x, hole=None):
    x,y=np.meshgrid(np.arange(-5.9,6,0.08),np.arange(-5.9,6,0.08))
    points=np.column_stack([x.ravel(),y.ravel(),height(x,y).ravel()])
    return points if hole is None else points[~hole(points[:,0],points[:,1])]


def test_flat_corridor_and_no_returns():
    r=analyze_terrain(cloud(),0,0,0)
    assert r.corridor()['status']=='GEOMETRY_PASS'
    assert r.corridor(-1)['status']=='GEOMETRY_PASS'
    empty=analyze_terrain(np.empty((0,3)),0,0,0)
    assert empty.corridor()['status']=='UNKNOWN'
    assert not empty.observed.any()


def test_uphill_downhill_limits_are_directional():
    # 14 degrees is allowed uphill (15), blocked downhill (12).
    r=analyze_terrain(cloud(lambda x,y:x*np.tan(np.radians(14))),0,0,0)
    assert r.corridor()['status']=='GEOMETRY_PASS'
    assert r.corridor(-1)['status']=='GEOMETRY_BLOCKED'
    assert r.corridor(-1)['reasons']['downhill_limit']>0
    assert r.slope_deg[15,15]==pytest.approx(14,abs=0.1)


def test_heading_rotation_reverses_slope():
    pts=cloud(lambda x,y:x*np.tan(np.radians(14)))
    r=analyze_terrain(pts,0,0,math.pi)
    assert r.corridor()['status']=='GEOMETRY_BLOCKED'
    assert r.corridor(-1)['status']=='GEOMETRY_PASS'


def test_cross_slope():
    r=analyze_terrain(cloud(lambda x,y:y*np.tan(np.radians(16))),0,0,0)
    assert r.corridor()['reasons']['cross_slope_limit']>0
    assert r.corridor(-1)['status']=='GEOMETRY_BLOCKED'


@pytest.mark.parametrize('height,reason',[(0.4,'step_up_limit'),(-0.4,'drop_limit')])
def test_observed_step_and_drop(height,reason):
    r=analyze_terrain(cloud(lambda x,y:np.where(x>1.4,height,0.0)),0,0,0)
    assert r.corridor()['status']=='GEOMETRY_BLOCKED'
    assert r.corridor()['reasons'][reason]>0


def test_missing_ground_is_unknown_not_a_cliff():
    r=analyze_terrain(cloud(hole=lambda x,y:(x>1)&(x<2)),0,0,0)
    assert r.corridor()['status']=='UNKNOWN'
    assert 'drop_limit' not in r.corridor()['reasons']


def test_rough_surface_blocks():
    r=analyze_terrain(cloud(lambda x,y:0.2*np.sin(50*x)*np.cos(50*y)),0,0,0)
    assert r.corridor()['reasons']['roughness_or_obstacle']>0


def test_translation_and_world_height_are_preserved():
    pts=cloud();pts+=np.array([20,-7,3.5])
    r=analyze_terrain(pts,20,-7,0)
    assert r.corridor()['status']=='GEOMETRY_PASS'
    assert np.nanmedian(r.centers[:,:,2])==pytest.approx(3.5)


def test_nonfinite_and_degenerate_input():
    r=analyze_terrain(np.array([[np.nan,0,0],[1,np.inf,2]]),0,0,0)
    assert r.corridor()['status']=='UNKNOWN'
    points=np.column_stack([np.linspace(-5,5,300),np.zeros(300),np.zeros(300)])
    assert analyze_terrain(points,0,0,0).corridor()['status']=='UNKNOWN'


@pytest.mark.parametrize('kwargs',[{'resolution':0},{'max_uphill_deg':90},{'radius':1},{'max_drop':float('nan')}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): TerrainConfig(**kwargs)


def test_motion_ascending_descending_stationary_and_clock_reset():
    m=MotionEstimator()
    assert m.update(0,0,0,0)['state']=='UNKNOWN'
    assert m.update(1,1,0,.2)['state']=='ASCENDING'
    assert m.update(2,2,0,0)['state']=='DESCENDING'
    assert m.update(3,2,0,0)['state']=='STATIONARY'
    assert m.update(.5,0,0,0)['state']=='UNKNOWN'


def test_imu_motion_follows_displacement_not_pitch_sign_alone():
    from ugv_self_supervised_traversability.terrain_analysis import attitude_motion
    pitch=-np.radians(14)
    assert attitude_motion(0,pitch,0,1,0)['state']=='ASCENDING'
    assert attitude_motion(0,pitch,0,-1,0)['state']=='DESCENDING'
    assert attitude_motion(0,pitch,0,0,0)['state']=='STATIONARY'
    assert attitude_motion(0,pitch,math.pi/2,0,1)['state']=='ASCENDING'
    assert attitude_motion(0,np.pi/2,0,1,0)['state']=='UNKNOWN'

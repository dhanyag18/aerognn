import numpy as np

from aerognn.geometry.superformula import generate_cross_section, apply_aspect_ratio, normalize_area
from aerognn.geometry.extrusion import extrude_building

def test_circle_when_n_equals_2():
    cs = generate_cross_section(n_1=2, n_2=2, n_3=2, m=4)
    radii = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
    assert np.allclose(radii, radii[0], atol=1e-6)
    print("worked")

def test_square():
    cs = generate_cross_section(n_1=500, n_2=500, n_3=500, m=4)
    radii = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
    r_min = np.min(radii)
    r_max = np.max(radii)
    assert np.allclose(r_min, 1, atol=1e-2)
    assert np.allclose(r_max, np.sqrt(2), atol=1e-2)

def test_aspect_ratio():
    cs = generate_cross_section(n_1=1000, n_2=1000, n_3=1000, m=4)
    cs_scaled = apply_aspect_ratio(cs, 2)
    width = np.ptp(cs_scaled[:, 0])
    height = np.ptp(cs_scaled[:, 1])
    assert np.allclose(width/height, 2, atol=1e-2)

def test_area_normalization():
    cs = generate_cross_section(n_1=10, n_2=10, n_3=10, m=3)
    cs_scaled = apply_aspect_ratio(cs, 2)
    cs_normalized = normalize_area(cs_scaled)
    
    x=cs_normalized[:,0]
    y=cs_normalized[:,1]
    S1=np.sum(x*np.roll(y,-1))
    S2=np.sum(y*np.roll(x,-1))
    current_area=0.5*np.abs(S1-S2)

    assert np.allclose(current_area, 5000, atol=1e-6)

def test_height():
    cs = generate_cross_section(n_1=10, n_2=10, n_3=10, m=3)
    vert, face = extrude_building(cs)
    max_z = np.max(vert[:, 2])

    assert np.allclose(max_z, 200, atol=1e-6)





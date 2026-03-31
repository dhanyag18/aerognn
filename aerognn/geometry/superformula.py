import numpy as np
def generate_cross_section(   
        n_1: float,
        n_2: float,
        n_3: float,
        m: int,
        num_points: int,
        a: float = 1.0,
        b: float = 1.0,
):
    phi = np.linspace(0, 2 * np.pi, num_points)
    r = (np.abs(np.cos(m*phi/4)/a)**n_2 + np.abs(np.sin(m*phi/4)/b)**n_3)**(-1/n_1)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack((x,y))

def apply_aspect_ratio(
        cross_section: np.ndarray,
        aspect_ratio: float
):
    scaled=cross_section.copy()
    scaled[:, 0]*=aspect_ratio
    return scaled

def normalize_area(
        cross_section: np.ndarray,
        target_area: float=5000.0
):
    x=cross_section[:,0]
    y=cross_section[:,1]
    S1=np.sum(x*np.roll(y,-1))
    S2=np.sum(y*np.roll(x,-1))
    current_area=0.5*np.abs(S1-S2)

    scale=np.sqrt(target_area/current_area)
    return cross_section*scale


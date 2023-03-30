import numpy as np
from scipy.interpolate import CubicHermiteSpline
import cv2

def generate_gt_mask(points_file_path: str, width: int, height: int) -> np.ndarray:
    """
    Generate a binary numpy array with dimensions (height, width) containing a smooth closed curve that passes
    through all the control points specified in the file at points_file_path, using Hermite spline interpolation.
    The curve passes through the endpoints. All the pixel values within the computed curve equal 1, and 0 otherwise.

    Args:
        points_file_path (str): The path of the text file containing the x-y coordinates of the control points.
        width (int): The width of the output binary array.
        height (int): The height of the output binary array.

    Returns:
        np.ndarray: A binary numpy array of size (height, width) where all the pixel values within the computed
        curve equal 1, and 0 otherwise.

    """

    # Load control points from file
    points = np.loadtxt(points_file_path)

    # Ensure that the curve passes through the endpoints
    points = np.vstack((points[-1], points, points[0]))

    # Compute tangents at each point
    tangents = np.zeros_like(points)
    tangents[1:-1] = (points[2:] - points[:-2]) / 2

    # Interpolate using Hermite spline
    x_interp = np.linspace(0, len(points)-1, 10000)
    spline_x = CubicHermiteSpline(np.arange(len(points)), points[:,0], tangents[:,0])
    spline_y = CubicHermiteSpline(np.arange(len(points)), points[:,1], tangents[:,1])
    interp_points = np.stack((spline_x(x_interp), spline_y(x_interp)), axis=1)

    # Create mask from interpolated curve
    mask = np.zeros((height, width))
    interp_points = interp_points.astype(int)
    mask[interp_points[:,1], interp_points[:,0]] = 1

    # Fill inside of mask using flood fill
    h, w = mask.shape[:2]
    mask_full = np.zeros((h+2, w+2), dtype=np.uint8)
    cv2.floodFill(mask.astype(np.uint8), mask_full, (0,0), 1)
    mask = mask_full[1:-1, 1:-1]

    return mask.astype(bool)
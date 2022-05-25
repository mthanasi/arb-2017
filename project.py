import numpy as np
import cv2
import math

CV_QR_NORTH = 0
CV_QR_EAST = 1
CV_QR_SOUTH = 2
CV_QR_WEST = 3

# Start of Main Loop
# ------------------------------------------------------------------------------------------------------------------------
def main(args):
    image = cv2.imread(args[1])
    key = 0
    align = None

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200, 3)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    mark = 0

    # Get Moments for all Contours and the mass centers
    mu = [cv2.moments(c, False) for c in contours]
    mc = [0 for _ in mu]
    for (i, m) in enumerate(mu):
        try:
            mc[i] = [m["m10"] / m["m00"], m["m01"] / m["m00"]]
        except ZeroDivisionError:
            mc[i] = [float("inf"), float("inf")]

    # Start processing the contour data

    # Find Three repeatedly enclosed contours A,B,C
    # NOTE: 1. Contour enclosing other contours is assumed to be the three Alignment markings of the QR code.
    # 2. Alternately, the Ratio of areas of the "concentric" squares can also be used for identifying base Alignment markers.
    # The below demonstrates the first method

    mark = 0
    for i in range(len(contours)):
        k = i
        c = 0

        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c += 1

        if hierarchy[k][2] != -1:
            c += 1

        if c >= 5:

            if mark == 0:
                A = i
            elif mark == 1:  # i.e., A is already found, assign current contour to B
                B = i
            elif mark == 2:  # i.e., A and B are already found, assign current contour to C
                C = i
            mark += 1

    if mark >= 3:  # // Ensure we have (atleast 3; namely A,B,C) 'Alignment Markers' discovered
        # We have found the 3 markers for the QR code; Now we need to determine which of them are 'top', 'right' and 'bottom' markers

        # Determining the 'top' marker
        # Vertex of the triangle NOT involved in the longest side is the 'outlier'

        AB = cv_distance(mc[A], mc[B])
        BC = cv_distance(mc[B], mc[C])
        CA = cv_distance(mc[C], mc[A])

        if AB > BC and AB > CA:
            outlier = C
            median1 = A
            median2 = B
        elif CA > AB and CA > BC:
            outlier = B
            median1 = A
            median2 = C
        elif BC > AB and BC > CA:
            outlier = A
            median1 = B
            median2 = C

        top = outlier  # the obvious choice

        dist = cv_lineEquation(
            mc[median1], mc[median2], mc[outlier]
        )  # Get the Perpendicular distance of the outlier from the longest side
        (align, slope) = cv_lineSlope(mc[median1], mc[median2], align)  # Also calculate the slope of the longest side

        # Now that we have the orientation of the line formed median1 & median2 and we also have the position of the outlier w.r.t. the line
        # Determine the 'right' and 'bottom' markers

        if align == 0:
            bottom = median1
            right = median2
        elif slope < 0 and dist < 0:  # Orientation - North
            bottom = median1
            right = median2
            orientation = CV_QR_NORTH
        elif slope > 0 and dist < 0:  # Orientation - East
            right = median1
            bottom = median2
            orientation = CV_QR_EAST
        elif slope < 0 and dist > 0:  # Orientation - South
            right = median1
            bottom = median2
            orientation = CV_QR_SOUTH
        elif slope > 0 and dist > 0:  # Orientation - West
            bottom = median1
            right = median2
            orientation = CV_QR_WEST

        # To ensure any unintended values do not sneak up when QR code is not present
        # float area_top, area_right, area_bottom;

        if (
            top < len(contours)
            and right < len(contours)
            and bottom < len(contours)
            and cv2.contourArea(contours[top]) > 10
            and cv2.contourArea(contours[right]) > 10
            and cv2.contourArea(contours[bottom]) > 10
        ):

            # vector<Point2f> L,M,O, tempL,tempM,tempO;
            # Point2f N;

            # vector<Point2f> src,dst;		// src - Source Points basically the 4 end co-ordinates of the overlay image
            # 								// dst - Destination Points to transform overlay image

            # Mat warp_matrix;

            tempL = cv_getVertices(contours, top, slope)
            tempM = cv_getVertices(contours, right, slope)
            tempO = cv_getVertices(contours, bottom, slope)

            L = cv_updateCornerOr(orientation, tempL)  # Re-arrange marker corners w.r.t orientation of the QR code
            M = cv_updateCornerOr(orientation, tempM)  # Re-arrange marker corners w.r.t orientation of the QR code
            O = cv_updateCornerOr(orientation, tempO)  # Re-arrange marker corners w.r.t orientation of the QR code

            (iflag, N) = getIntersectionPoint(M[1], M[2], O[3], O[2])

            src = np.array([L[0], M[1], N, O[3]]).astype(np.float32)
            dest = np.array([[0, 0], [100, 0], [100, 100], [0, 100]]).astype(np.float32)
            warp_matrix = cv2.getPerspectiveTransform(src, dest)
            print(warp_matrix)

            cv2.drawContours(image, contours, top, [255, 200, 0], 2, 8, hierarchy, 0)
            cv2.drawContours(image, contours, right, [0, 0, 255], 2, 8, hierarchy, 0)
            cv2.drawContours(image, contours, bottom, [255, 0, 100], 2, 8, hierarchy, 0)

    cv2.imshow("Image", image)

    return 0


# End of Main Loop
# --------------------------------------------------------------------------------------


# Routines used in Main loops

# Function: Routine to get Distance between two points
# Description: Given 2 points, the function returns the distance
def cv_distance(P, Q):
    return math.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)


# Function: Perpendicular Distance of a Point J from line formed by Points L and M; Equation of the line ax+by+c=0
# Description: Given 3 points, the function derives the line quation of the first two points,
# 	  calculates and returns the perpendicular distance of the the 3rd point from this line.
def cv_lineEquation(L, M, J):
    a = -((M[1] - L[1]) / (M[0] - L[0]))
    b = 1.0
    c = (((M[1] - L[1]) / (M[0] - L[0])) * L[0]) - L[0]

    # Now that we have a, b, c from the equation ax + by + c, time to substitute (x,y) by values from the Point J
    return (a * J[0] + (b * J[1]) + c) / math.sqrt((a * a) + (b * b))


# Function: Slope of a line by two Points L and M on it; Slope of line, S = (x1 -x2) / (y1- y2)
# Description: Function returns the slope of the line formed by given 2 points, the alignement flag
# 	  indicates the line is vertical and the slope is infinity.
def cv_lineSlope(L, M, alignment):
    d = np.array(M) - np.array(L)

    if d[1] != 0:
        return 1, d[1] / d[0]
    else:
        return 0, 0.0


# Function: Routine to calculate 4 Corners of the Marker in Image Space using Region partitioning
# Theory: OpenCV Contours stores all points that describe it and these points lie the perimeter of the polygon.
# 	The below function chooses the farthest points of the polygon since they form the vertices of that polygon,
# 	exactly the points we are looking for. To choose the farthest point, the polygon is divided/partitioned into
# 	4 regions equal regions using bounding box. Distance algorithm is applied between the centre of bounding box
# 	every contour point in that region, the farthest point is deemed as the vertex of that region. Calculating
# 	for all 4 regions we obtain the 4 corners of the polygon ( - quadrilateral).
def cv_getVertices(contours, c_id, slope):
    box = cv2.boundingRect(contours[c_id])

    # Point2f M0,M1,M2,M3;
    M0 = []
    M1 = []
    M2 = []
    M3 = []
    # Point2f A, B, C, D, W, X, Y, Z;

    A = box_tl(box)
    B = np.array([box_br(box)[0], box_tl(box)[1]])

    C = box_br(box)
    D = np.array([box_tl(box)[0], box_br(box)[1]])

    W = np.array([(A[0] + B[0]) / 2, A[1]])

    X = np.array([B[0], (B[1] + C[1]) / 2])

    Y = np.array([(C[0] + D[0]) / 2, C[1]])

    Z = np.array([D[0], (D[1] + A[1]) / 2])

    dmax = np.zeros(4, dtype=float)

    if slope > 5 or slope < -5:
        for c in contours[c_id]:
            pd1 = cv_lineEquation(C, A, c)  # Position of point w.r.t the diagonal AC
            pd2 = cv_lineEquation(B, D, c)  # Position of point w.r.t the diagonal BD

            if (pd1 >= 0.0) and (pd2 > 0.0):
                (dmax[1], M1) = cv_updateCorner(c, W, dmax[1], M1)
            elif (pd1 > 0.0) and (pd2 <= 0.0):
                (dmax[2], M2) = cv_updateCorner(c, X, dmax[2], M2)
            elif (pd1 <= 0.0) and (pd2 < 0.0):
                (dmax[3], M3) = cv_updateCorner(c, Y, dmax[3], M3)
            elif (pd1 < 0.0) and (pd2 >= 0.0):
                (dmax[0], M0) = cv_updateCorner(c, Z, dmax[0], M0)
            else:
                continue
    else:
        halfx = (A[0] + B[0]) / 2
        halfy = (A[1] + D[1]) / 2

        for c in contours[c_id]:
            c = c[0]
            if (c[0] < halfx) and (c[1] <= halfy):
                (dmax[2], M0) = cv_updateCorner(c, C, dmax[2], M0)
            elif (c[0] >= halfx) and (c[1] < halfy):
                (dmax[3], M1) = cv_updateCorner(c, D, dmax[3], M1)
            elif (c[0] > halfx) and (c[1] >= halfy):
                (dmax[0], M2) = cv_updateCorner(c, A, dmax[0], M2)
            elif (c[0] <= halfx) and (c[1] > halfy):
                (dmax[1], M3) = cv_updateCorner(c, B, dmax[1], M3)

    return np.array([M0, M1, M2, M3])


def box_tl(box):
    (x, y, w, h) = box
    return np.array([x, y])


def box_br(box):
    (x, y, w, h) = box
    return np.array([x + w, y + h])


# Function: Compare a point if it more far than previously recorded farthest distance
# Description: Farthest Point detection using reference point and baseline distance
def cv_updateCorner(P, ref, baseline, corner):
    temp_dist = cv_distance(P, ref)

    if temp_dist > baseline:
        return temp_dist, P
    else:
        return baseline, corner


# Function: Sequence the Corners wrt to the orientation of the QR Code
def cv_updateCornerOr(orientation, IN):
    if orientation == CV_QR_NORTH:
        M0 = IN[0]
        M1 = IN[1]
        M2 = IN[2]
        M3 = IN[3]
    elif orientation == CV_QR_EAST:
        M0 = IN[1]
        M1 = IN[2]
        M2 = IN[3]
        M3 = IN[0]
    elif orientation == CV_QR_SOUTH:
        M0 = IN[2]
        M1 = IN[3]
        M2 = IN[0]
        M3 = IN[1]
    elif orientation == CV_QR_WEST:
        M0 = IN[3]
        M1 = IN[0]
        M2 = IN[1]
        M3 = IN[2]

    return np.array([M0, M1, M2, M3])


def getIntersectionPoint(a1, a2, b1, b2):
    p = a1
    q = b1
    r = a2 - a1
    s = b2 - b1

    if cross(r, s) == 0:
        return False, None

    t = cross(q - p, s) / cross(r, s)

    return True, p + t * r


def cross(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))

# Code referenced and adapted from https://github.com/iperov/DeepFaceLab/blob/c96370339525b7145d2843a0fb96de5ccaab1a8d/facelib/LandmarksProcessor.py#L701

import numpy as np
import cv2
import numpy.linalg as npla
from libs.umeyama import umeyama
from libs.landmarkMasking import get_image_hull_mask

landmarks_2D_new = np.array([
[ 0.000213256,  0.106454  ], #17
[ 0.0752622,    0.038915  ], #18
[ 0.18113,      0.0187482 ], #19
[ 0.29077,      0.0344891 ], #20
[ 0.393397,     0.0773906 ], #21
[ 0.586856,     0.0773906 ], #22
[ 0.689483,     0.0344891 ], #23
[ 0.799124,     0.0187482 ], #24
[ 0.904991,     0.038915  ], #25
[ 0.98004,      0.106454  ], #26
[ 0.490127,     0.203352  ], #27
[ 0.490127,     0.307009  ], #28
[ 0.490127,     0.409805  ], #29
[ 0.490127,     0.515625  ], #30
[ 0.36688,      0.587326  ], #31
[ 0.426036,     0.609345  ], #32
[ 0.490127,     0.628106  ], #33
[ 0.554217,     0.609345  ], #34
[ 0.613373,     0.587326  ], #35
[ 0.121737,     0.216423  ], #36
[ 0.187122,     0.178758  ], #37
[ 0.265825,     0.179852  ], #38
[ 0.334606,     0.231733  ], #39
[ 0.260918,     0.245099  ], #40
[ 0.182743,     0.244077  ], #41
[ 0.645647,     0.231733  ], #42
[ 0.714428,     0.179852  ], #43
[ 0.793132,     0.178758  ], #44
[ 0.858516,     0.216423  ], #45
[ 0.79751,      0.244077  ], #46
[ 0.719335,     0.245099  ], #47
[ 0.254149,     0.780233  ], #48
[ 0.726104,     0.780233  ], #54
], dtype=np.float32)

# 68 point landmark definitions
landmarks_68_pt = { "mouth": (48,68),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "nose": (27, 36), # missed one point
                    "jaw": (0, 17) }

def get_transform_mat (image_landmarks, output_size, face_type, scale=1.0):
    
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks)

    # estimate landmarks transform from global space to local aligned space with bounds [0..1]
    mat = umeyama( np.concatenate ( [ image_landmarks[17:49] , image_landmarks[54:55] ] ) , landmarks_2D_new, True)[0:2]
    
    # get corner points in global space
    g_p = transform_points (  np.float32([(0,0),(1,0),(1,1),(0,1),(0.5,0.5) ]) , mat, True)
    g_c = g_p[4]

    # calc diagonal vectors between corners in global space
    tb_diag_vec = (g_p[2]-g_p[0]).astype(np.float32)
    tb_diag_vec /= npla.norm(tb_diag_vec)
    bt_diag_vec = (g_p[1]-g_p[3]).astype(np.float32)
    bt_diag_vec /= npla.norm(bt_diag_vec)

    # calc modifier of diagonal vectors for scale and padding value
    padding = 0.30
    mod = (1.0 / scale)* ( npla.norm(g_p[0]-g_p[2])*(padding*np.sqrt(2.0) + 0.5) )
    
    # adjust center for WHOLE_FACE, 7% below in order to cover more forehead
    vec = (g_p[0]-g_p[3]).astype(np.float32)
    vec_len = npla.norm(vec)
    vec /= vec_len

    g_c += vec*vec_len*0.07
    
    l_t = np.array( [ g_c - tb_diag_vec*mod,
                      g_c + bt_diag_vec*mod,
                      g_c + tb_diag_vec*mod ] )

    # calc affine transform from 3 global space points to 3 local space points size of 'output_size'
    pts2 = np.float32(( (0,0),(output_size,0),(output_size,output_size) ))
    mat = cv2.getAffineTransform(l_t,pts2)
    return mat

def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform (mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points

def polygon_area(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def draw_landmarks (image, image_landmarks, color, draw_circles, thickness, transparent_mask):
    if len(image_landmarks) != 68:
        raise Exception('get_image_eye_mask works only with 68 landmarks')

    int_lmrks = np.array(image_landmarks, dtype=np.int)

    jaw = int_lmrks[slice(*landmarks_68_pt["jaw"])]
    right_eyebrow = int_lmrks[slice(*landmarks_68_pt["right_eyebrow"])]
    left_eyebrow = int_lmrks[slice(*landmarks_68_pt["left_eyebrow"])]
    mouth = int_lmrks[slice(*landmarks_68_pt["mouth"])]
    right_eye = int_lmrks[slice(*landmarks_68_pt["right_eye"])]
    left_eye = int_lmrks[slice(*landmarks_68_pt["left_eye"])]
    nose = int_lmrks[slice(*landmarks_68_pt["nose"])]

    if draw_circles:
        # open shapes
        cv2.polylines(image, tuple(np.array([v]) for v in ( right_eyebrow, jaw, left_eyebrow, np.concatenate((nose, [nose[-6]])) )),
                      False, color, thickness=thickness, lineType=cv2.LINE_AA)
        # closed shapes
        cv2.polylines(image, tuple(np.array([v]) for v in (right_eye, left_eye, mouth)),
                      True, color, thickness=thickness, lineType=cv2.LINE_AA)
        
        # the rest of the cicles
        for x, y in np.concatenate((right_eyebrow, left_eyebrow, mouth, right_eye, left_eye, nose), axis=0):
            cv2.circle(image, (x, y), 1, color, 1, lineType=cv2.LINE_AA)
        # jaw big circles
        for x, y in jaw:
            cv2.circle(image, (x, y), 2, color, lineType=cv2.LINE_AA)
        
    if transparent_mask:
        mask = get_image_hull_mask(image.shape, image_landmarks)
        mask = cv2.merge((mask,mask,mask))

        image[...] = ( image * (1-mask) + image * mask / 255 )[...]
        
        return image
        
def draw_polygon (image, points, color, thickness = 1):
    points_len = len(points)
    for i in range (0, points_len):
        p0 = tuple( points[i] )
        p1 = tuple( points[ (i+1) % points_len] )
        cv2.line (image, p0, p1, color, thickness=thickness)


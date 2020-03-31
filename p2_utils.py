import math
import numpy as np
from scipy import io
import scipy
import cv2
import matplotlib.pyplot as plt

#################################### MAP #####################################
def initializeMap(res, xmin, ymin, xmax, ymax, memory = None, trust = None, optimism = None, occupied_thresh = None, free_thresh = None, confidence_limit = None):
    if memory == None:
        memory = 1 # set to value between 0 and 1 if memory is imperfect
    if trust == None:
        trust = 0.8
    if optimism == None:
        optimism = 0.5
    if occupied_thresh == None:
        occupied_thresh = 0.85
    if free_thresh == None:
        free_thresh = 0.2 # 0.5 # 0.25
    if confidence_limit == None:
        confidence_limit = 100 * memory
    
    MAP = {}
    MAP['res']   = res #meters; used to detrmine the number of square cells
    MAP['xmin']  = xmin  #meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # number of horizontal cells
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) # number of vertical cells
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float64) # contains log odds. DATA TYPE: char or int8
    
    # Related to log-odds 
    MAP['memory'] = memory
    MAP['occupied'] = np.log(trust / (1 - trust))
    MAP['free'] = optimism * np.log((1 - trust) / trust) # Try to be optimistic about exploration, so weight free space
    MAP['confidence_limit'] = confidence_limit

    # Related to occupancy grid
    MAP['occupied_thresh'] = np.log(occupied_thresh / (1 - occupied_thresh))
    MAP['free_thresh'] = np.log(free_thresh / (1 - free_thresh))
    (h, w) = MAP['map'].shape
    MAP['plot'] = np.zeros((h, w, 3), np.uint8) 
    
    return MAP 

def updateMap(MAP, x_w, y_w, x_curr, y_curr):
    
    # convert lidar hits to map coordinates
    x_m, y_m = world2map(MAP, x_w, y_w)
    
    # convert robot's position to map coordinates
    x_curr_m, y_curr_m = world2map(MAP, x_curr, y_curr)
    
    indGood = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),
                             (y_m < MAP['sizey']))
    
    MAP['map'] = MAP['map'] * MAP['memory']
    MAP['map'][x_m[0][indGood[0]], y_m[0][indGood[0]]] += MAP['occupied'] - MAP['free'] # we're going to add the MAP['free'] back in a second
    
    # initialize a mask where we will label the free cells
    free_grid = np.zeros(MAP['map'].shape).astype(np.int8) 
    x_m = np.append(x_m, x_curr_m) # Must consider robot's current cell
    y_m = np.append(y_m, y_curr_m)
    contours = np.vstack((x_m, y_m)) # SWITCH

    # find the cells that are free, and update the map
    cv2.drawContours(free_grid, [contours.T], -1, MAP['free'], -1) 
    MAP['map'] += free_grid    
    
    # prevent overconfidence
    MAP['map'][MAP['map'] > MAP['confidence_limit']] = MAP['confidence_limit']
    MAP['map'][MAP['map'] < -MAP['confidence_limit']] = -MAP['confidence_limit']
    
    # update plot
    occupied_grid = MAP['map'] > MAP['occupied_thresh']
    free_grid = MAP['map'] < MAP['free_thresh']
    
    MAP['plot'][occupied_grid] = [0, 0, 0]
    MAP['plot'][free_grid] = [255, 255, 255] 
    
    MAP['plot'][np.logical_and(np.logical_not(free_grid), np.logical_not(occupied_grid))] = [127, 127, 127]
    
    x_m, y_m = world2map(MAP, x_w, y_w)
    MAP['plot'][y_m, x_m] = [0, 255, 0]    # plot latest lidar scan hits
    
def lidar2map(MAP, x_l, y_l):
    #x_w, y_w = lidar2world()
    x_m, y_m = world2map(MAP, x_l, y_l)
    
    # build a single map in the lidar's frame of reference
    indGood = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),
                             (y_m < MAP['sizey']))

    map = np.zeros(MAP['map'].shape)
    map[x_m[0][indGood[0]], y_m[0][indGood[0]]] = 1
    np.int8(map)
        
    return map

def world2map(MAP, x_w, y_w): 
    # convert from meters to cells
    x_m = np.ceil((x_w - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    y_m = np.ceil((y_w - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    
    indGood = np.logical_and(np.logical_and(np.logical_and((x_m > 1), (y_m > 1)), (x_m < MAP['sizex'])),
                             (y_m < MAP['sizey']))
    
    x_m = x_m[indGood]
    y_m = y_m[indGood]
    
    return x_m.astype(np.int), y_m.astype(np.int)

################################## PARTICLES #################################
# A particle is just a tuple of weights (scalar) and pose (3x1)
def initializeParticles(num = None, n_thresh = None, noise_cov = None):
    if num == None:
        num = 100
    if n_thresh == None:
        n_thresh = 0.1 * num # set threshold to 20% of original number of particles to resample
    if noise_cov == None:
        noise_cov = np.zeros((3,3)) # for debugging purposes
        noise_cov = 0.5 * np.eye(3) # set noise covariances for multivariate Gaussian. This is 10% of the delta_pose movement (check predictParticles)
        noise_cov = np.array([[.1, 0, 0], [0, .1, 0], [0, 0, 0.005]])
    
    PARTICLES = {}
    PARTICLES['num'] = num
        
    PARTICLES['n_thresh'] = n_thresh    # below this value, resample
    PARTICLES['noise_cov'] = noise_cov  # covariances for Gaussian noise in each dimension
    
    PARTICLES['weights'] = np.ones(PARTICLES['num']) / PARTICLES['num'] 
    PARTICLES['poses'] = np.zeros((PARTICLES['num'], 3))

    return PARTICLES

def predictParticles(PARTICLES, d_x, d_y, d_yaw, x_prev, y_prev, yaw_prev):
    
    noise_cov =  np.matmul(PARTICLES['noise_cov'], np.abs(np.array([[d_x, 0, 0], [0, d_y, 0], [0, 0, d_yaw]]))) 
        
    # create hypothesis (particles) poses
    noise = np.random.multivariate_normal([0, 0, 0], noise_cov, PARTICLES['num'])
    PARTICLES['poses'] = noise + np.array([[x_prev, y_prev, yaw_prev]])
    
    # update poses according to deltas
    PARTICLES['poses'] += np.array([[d_x, d_y, d_yaw]])
    return

def updateParticles(PARTICLES, MAP, x_l, y_l, psi, theta):
    
    n_eff = 1 / np.sum(np.square(PARTICLES['weights']))
    
    if (n_eff < PARTICLES['n_thresh']):
        print("resampling!")
        resampleParticles(PARTICLES)
    
    correlations = np.zeros(PARTICLES['num'])
    
    _, plot = cv2.threshold(MAP['plot'], 127, 255, cv2.THRESH_BINARY)
    
    for i in range(PARTICLES['num']):
        x_w, y_w, _ = lidar2world(psi, theta, x_l, y_l, PARTICLES['poses'][i][0], PARTICLES['poses'][i][1], PARTICLES['poses'][i][2])        
        x_m, y_m = world2map(MAP, x_w, y_w)
        
        particle_plot = np.zeros(MAP['plot'].shape)
        particle_plot[y_m, x_m] = [0, 1, 0]

        correlations[i] = np.sum(np.logical_and(plot, particle_plot)) # switched x and y
    
    weights = scipy.special.softmax(correlations - np.max(correlations)) # np.multiply(PARTICLES['weights'], scipy.special.softmax(correlations)) # multiply or add or replace?

    if (np.count_nonzero(correlations) == 0):
        print("ALL ZERO CORRELATIONS")
    
    PARTICLES['weights'] = weights
    
    return

def resampleParticles(PARTICLES):
    # implemented low-variance resampling according to: https://robotics.stackexchange.com/questions/7705/low-variance-resampling-algorithm-for-particle-filter
    
    M = PARTICLES['num']
    new_poses = np.zeros(PARTICLES['poses'].shape)
    
    r = np.random.uniform(0, 1 / M)
    w = PARTICLES['weights'][0]
    i = 0
    j = 0
    
    for m in range(M):
        U = r + m / M
        while (U > w):
            i += 1
            w += PARTICLES['weights'][i]
        new_poses[j, :] = PARTICLES['poses'][i, :]
        j += 1

    PARTICLES['poses'] = new_poses
    PARTICLES['weights'] = np.ones(PARTICLES['num']) / PARTICLES['num'] 

    return

################################# TRANSFORMS #################################

b = 0.93 # distance from world to body in meters
h = 0.33 # distance from body to head
l = 0.15 # distance from head to lidar
k = 0.07 # distance from head to kinect

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    return x.T, y.T

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles.
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# Calculates Rotation Matrix given euler angles
def eulerAnglesToRotationMatrix(angles):
    R_x = rot_x(angles[0])

    R_y = rot_y(angles[1])

    R_z = rot_z(angles[2])

    R = np.matmul(R_z, np.matmul(R_y, R_x))
    return R

# example: (roll)
def rot_x(phi):
    R_x = np.array([[1,                     0,              0],
                    [0,         math.cos(phi), -math.sin(phi)],
                    [0,         math.sin(phi),  math.cos(phi)]
                    ])

    return R_x

# example: head angle (pitch)
def rot_y(theta):
    R_y = np.array([[ math.cos(theta),    0,      math.sin(theta)],
                    [               0,    1,                    0],
                    [-math.sin(theta),    0,      math.cos(theta)]
                    ])

    return R_y

# example: neck angle (yaw)
def rot_z(psi):
    R_z = np.array([[math.cos(psi),    -math.sin(psi),    0],
                    [math.sin(psi),     math.cos(psi),    0],
                    [            0,                 0,    1]
                    ])

    return R_z

def lidarToHeadTransform():
    h_T_l = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, l],
                      [0, 0, 0, 1]
                      ])
    return h_T_l

def kinectToHeadTransform():
    h_T_k = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, k],
                      [0, 0, 0, 1]
                      ])
    return h_T_k

# psi is left/right counter-clockwise NECK angle in z-axis (yaw)
# theta is up/down counter-clockwise HEAD angle in y-axis (pitch)
def headToBodyTransform(psi, theta):
    R = np.matmul(rot_z(psi), rot_y(theta))

    b_T_h = np.vstack((R, np.zeros((1,3))))
    b_T_h = np.hstack((b_T_h, np.array(([0], [0], [h], [1]))))
    return b_T_h

# psi is left/right counter-clockwise NECK angle in z-axis (yaw)
# theta is up/down counter-clockwise HEAD angle in y-axis (pitch)
def lidarToBodyTransform(psi, theta):
    b_T_h = headToBodyTransform(psi, theta)
    h_T_l = lidarToHeadTransform()

    b_T_l = np.matmul(b_T_h, h_T_l)
    return b_T_l

# psi is left/right counter-clockwise NECK angle in z-axis (yaw)
# theta is up/down counter-clockwise HEAD angle in y-axis (pitch)
def kinectToBodyTransform(psi, theta):
    b_T_h = headToBodyTransform(psi, theta)
    h_T_k = kinectToHeadTransform()

    b_T_k = np.matmul(b_T_h, h_T_k)
    return b_T_k

# x, y, yaw obtained from odometry
def bodyToWorldTransform(x, y, yaw):
    R = rot_z(yaw)

    w_T_b = np.vstack((R, np.zeros((1,3))))
    w_T_b = np.hstack((w_T_b, np.array(([x], [y], [b], [1]))))
    return w_T_b

# x, y, yaw obtained from odometry
# psi, theta obtained from head
def lidarToWorldTransform(psi, theta, x, y, yaw):
    w_T_b = bodyToWorldTransform(x, y, yaw)
    b_T_l = lidarToBodyTransform(psi, theta)

    w_T_l = np.matmul(w_T_b, b_T_l)
    return w_T_l

# x, y, yaw obtained from odometry
# psi, theta obtained from head
def kinectToWorldTransform(psi, theta, x, y, yaw):
    w_T_b = bodyToWorldTransform(x, y, yaw)
    b_T_k = kinectToBodyTransform(psi, theta)

    w_T_k = np.matmul(w_T_b, b_T_k)
    return w_T_k

def lidar2head(x_l, y_l):
    coordinates_l = np.vstack((np.vstack((x_l, y_l)), np.zeros((1, x_l.shape[1])), np.ones((1, x_l.shape[1]))))

    coordinates_h = np.matmul(lidarToHeadTransform(), coordinates_l)
    x_h = coordinates_h[0, :]
    y_h = coordinates_h[1, :]
    return (x_h, y_h)

def lidar2body(psi, theta, x_l, y_l):
    coordinates_l = np.vstack((np.vstack((x_l, y_l)), np.zeros((1, x_l.shape[1])), np.ones((1, x_l.shape[1]))))

    coordinates_b = np.matmul(lidarToBodyTransform(psi, theta), coordinates_l)
    x_b = coordinates_b[0, :]
    y_b = coordinates_b[1, :]
    return (x_b, y_b)

# *_curr variables come from cumulative delta_pose
def lidar2world(psi, theta, x_l, y_l, x_curr, y_curr, yaw_curr):
    coordinates_l = np.vstack((np.vstack((x_l, y_l)), np.zeros((1, x_l.shape[1])), np.ones((1, x_l.shape[1]))))

    coordinates_w = np.matmul(lidarToWorldTransform(psi, theta, x_curr, y_curr, yaw_curr), coordinates_l)
    x_w = coordinates_w[0, :]
    y_w = coordinates_w[1, :]
    z_w = coordinates_w[2, :]
    
    x_w = x_w[:, np.newaxis]
    y_w = y_w[:, np.newaxis]
    z_w = z_w[:, np.newaxis]
    
    # remove scans that are too close to the ground
    indValid = (z_w > 0.1)
    x_w = x_w[indValid]
    y_w = y_w[indValid]
    z_w = z_w[indValid]

    return (x_w, y_w, z_w)
    
    
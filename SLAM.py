# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:54:34 2020

@author: roume
"""
import time
import p2_utils as p2
import load_data as ld
import cv2
import numpy as np
from matplotlib import pyplot as plt

def getDataset(dataset):
    if (dataset == 0):
        joint_data = ld.get_joint("joint/train_joint0")
        lidar_data = ld.get_lidar("lidar/train_lidar0")
    #    rgd_data = ld.get_rgb("cam/RGB_0")
    #    depth_data = ld.get_depth("cam/DEPTH_0")
    #    exIR_RGB = ld.getExtrinsics_IR_RGB()
    #    IRCalib = ld.getIRCalib()
    #    RGBCalib = ld.getRGBCalib()
    elif (dataset == 1):
        joint_data = ld.get_joint("joint/train_joint1")
        lidar_data = ld.get_lidar("lidar/train_lidar1")
    elif (dataset == 2):
        joint_data = ld.get_joint("joint/train_joint2")
        lidar_data = ld.get_lidar("lidar/train_lidar2")
    elif (dataset == 3):
        joint_data = ld.get_joint("joint/train_joint3")
        lidar_data = ld.get_lidar("lidar/train_lidar3")
    #    rgd_data = ld.get_rgb("cam/RGB_3")
    #    depth_data = ld.get_depth("cam/DEPTH_3")
    #    exIR_RGB = ld.getExtrinsics_IR_RGB()
    #    IRCalib = ld.getIRCalib()
    #    RGBCalib = ld.getRGBCalib()
    elif (dataset == 4):
        joint_data = ld.get_joint("joint/train_joint4")
        lidar_data = ld.get_lidar("lidar/train_lidar4")
    #    rgd_data = ld.get_rgb("cam/RGB_4")
    #    depth_data = ld.get_depth("cam/DEPTH_4")
    #    exIR_RGB = ld.getExtrinsics_IR_RGB()
    #    IRCalib = ld.getIRCalib()
    #    RGBCalib = ld.getRGBCalib()
    
    return joint_data, lidar_data

def initializeSLAM(num_particles):
    MAP = p2.initializeMap(0.05, -25, -25, 20, 20)
    
    PARTICLES = p2.initializeParticles(num_particles)
    
    TRAJECTORY_w = {}
    TRAJECTORY_w['particle'] = []
    TRAJECTORY_w['odometry'] = []
    
    TRAJECTORY_m = {}
    TRAJECTORY_m['particle'] = []
    TRAJECTORY_m['odometry'] = []
    
    return MAP, PARTICLES, TRAJECTORY_w, TRAJECTORY_m

#################################### MAIN ####################################
if __name__ == "__main__":
    start_time = time.time()

    num_particles = [5, 10, 20, 40, 80]
    
    for d in range(5):
        joint_data, lidar_data = getDataset(d)
        MAP, PARTICLES, TRAJECTORY_w, TRAJECTORY_m = initializeSLAM(num_particles[d])

        for i in range(len(lidar_data)):
            
            # According to LIDAR sensor
            lidar_angles = np.array([np.arange(-135, 135.25, 0.25) * np.pi / 180.]).T
            lidar_scan = np.double(lidar_data[i]['scan'][0])
            
            # Remove scans that are too close (< 0.1), too far (> 30)
            good_range_ind = np.logical_and((lidar_scan < 30), (lidar_scan > 0.1))
            lidar_scan = lidar_scan[good_range_ind]
            lidar_angles = np.ndarray.flatten(lidar_angles[good_range_ind])
            
            # Record trajectories
            delta_pose = np.ndarray.flatten(lidar_data[i]['delta_pose'])
            
            if (i == 0):
                pose = PARTICLES['poses'][np.argmax(PARTICLES['weights']), :]
                
                TRAJECTORY_w['particle'].append(np.expand_dims(pose, axis=0))
                p_x_m, p_y_m = p2.world2map(MAP, TRAJECTORY_w['particle'][i][0][0], TRAJECTORY_w['particle'][i][0][1])
                TRAJECTORY_m['particle'].append(np.array([[p_x_m[0], p_y_m[0], TRAJECTORY_w['particle'][i][0][2]]]))
    
                TRAJECTORY_w['odometry'].append(lidar_data[i]['delta_pose'])
                o_x_m, o_y_m = p2.world2map(MAP, TRAJECTORY_w['odometry'][i][0][0], TRAJECTORY_w['odometry'][i][0][1])
                TRAJECTORY_m['odometry'].append(np.array([[o_x_m[0], o_y_m[0], TRAJECTORY_w['odometry'][i][0][2]]]))
            else:
                TRAJECTORY_w['odometry'].append(lidar_data[i]['delta_pose'] + TRAJECTORY_w['odometry'][i - 1])
                o_x_m, o_y_m = p2.world2map(MAP, TRAJECTORY_w['odometry'][i][0][0], TRAJECTORY_w['odometry'][i][0][1])
                TRAJECTORY_m['odometry'].append(np.array([[o_x_m[0], o_y_m[0], TRAJECTORY_w['odometry'][i][0][2]]]))
            
            # match lidar data to head_angles from joint_data using respective timestamps
            matching_index = np.argmin(np.abs(joint_data['ts'][0] - lidar_data[i]['t'][0]))
            neck_psi = joint_data['head_angles'][0][matching_index]
            head_theta = -joint_data['head_angles'][1][matching_index] # correct angle by multiplying with -1
            
            # xy lidar position vectors in the lidar frame (ground hits already filtered out)
            x_l, y_l = np.double(p2.pol2cart(lidar_scan, lidar_angles))
            x_w, y_w, _ = p2.lidar2world(neck_psi, head_theta, x_l, y_l, pose[0], pose[1], pose[2])
    
            # add first scan to map, then continue to next scan
            if (i == 0):
                p2.updateMap(MAP, x_w, y_w, pose[0], pose[1])
                continue
            
            # particle filter steps. Resample is called in the updateParticles() function
            p2.predictParticles(PARTICLES, delta_pose[0], delta_pose[1], delta_pose[2], pose[0], pose[1], pose[2])
            p2.updateParticles(PARTICLES, MAP, x_l, y_l, neck_psi, head_theta)
            
            # update trajectories again
            pose = PARTICLES['poses'][np.argmax(PARTICLES['weights']), :]
                
            TRAJECTORY_w['particle'].append(np.expand_dims(pose, axis=0))
            p_x_m, p_y_m = p2.world2map(MAP, TRAJECTORY_w['particle'][i][0][0], TRAJECTORY_w['particle'][i][0][1])
            TRAJECTORY_m['particle'].append(np.array([[p_x_m[0], p_y_m[0], TRAJECTORY_w['particle'][i][0][2]]]))
            
            # update map based on particle's view of lidar scan
            p2.updateMap(MAP, x_w, y_w, pose[0], pose[1])
    
            if ((i + 1) % 500 == 0 or i == len(lidar_data) - 1):
                plt.imshow(MAP['plot'])
                plt.scatter(np.asarray(TRAJECTORY_m['particle'])[:].T[0], np.asarray(TRAJECTORY_m['particle'])[:].T[1], c='r', marker="s")
                plt.scatter(np.asarray(TRAJECTORY_m['odometry'])[:].T[0], np.asarray(TRAJECTORY_m['odometry'])[:].T[1], c='b', marker="s")
                plt.title("dataset " + str(d) + " trajectory after " + str(i + 1) + " scans")
                filename = "d" + str(d) + "s" + str(i + 1) + "p" + str(num_particles[d]) + ".png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')                
                plt.show()
                
                plt.imshow(MAP['map'])
                plt.title("dataset " + str(d) + " log-odds after " + str(i + 1) + " scans")
                filename = "lo-d" + str(d) + "s" + str(i + 1) + "p" + str(num_particles[d]) + ".png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')                
                plt.show()
                
                print("occupancy grid and log-odds at scan =", i + 1, "of", len(lidar_data), "at time elapsed:", (time.time() - start_time))
                print("current particle position:", TRAJECTORY_m['particle'][i].T[0][0], ",", TRAJECTORY_m['particle'][i].T[1][0])
                print("current odometry position:", TRAJECTORY_m['odometry'][i].T[0][0], ",", TRAJECTORY_m['odometry'][i].T[1][0])        
This project is composed of the following files
	- load_data.py
	- p2_utils.py
	- SLAM.py
	
load_data.py
	- unchanged compared to given load_data.py

p2_utils.py
	- Contains three sections MAP, PARTICLES, TRANSFORMS:
	
	- MAP section contains functions initializeMap(), updateMap(), lidar2map(), and world2map()
		- initializeMap() returns a dictionary object MAP with several elements relating to the important thresholds of the map, our trust in the lidar scans, if we decay the map between updates. This dictionary also contains an RGB image 'plot' that is used for displaying the map.
		- updateMap() takes the world-frame coordinates of the lidar hits and the robot's current pose, and uses these to update the log-odds grid. We used cv2.drawContours, which is slightly less accurate but much faster than the Bresenham line algorithm, to decide which spots are occupied and which spots are free. We also perform a bounding of the log-odds, so that we prevent overconfidence in a cell (we allow faster recovery given several scans contradicting the previous information we had about the cell). And finally, we also update the 'plot'.
		- lidar2map() takes lidar hits in the lidar-frame and converts them to world-frame, then converts these to cell coordinates (map-coordinates) for us to update the log-odds grid
		- world2map() takes world-frame coordinates and converts them to map-coordinates
	- PARTICLES section contains functions initializeParticles(), predictParticles(), updateParticles(), and resampleParticles()
		- initializeParticles() returns a dictionary object that is used to store particle poses and their relative importance weights. It also contains 'num' the number of particles, 'n_thresh' which is a hyperparamater used to determine when to resample, and 'noise_cov' which is a 3-by-3 matrix used to generate noise for the particle poses in the predictParticles() function
		- predictParticles() uses the 'noise_cov' matrix to multiply the absolute values of the delta_pose values at the current timestep, and uses this to add noise to the particle poses and update the motion model.
		- updateParticles() determines if we need to resample by computing 'n_eff' the relative number of particles with significant weight with 'n_thresh'. After this, we enter a loop: for every particle, use the particle's pose to compute the current timestamp's lidar hits in this particle's pose. For all cells at these coordinates that are also occupied in 'plot', we sum them up and use this as a metric of correlation. To compute the relative weights, we set the weights = softmax(correlations - max(correlations)), and in this way we avoid numerical instability issues, and this gives better performance because we typically track the same particle for longer periods this way.
		- resampleParticles() simply uses the relative weights to resample particle poses, and then sets these new poses to have uniform weights. We used low-variance resampling.
	- TRANSFORMS section contains several methods whose job it is to convert between different physical coordinate systems and frames. 
	
SLAM.py
	- contains two functions getDataset() and initializeSLAM(), and main()
	- getDataset() isn't very interesting. It simply loads in and returns the proper dataset for the current run, so valid inputs are ints 0, 1, 2, 3, and 4. 
	- initializeSLAM() initializes the MAP and PARICLES dictionaries with some chosen values for the hyperparamaters (num_particles, lidar trust, log-odds thresholds, etc), as well as dictionaries used to track the trajectory of the robot (one in world coordinates TRAJECTORY_w, the other in map cell coordinates TRAJECTORY_m).
	- main() is where the magic happens. We wrapped it in a for-loop to loop over all the datasets. Inside this is another loop, over all the lidar-scans in the dataset. We use these scans, their closest matching head and neck angles (by finding the minimum difference between the lidar timestamp and the head angles timestamp), and the current relative odometry reading to obtain the coordinates of the lidar hits in the world-frame, perform the predictParticles() step, then the updateParticles() step, append the current pose to the TRAJECTORY variables, and finally we updateMap(). We print and export figures of the log-odds grid and the occupancy-grid (complete with trajectories) every 500 scans. 
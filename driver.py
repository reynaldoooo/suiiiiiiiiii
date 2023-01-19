import numpy as np
import time

from numpy.lib import gradient

class Driver:    

    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 5000000
    STRAIGHTS_SPEED = 8.0
    CORNERS_SPEED = 6.5
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees
    CLOSE_THRESHOLD = 5
    TOO_CLOSE_TRESHOLD = 0.5
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None
        self.visualiser_range = []
        self.obstacle_range = []
        self.best_point = 0
        self.best_speed = 0
    
    def set_bounds_i(self, array, index, r, i):
        min_index = index - r
        max_index = index + r
        if min_index < 0: min_index = 0
        if max_index >= len(array): max_index = len(array) - 1
        array = map(lambda x: array[x]*i, array)

    def find_speed_coeff(self, array, index, gap_start, gap_end):
        """
        A sharp corner is about (0.01 - 0.05)
        A wide corner is about (0.05 - 0.08)
        """
        min_index = index - 10
        max_index = index + 10
        if min_index < gap_start:
            min_index = gap_start
        if max_index > gap_end:
            max_index = gap_end
        curve_gradients = np.gradient(array[min_index:max_index])
        curve_mean = abs(np.mean(curve_gradients))
        curve_coeff = 1.8 * (np.power(curve_mean, (1.0/7.0))) # 1.6 is taking 0.04 as average velocity
        speed_coeff = curve_coeff #(curve_coeff*0.8)
        if speed_coeff > 1.2: speed_coeff = 1.2
        if speed_coeff < 0.7: speed_coeff = 0.7
        return speed_coeff, curve_coeff

    def find_aeb_coeff(self, proc_ranges):
        aeb_range = proc_ranges[248:572]
        aeb_coeff = 1
        if aeb_range.mean() < self.CLOSE_THRESHOLD:
            aeb_coeff = aeb_range.mean() / self.CLOSE_THRESHOLD
            if aeb_coeff < 0.3: 
                aeb_coeff = 0.3
            print("AEB")
        return aeb_coeff

    def optimise_speed(self, distance, base_speed, speed_coeff):
        opt_speed = base_speed + 0.05*(distance**2)
        opt_speed_speed = opt_speed * speed_coeff
        return opt_speed_speed

    def preprocess_lidar(self, ranges, visual):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2*np.pi) / len(ranges)
	    # we won't use the LiDAR data from directly behind us
        if visual == False:
            proc_ranges = np.array(ranges[135:-135])
        else:
            proc_ranges = ranges
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_obstacles(self, proc_ranges):
        #window = proc_proc_rangess[270:540] 
        gradients_raw = list(map(lambda x: abs(x), np.gradient(proc_ranges)))
        gradients_raw = list(map(lambda x: abs(x), np.gradient(gradients_raw)))
        gradients = np.convolve(gradients_raw, np.ones_like(proc_ranges), 'same')
        for i in range(0, len(proc_ranges)):
            gradients[i] = gradients[i] / proc_ranges[i]
        tolerance_point = abs(np.mean(gradients)*2)
        #print(tolerance_point, np.max(gradients))
        obstacle_points = np.argwhere(gradients > tolerance_point)
        #print(obstacle_points)
        self.obstacle_range = obstacle_points

    def find_gaps(self, free_space_ranges):
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges<self.TOO_CLOSE_TRESHOLD, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)  
        return slices

    def slice_score(self, proc_ranges, slice):
        gap_start = slice.start
        gap_end = slice.stop
        gap_size = gap_end - gap_start
        width_score = (gap_size / 810)
        distance_score = proc_ranges[gap_start:gap_end].mean() / proc_ranges.mean()
        slice_score = (width_score*0.7) + (distance_score*0.3)
        return slice_score

    def find_best_point_mk2(self, proc_ranges):
        slices = self.find_gaps(proc_ranges)
        if len(slices) == 1:
            gap_start = slices[0].start
            gap_end = slices[0].stop
            best = self.find_best_point(gap_start, gap_end, proc_ranges)
            return best, gap_start, gap_end
        else:
            chosen_slice = slices[0]
            # self.sclice_score() takes in gap size and distances of slice and produces a "slice_score"
            max_slice_score = self.slice_score(proc_ranges, slices[0])
            for slice in slices:
                slice_score = self.slice_score(proc_ranges, slice)
                # best slice score wins
                if slice_score > max_slice_score:
                    max_slice_score = slice_score
                    chosen_slice = slice
            gap_start = chosen_slice.start
            gap_end = chosen_slice.stop
            best = self.find_best_point(gap_start, gap_end, proc_ranges)
            return best, gap_start, gap_end
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(int(self.BEST_POINT_CONV_SIZE*1)), 'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle
        """
        lidar_angle = (range_index - (range_len/2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges, False)
        self.visualiser_range = self.preprocess_lidar(ranges, True)
    
        # Finds closest point and set to zero
        closest = np.argmin(proc_ranges)
        self.set_bounds_i(proc_ranges, closest, self.BUBBLE_RADIUS, 0.2)

        # Find points that are "too_close" and set to i ( multi-agent racing )
        too_close = np.argwhere(proc_ranges < self.CLOSE_THRESHOLD).flatten()
        for close in too_close:
            self.set_bounds_i(proc_ranges, close, self.BUBBLE_RADIUS, 0.5)

        #Finds best point to travel to
        good_gap = False
        while not good_gap:
            best, gap_start, gap_end = self.find_best_point_mk2(proc_ranges)
            #check gap size will fit car
            gap_size = gap_end - gap_start
            print(gap_size)
            if gap_size < 400:
                proc_ranges[gap_start:gap_end] = 0
                print("gap too small")
            else:
                good_gap = True
            

        self.best_point = best

        # Look for "Obstacles"
        self.find_obstacles(proc_ranges)

        #Find the sharpness of the turn
        speed_coeff, curve_coeff = self.find_speed_coeff(proc_ranges, best, gap_start, gap_end)

        #Find AEB_coeff
        aeb_coeff = self.find_aeb_coeff(proc_ranges)

        #Optimise Speed
        best_speed = self.optimise_speed(proc_ranges[best], self.CORNERS_SPEED - 2, speed_coeff)
        #best_speed *= aeb_coeff

        #Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            if curve_coeff > 0.85:
                speed = best_speed * 0.8
            else:
                speed = best_speed * 0.8
            print("CORNER")
        else: 
            speed = best_speed
            #print("STRAIGHT")
        # print('Steering angle in degrees: {}'.format((steering_angle/(np.pi/2))*90))
        #print(f"Speed : {speed}")
        self.best_speed = speed
        return speed, steering_angle

    def get_visualiser_ranges(self):
        return self.visualiser_range, self.obstacle_range, self.best_point, self.best_speed

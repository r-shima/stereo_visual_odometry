"""
Implementation of stereo visual odometry on the KITTI dataset

References: https://ieeexplore.ieee.org/document/6096039
            https://ieeexplore.ieee.org/document/6153423
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class StereoVisualOdometry():
    def __init__(self, dataset_path, sequence):
        self.left_images = self.get_images(dataset_path, 'image_0')
        self.right_images = self.get_images(dataset_path, 'image_1')
        pose_path = 'data_odometry_poses/dataset/poses/' + sequence + '.txt'
        calib_path = 'data_odometry_calib/dataset/sequences/' + sequence + '/calib.txt'
        self.ground_truth = self.get_ground_truth(pose_path)
        self.P_left, self.P_right = self.get_calib(calib_path)
        self.K_left, self.R_left, self.T_left = self.extract_params(self.P_left)
        self.K_right, self.R_right, self.T_right = self.extract_params(self.P_right)

    def get_images(self, dataset_path, image_dir, num_images=None):
        """
        Gets the images

        Args:
            dataset_path: the path to the image sequence
            image_dir: the image directory
            num_images: the number of images to load

        Returns:
            images: a list of images
        """
        image_folder = os.path.join(dataset_path, image_dir)

        if num_images == None:
            image_files = sorted(os.listdir(image_folder))
        else:
            image_files = sorted(os.listdir(image_folder))[:num_images]

        images = []
        for filename in image_files:
            image_path = os.path.join(image_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            images.append(image)

        return images

    def get_ground_truth(self, pose_path):
        """
        Gets the ground truth poses

        Args:
            pose_path: the path to the pose file

        Returns:
            poses: an array of ground truth poses
        """
        poses = np.loadtxt(pose_path, dtype=np.float64)
        poses = poses.reshape(-1, 3, 4)
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 1, 4)) +
                                np.array([[0, 0, 0, 1]])], axis=1)

        return poses

    def get_calib(self, calib_path):
        """
        Gets the projection matrices

        Args:
            calib_path: the path to the calibration file

        Returns:
            P_left: the projection matrix for the left camera
            P_right: the projection matrix for the right camera
        """
        P_left_vals = []
        P_right_vals = []

        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line.startswith('P0:'):
                    line = line.replace('P0:', '')
                    P_left_vals = np.fromstring(line, dtype=np.float64, sep=' ')
                elif line.startswith('P1:'):
                    line = line.replace('P1:', '')
                    P_right_vals = np.fromstring(line, dtype=np.float64, sep=' ')

        P_left = np.reshape(P_left_vals, (3, 4))
        P_right = np.reshape(P_right_vals, (3, 4))

        return P_left, P_right

    def extract_params(self, P):
        """
        Extracts the intrinsic matrix, rotation matrix, and translation vector

        Args:
            P: the projection matrix

        Returns:
            K: the intrinsic matrix
            R: the rotation matrix
            trans_vec: the 3D translation vector
        """
        K, R, trans_vec = cv.decomposeProjectionMatrix(P)[:3]

        # Turn the fourth value into 1
        trans_vec = (trans_vec / trans_vec[3])[:3]

        return K, R, trans_vec

    def find_features(self, image, detector_type='sift', mask=None):
        """
        Finds the features of an image

        Args:
            image: the image to find features on
            detector_type: the detector to use
            mask: the mask

        Returns:
            keypoints: a tuple of keypoints
            descriptors: an array of descriptors
        """
        if detector_type == 'orb':
            detector = cv.ORB_create()
        elif detector_type == 'sift':
            detector = cv.SIFT_create()

        keypoints, descriptors = detector.detectAndCompute(image, mask)
        # kp_image = cv.drawKeypoints(image, keypoints, None)
        # cv.imshow('Keypoints', kp_image)
        # cv.waitKey()

        return keypoints, descriptors

    def find_matches(self, prev_descriptors, descriptors, matching_type='bf', detector_type='sift',
                     sort=True):
        """
        Find matching features

        Args:
            prev_descriptors: the descriptor for previous frame
            descriptors: the descriptor for current frame
            matching_type: the type of matcher
            detector_type: the detector to use

        Returns:
            matches: a tuple of matches
        """
        if matching_type == 'bf':
            if detector_type == 'orb':
                method = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
            elif detector_type == 'sift':
                method = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        
        matches = method.knnMatch(prev_descriptors, descriptors, k=2)

        if sort:
            # Sort the matches according to their distance
            # NOTE: Smaller distance means better match and longer distance means not a good match
            matches = sorted(matches, key = lambda x:x[0].distance)

        return matches

    def generate_disparity_map(self, left_image, right_image, type):
        """
        Generates a disparity map by using the left and right images

        Args:
            left_image: the image of the left camera
            right_image: the image of the right camera
            type: the type of stereo matching

        Returns:
            disparity: the disparity map
        """
        if type == 'bm':
            bm = cv.StereoBM_create(numDisparities=6*16, blockSize=11)
            
        elif type == 'sgbm':
            # P1 is typically 8*number_of_image_channels*blockSize*blockSize
            # P2 is typically 32*number_of_image_channels*blockSize*blockSize
            bm = cv.StereoSGBM_create(numDisparities=6*16, minDisparity=0, blockSize=11,
                                      P1 = 8 * 1 * 11 ** 2, P2 = 32 * 1 * 11 ** 2,
                                      uniquenessRatio=10, speckleWindowSize=125,
                                      mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

        # Cast the result to a float and divide by 16 to get original values
        disparity = bm.compute(left_image, right_image).astype(np.float32) / 16

        return disparity

    def generate_depth_map(self, K_left, T_left, T_right, disparity):
        """
        Generates a depth map

        Args:
            K_left: the intrinsic matrix for the left camera
            T_left: the translation vector for the left camera
            T_right: the translation vector for the right camera
            disparity: the disparity map

        Returns:
            depth: the depth map
        """
        # Get focal length of x direction
        focal_length = K_left[0][0]

        # Calculate the baseline
        baseline = abs(T_left[0] - T_right[0])

        # Adjust the disparity to avoid doing division by zero and getting a negative depth
        # NOTE: 0 and -1 in disparity indicate that there is no overlap between the images
        disparity = np.where((disparity == 0.0) | (disparity == -1.0), 0.1, disparity)

        depth = focal_length * baseline / disparity

        return depth

    def get_depth_from_images(self, left_image, right_image, type='sgbm'):
        """
        Gets the depth map from stereo images

        Args:
            left_image: the image of the left camera
            right_image: the image of the right camera
            type: the type of stereo matching

        Returns:
            depth: the depth map
        """
        disparity = self.generate_disparity_map(left_image, right_image, type)
        depth = self.generate_depth_map(self.K_left, self.T_left, self.T_right, disparity)

        return depth
    
    def generate_mask(self, left_image, right_image, type='sgbm'):
        """
        Generates a mask for finding features

        Args:
            left_image: the image of the left camera
            right_image: the image of the right camera
            type: the type of stereo matching

        Returns:
            mask: the mask
        """
        depth = self.get_depth_from_images(left_image, right_image, type)
        mask = np.zeros(depth.shape, dtype=np.uint8)
        height = depth.shape[0]
        width = depth.shape[1]
        cv.rectangle(mask, (96, 0), (width, height), 255, thickness = -1)

        return mask
    
    def view_matches(self, matches, prev_keypoints, keypoints, prev_image, image):
        """
        Views the matches between features

        Args:
            matches: the matched features from previous and current images
            prev_keypoints: a tuple of keypoints of the previous left image
            keypoints: a tuple of keypoints of the current left image
            prev_image: the previous left image
            image: the current left image

        Returns: None
        """
        output = cv.drawMatches(prev_image, prev_keypoints, image, keypoints, matches, None, 
                                flags=2)
        plt.figure(figsize=(16, 6), dpi=100)
        plt.imshow(output)
        plt.show()

    def perform_ratio_test(self, matches, distance_ratio=0.45):
        """
        Performs Lowe's ratio test to filter out unreliable matches between features
        Reference: https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html

        Args:
            matches: the matched features from previous and current images
            distance_ratio: the ratio used as a threshold to determine good matches

        Returns:
            good_matches: a list of good matches
        """
        good_matches = []
        for m, n in matches:
            if m.distance <= distance_ratio * n.distance:
                good_matches.append(m)

        return good_matches

    def compute_camera_motion(self, good_matches, prev_keypoints, keypoints, depth, cutoff=3000):
        """
        Computes the camera motion between the previous image and the current image

        Args:
            good_matches: a list of good matches
            prev_keypoints: a tuple of keypoints of the previous left image
            keypoints: a tuple of keypoints of the current left image
            depth: the depth map
            cutoff: the cutoff value for the depth

        Returns:
            R: the rotation matrix
            trans_vec: the translation vector
        """
        prev_coord = []
        coord = []

        # Get arrays of 2D coordinates of the features that were matched
        for mat in good_matches:
            prev_coord.append(prev_keypoints[mat.queryIdx].pt)
            coord.append(keypoints[mat.trainIdx].pt)

        prev_coord = np.float32(prev_coord)
        coord = np.float32(coord)

        # Extract parameters from intrinsic matrix
        center_x = self.K_left[0, 2]
        center_y = self.K_left[1, 2]
        focal_length_x = self.K_left[0, 0]
        focal_length_y = self.K_left[1, 1]

        # Initialize array to store 3D coordinates
        coord_3d = np.zeros((0, 3))

        # Initialize list to store points that will be removed
        points_to_remove = []

        for i in range(0, len(prev_coord)):
            # Get the pixel coordinates
            u, v = prev_coord[i]

            # Extract the pixel-wise depth from the depth map
            d = depth[int(v), int(u)]

            # If the depth is above the cutoff value
            if d > cutoff:
                # Add the index of the feature with bad depth estimate
                points_to_remove.append(i)
                continue

            # Get x and y coordinates from pixel coordinates
            x = (u - center_x) * d / focal_length_x
            y = (v - center_y) * d / focal_length_y

            # z coordinate is the depth
            z = d

            coord_3d = np.vstack((coord_3d, np.array([x, y, z])))

        # Remove points stored in points_to_remove
        prev_coord = np.delete(prev_coord, points_to_remove, 0)
        coord = np.delete(coord, points_to_remove, 0)

        # Get the rotation and translation vectors
        # NOTE: Basically solving for the change in camera pose from the previous frame to the
        # current frame
        rot_vec, trans_vec = cv.solvePnPRansac(coord_3d, coord, self.K_left, None)[1:3]

        # Turn the rotation vector into a rotation matrix
        R = cv.Rodrigues(rot_vec)[0]

        return R, trans_vec
    
    def run_pipeline(self, matching_type='bf', detector_type='sift', stereo_matching_type='sgbm',
                     distance_ratio=0.45):
        """
        Runs the stereo visual odometry pipeline

        Args:
            matching_type: the type of matcher
            detector_type: the detector to use
            stereo_matching_type: the type of stereo matching
            distance_ratio: the ratio used as a threshold to determine good matches

        Returns:
            full_traj: the full trajectory of the camera
        """
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        x_path = self.ground_truth[:, 0, 3]
        y_path = self.ground_truth[:, 1, 3]
        z_path = self.ground_truth[:, 2, 3]

        # Set x, y, and z axes to be on the same scale
        ax.set_box_aspect((np.ptp(x_path), np.ptp(y_path), np.ptp(z_path)))

        ax.plot(x_path, y_path, z_path, c='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Create a homogeneous transformation matrix
        T_hom = np.eye(4)

        # Initialize an array to store the full trajectory of the camera
        full_traj = np.zeros((len(self.left_images), 3, 4))

        # Store the first pose
        full_traj[0] = T_hom[:3, :]

        # Get a mask for finding features
        mask = self.generate_mask(self.left_images[0], self.right_images[0], stereo_matching_type)

        # Set the current left and right images to the first images in the respective lists
        left_image = self.left_images[0]
        right_image = self.right_images[0]

        for i in range(0, len(self.left_images)-1):
            # Set the previous left and right images
            prev_left_image = left_image
            prev_right_image = right_image

            # Set the current left and right images
            left_image = self.left_images[i+1]
            right_image = self.right_images[i+1]

            # Detect features from the previous and current left images
            prev_left_keypoints, prev_left_descriptors = self.find_features(prev_left_image,
                                                                            detector_type, mask)
            left_keypoints, left_descriptors = self.find_features(left_image, detector_type, mask)

            # Find matching features from the previous and current left images
            matches = self.find_matches(prev_left_descriptors, left_descriptors, matching_type,
                                        detector_type)

            # Apply Lowe's ratio test to get reliable matches
            good_matches = self.perform_ratio_test(matches, distance_ratio)

            # Get a depth map
            depth = self.get_depth_from_images(prev_left_image, prev_right_image,
                                               stereo_matching_type)

            # Compute the camera motion between previous and current left images
            R, trans_vec = self.compute_camera_motion(good_matches, prev_left_keypoints,
                                                      left_keypoints, depth)
            
            # Store the rotation matrix and translation vector in a homogeneous transformation matrix
            T_i = np.eye(4)
            T_i[:3, :3] = R
            T_i[:3, 3] = trans_vec.T

            # Relate the coordinates in the camera frame to the global frame (inverse of what is
            # given by solvePnPRansac)
            T_hom = T_hom.dot(np.linalg.inv(T_i))

            # Add the camera pose estimate to the trajectory
            full_traj[i+1, :, :] = T_hom[:3, :]

            x_path = full_traj[:i+2, 0, 3]
            y_path = full_traj[:i+2, 1, 3]
            z_path = full_traj[:i+2, 2, 3]
            ax.plot(x_path, y_path, z_path, c='m')
            plt.pause(0.0000000000000000001)

            cv.imshow('Camera View', left_image)
            cv.waitKey(1)

        return full_traj

def main():
    dataset_path = 'data_odometry_gray/dataset/sequences/09'
    svo = StereoVisualOdometry(dataset_path, sequence='09')

    # NOTE: When using orb, set distance_ratio to 0.6 and stereo_matching_type to bm
    full_traj = svo.run_pipeline(matching_type='bf', detector_type='sift',
                                 stereo_matching_type='sgbm', distance_ratio=0.45)

if __name__ == '__main__':
    main()
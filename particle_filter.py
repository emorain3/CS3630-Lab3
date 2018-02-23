from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
from numpy.random import choice


def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []

    # If the robot hasn't moved just return particles
    if sum(odom) == 0:
        return particles

    for particle in particles:
        # update particle distance, angle, header according to odom measurements
        noisy_odom = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)
        x_dist_adj = particle.x + odom[0] + noisy_odom[0]
        y_dist_adj = particle.y + odom[1] + noisy_odom[1]
        new_h = particle.h + diff_heading_deg(particle.h, odom[2])

        new_x, new_y = rotate_point(x_dist_adj, y_dist_adj, new_h)
        new_particle = Particle(new_x, new_y, new_h)

        motion_particles.append(new_particle)

    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one

        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """

    #update weight of each particle according to how closely the markeres
    # it sees matches the markers that the robot sees
    # What is meant by weights?

    measured_particles = []
    particle_accuracies = []

    # for each particle measure the markers and organize such that the particles markers are
    #  a best fit comparison to the robot markers



    # fill accuracy list with the measured accuracy of each particle
    # Here accuracy of a particle is evaluated by measuring the difference in the markers the particle sees
    #   versus the particles that the robot sees
    for particle in particles:
        particle_markers_list = particle.read_markers(grid)
        prob = 1.0

        # Update probability score for this particle
        for marker in particle_markers_list:
            # Compare to robot marker with greatest similarity/proximity
            robot_marker = get_closest_marker(marker, measured_marker_list)
            dist_diff = grid_distance(marker[0], marker[1], robot_marker[0],robot_marker[1])
            angle_diff = diff_heading_deg(marker[2], robot_marker[2])
            prob *= np.exp(-1*(((dist_diff**2) / (2 * MARKER_TRANS_SIGMA **2)) + ((angle_diff**2) / (2 * MARKER_ROT_SIGMA**2))))



    return measured_particles


# This is very inelegant code, but it is all that I have in me right now.
def get_closest_marker(marker, objective_marker_list):

    closest_marker = objective_marker_list[0]
    shortest_dist = grid_distance(marker[0], marker[1], objective_marker_list[0][0], objective_marker_list[0][1])
    for i in range(objective_marker_list.len()):
        if(grid_distance(marker[0], marker[1], objective_marker_list[i][0], objective_marker_list[i][1]) < shortest_dist):
            closest_marker = objective_marker_list[i]
    return closest_marker



def resample_particles(particles, particle_prob_list, grid):

    particle_indicies = np.array(range(0,len(particle_prob_list),1))

    #Create the distribution
    particle_prob_dist = [0.0]
    i = 0
    for prob in particle_prob_list:
        particle_prob_dist.append(prob + particle_prob_dist[i])
        i += 1

    #Sample the indicies with replacement
    resampled_indicies = choice(particle_indicies, len(particle_indicies), p=particle_prob_dist) #read the manual pages to see if the distribution is accepted
    resampled_indicies = np.array(resampled_indicies)

    #Check the probability distribution out of indicies pulled
    a = particle_prob_list[resampled_indicies]

    #Flag indicies that are below the threshold

    #Pull the newly sample particles

    #Replace the particles with flaged indicies with leagal random particles

    # Eliminate particles of low probability and create a random sample from what arledy exists
    threshold = 0.000001  # Tweak This
    num_of_random_samples = len(particle_prob_list[particle_prob_list < threshold])

    # Replace Low Prob. Particles With Random Particles
    if num_of_random_samples > 0:
        a = 1
        # Replace particle prob list to after its sampled
        # measured_particles[particle_prob_list < threshold] = np.array(
        #     Particle.create_random(num_of_random_samples, grid=grid))

    # Keep a small sample of particles random
    # Check if each particle is in a legal positon if not set to zero
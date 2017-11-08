/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath> 
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 10;
  weights.resize(num_particles, 1.);
  particles.reserve(num_particles);
  std::mt19937_64 gen;
  std::normal_distribution<> dist_x(x, std[0]);
  std::normal_distribution<> dist_y(y, std[1]);
  std::normal_distribution<> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; ++i) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    double theta = dist_theta(gen);
    particles.emplace_back(Particle{i, x, y, theta, 1., {}, {}, {}});
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  std::mt19937_64 gen;
  std::normal_distribution<> dist_x(0., std_pos[0]);
  std::normal_distribution<> dist_y(0., std_pos[1]);
  std::normal_distribution<> dist_theta(0., std_pos[2]);
  for (auto&& p : particles) {
    double coef = velocity/yaw_rate;
    double delta_theta = p.theta + delta_t*yaw_rate;

    p.x += coef*(sin(delta_theta) - sin(p.theta)) + dist_x(gen);
    p.y += coef*(cos(p.theta) - cos(delta_theta)) + dist_y(gen);
    p.theta = delta_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (auto&& pred : predicted) {
    for (auto&& obs : observations) {
    }
  }
}

LandmarkObs carToMap(const Particle& p, const LandmarkObs& obs) {
  double cos_t = std::cos(p.theta);
  double sin_t = std::sin(p.theta);
  double x_m = p.x + cos_t*obs.x - sin_t*obs.y;
  double y_m = p.y + sin_t*obs.x + cos_t*obs.y;
  return {obs.id, x_m, y_m};
}

int nearestLandmark(const LandmarkObs& obs, const Map& map) {
  double min_dist = dist(obs.x, obs.y, map.landmark_list[0].x_f, map.landmark_list[0].y_f);
  int min_idx = 0;
  int idx = 0;
  for (auto&& l : map.landmark_list) {
    double dist_l = dist(obs.x, obs.y, l.x_f, l.y_f);
    if (dist_l < min_dist) {
      min_dist = dist_l;
      min_idx = idx;
    }
    ++idx;
  }
  return min_idx;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  double x[2];
  double mu[2];
  int p_i = 0;
  for (auto&& p : particles) {
    double weight = 1.;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    for (auto&& o : observations) {
      auto map_obs = carToMap(p, o);
      int nearest = nearestLandmark(map_obs, map_landmarks);
      x[0] = map_obs.x;
      x[1] = map_obs.y;
      auto& l = map_landmarks.landmark_list[nearest];
      mu[0] = l.x_f;
      mu[1] = l.y_f;
      weight *= gaussian(x, mu, std_landmark);
      p.associations.push_back(l.id_i);
      p.sense_x.push_back(map_obs.x);
      p.sense_y.push_back(map_obs.y);
    }
    p.weight = weight;
    weights[p_i++] = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::mt19937_64 gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  new_particles.reserve(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    new_particles.emplace_back(particles[d(gen)]);
  }
  particles = std::move(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

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
  num_particles = 100;
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
    particles.emplace_back(Particle{ i, x, y, theta, 1., {}, {}, {} });
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  std::mt19937_64 gen;
  std::normal_distribution<> dist_x(0., std_pos[0]);
  std::normal_distribution<> dist_y(0., std_pos[1]);
  std::normal_distribution<> dist_theta(0., std_pos[2]);
  for (auto&& p : particles) {
    double delta_theta = p.theta + delta_t*yaw_rate;
    if (std::fabs(yaw_rate) > 1e-4) {
      double coef = velocity / yaw_rate;
      p.x += coef*(sin(delta_theta) - sin(p.theta)) + dist_x(gen);
      p.y += coef*(cos(p.theta) - cos(delta_theta)) + dist_y(gen);
    }
    else {
      p.x += velocity*cos(p.theta)*delta_t + dist_x(gen);
      p.y += velocity*sin(p.theta)*delta_t + dist_y(gen);
    }
    p.theta = delta_theta + dist_theta(gen);
  }
}

LandmarkObs carToMap(const Particle& p, const LandmarkObs& obs) {
  double cos_t = std::cos(p.theta);
  double sin_t = std::sin(p.theta);
  double x_m = p.x + cos_t*obs.x - sin_t*obs.y;
  double y_m = p.y + sin_t*obs.x + cos_t*obs.y;
  return { obs.id, x_m, y_m };
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

int nearestLandmark(const LandmarkObs& obs, const Map& map, const std::vector<int>& lms_in_range) {
  double min_dist = std::numeric_limits<double>::max();
  int min_idx = -1;
  for (int l_i : lms_in_range) {
    auto& l = map.landmark_list[l_i];
    double dist_l = dist(obs.x, obs.y, l.x_f, l.y_f);
    if (dist_l < min_dist) {
      min_dist = dist_l;
      min_idx = l_i;
    }
  }
  return min_idx;
}

// Filter all landmarks that are outside the given range. 
std::vector<int> getLandmarksInRange(double x, double y, double range, const Map& map) {
  std::vector<int> lms_in_range;
  int ind = 0;
  for (auto&& l : map.landmark_list) {
    if (dist(x, y, l.x_f, l.y_f) < range) {
      lms_in_range.push_back(ind);
    }
    ++ind;
  }
  return lms_in_range;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  double x[2];
  double mu[2];
  int p_i = 0;
  for (auto&& p : particles) {
    double weight = 1.;
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
    // If the difference between a landmark and an observation is more than 10m,
    // the Gaussian is numerically zero for a stddev of {0.3, 0.3} meters
    auto lms_in_range = getLandmarksInRange(p.x, p.y, sensor_range + 10., map_landmarks);
    // If there are no landmarks in range then the weight is zero
    if (lms_in_range.size() > 0) {
      for (auto&& o : observations) {
        // Transform from car coord system to map coord system
        auto map_obs = carToMap(p, o);
        // Find nearest landmark on the map using nearest neighbours
        int nearest = nearestLandmark(map_obs, map_landmarks, lms_in_range);
        x[0] = map_obs.x;
        x[1] = map_obs.y;
        auto& l = map_landmarks.landmark_list[nearest];
        mu[0] = l.x_f;
        mu[1] = l.y_f;
        // Calculate weight for current observation using the PDF of the normal distribution centered on the nearest landmark
        weight *= gaussian(x, mu, std_landmark);
        p.associations.push_back(l.id_i);
        p.sense_x.push_back(map_obs.x);
        p.sense_y.push_back(map_obs.y);
      }
    }
    else {
      weight = 0.;
    }
    p.weight = weight;
    weights[p_i++] = weight;
  }
}

void ParticleFilter::resample() {
  std::mt19937_64 gen;
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  new_particles.reserve(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    new_particles.emplace_back(particles[d(gen)]);
  }
  particles = std::move(new_particles);
}

std::string ParticleFilter::getAssociations(const Particle& best)
{
  const std::vector<int>& v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseX(const Particle& best)
{
  const std::vector<double>& v = best.sense_x;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
std::string ParticleFilter::getSenseY(const Particle& best)
{
  const std::vector<double>& v = best.sense_y;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

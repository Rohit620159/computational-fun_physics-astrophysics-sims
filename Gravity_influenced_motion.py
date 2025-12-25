import numpy as np

G = 6.674 * 10**-11
def gravitational_force(m ,n, r):
  force = (G*m*n)/(r**2)
  return force
def instanteneous_velocity(force, velocity, mass, dt)
  acceleration = force/mass
  new_velocity = velocity + acceleration * dt
  return new_velocity
def instanteneous_position(velocity, dt, position)
  new_position = position + velocity*dt
  return new_position

m = 5
n = 3
r = 2
force = gravitational_force(m, n, r)

# Print the result
print("Gravitational Force:", force)

import numpy as np
import numpy.linalg as LA
import numpy.matlib as MA
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal


class ParticleFilter():
    def __init__(self, Q, R, f, h):
        self.Q = Q
        self.R = R
        self.f = f
        self.h = h
        self.particles = {}

    def initialize(self, n, mean, cov):
        self.n_particles = n

        # weight initialization
        self.particles['w'] = (np.ones((1, self.n_particles)) / self.n_particles)

        # state initialization
        self.n_state = mean.shape[0]
        self.particles['x'] = MA.repmat(mean, 1, self.n_particles) + LA.cholesky(cov) @ \
            np.random.randn(self.n_state, self.n_particles)
            
        self.X = np.average(self.particles['x'], weights=self.particles['w'].flatten(), axis=1).reshape(-1,1)
        self.P = np.diag(np.average((self.particles['x'] - self.X)**2, weights=self.particles['w'].flatten(), axis=1))

    def predict(self, u, f_B):
        for i in range(self.n_particles):
            self.particles['x'][:, i] = (self.f(self.particles['x'][:, i], u, f_B) + \
                sqrtm(self.Q) @ np.random.randn(self.n_state, 1)).flatten()

        self.X = np.average(self.particles['x'], weights=self.particles['w'].flatten(), axis=1).reshape(-1, 1)
        self.P = np.diag(np.average((self.particles['x'] - self.X)**2, weights=self.particles['w'].flatten(), axis=1))

        state_pred = self.X
        cov_pred = self.P
        return state_pred, cov_pred

    def update(self, z, u, f_B):
        ratio = 0.5

        w = self.likelihood(z, u, f_B)

        self.particles['w'] = np.multiply(self.particles['w'], w)
        self.particles['w'] += 1e-300 # avoid round-off to zero
        self.particles['w'] = np.divide(self.particles['w'], np.sum(self.particles['w'])) # normalize

        self.neff_particles = 1 / np.sum(np.square(self.particles['w']))
        if self.neff_particles / self.n_particles < ratio:
            self.systematic_resampling()

        self.X = np.average(self.particles['x'], weights=self.particles['w'].flatten(), axis=1).reshape(-1, 1)
        self.P = np.diag(np.average((self.particles['x'] - self.X)**2, weights=self.particles['w'].flatten(), axis=1))
            
        state_est = self.X
        cov_est = self.P
        return state_est, cov_est

    def likelihood(self, z, u, f_B):
        w = np.zeros((1, self.n_particles))
        for i in range(self.n_particles):
            z_hat = self.h(self.particles['x'][:, i], u, f_B).flatten()
            w[0][i] = multivariate_normal.pdf(z.flatten(), z_hat, self.R)

        return w

    def systematic_resampling(self):
        """Performs the systemic resampling algorithm used by particle filters.
        
        This algorithm separates the sample space into N divisions. A single random
        offset is used to to choose where to sample from for all divisions. This
        guarantees that every sample is exactly 1/N apart.
        """
        N = self.particles['w'].shape[1]
        # make N subdivisions, and chose a random position within each one
        positions = (np.arange(0, N) + np.random.rand()) / N
        
        indices = np.zeros(N, dtype=np.int32)
        cumulative_sum = np.cumsum(self.particles['w'])
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particles['x'] = self.particles['x'][:, indices].reshape(self.n_state, -1)
        self.particles['w'] = np.ones((1, self.n_particles)) / self.n_particles

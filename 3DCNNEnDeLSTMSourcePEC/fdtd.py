import numpy as np


# Yee-Grid
class Grid(object):
	def __init__(self, dim=64):
		self.height = dim
		self.width = dim
		self.ez = np.zeros((self.height, self.width), dtype=float)
		self.train_data = []  # train data: dz, hx, hy
		self.target_data = [] # ez
		self.dz = np.zeros((self.height, self.width), dtype=float)
		self.hx = np.zeros((self.height, self.width), dtype=float)
		self.hy = np.zeros((self.height, self.width), dtype=float)
		self.iz = np.zeros((self.height, self.width), dtype=float)
		self.ihx = np.zeros((self.height, self.width), dtype=float)
		self.ihy = np.zeros((self.height, self.width), dtype=float)
		self.ez_inc = np.zeros(self.width, dtype=float)
		self.hx_inc = np.zeros(self.width, dtype=float)
		self.gaz = np.ones((self.height, self.width), dtype=float)  # train data
		self.gbz = np.zeros((self.height, self.width), dtype=float)

		# object circle center
		hc = int(self.height / 2 - 1)
		wc = int(self.width / 2 - 1)

		# pml bound
		self.ha = 7
		self.hb = self.height - self.ha - 1
		self.wa = 7
		self.wb = self.width - self.wa - 1

		self.total_field_range = np.arange(self.ha + 1, self.hb)

		ddx = 0.01  # Cell size
		dt = ddx / 6e8  # Time step size

		# Specify the dielectric cylinder
		epsr = 1#30
		sigma = 1000
		radius = 6
		# Create Dielectric Profile
		epsz = 8.854e-12

		for j in range(self.wa, self.wb):
			for i in range(self.ha, self.hb):
				xdist = (hc - i)
				ydist = (wc - j)
				dist = np.sqrt(xdist ** 2 + ydist ** 2)
				if dist <= radius:
					# lossy material
					self.gaz[i, j] = 1 / (epsr + (sigma * dt / epsz))
					self.gbz[i, j] = (sigma * dt / epsz)

		self.boundary_low = [0, 0]
		self.boundary_high = [0, 0]

		# Calculate the PML parameters
		self.gi2 = np.ones(self.height)
		self.gi3 = np.ones(self.height)
		self.fi1 = np.zeros(self.height)
		self.fi2 = np.ones(self.height)
		self.fi3 = np.ones(self.height)
		self.gj2 = np.ones(self.width)
		self.gj3 = np.ones(self.width)
		self.fj1 = np.zeros(self.width)
		self.fj2 = np.ones(self.width)
		self.fj3 = np.ones(self.width)
		# Create the PML as described in Section 3.2
		npml = 8

		for n in range(npml):
			xnum = npml - n
			xd = npml
			xxn = xnum / xd
			xn = 0.33 * xxn ** 3
			self.gi2[n] = 1 / (1 + xn)
			self.gi2[self.height - 1 - n] = 1 / (1 + xn)
			self.gi3[n] = (1 - xn) / (1 + xn)
			self.gi3[self.height - 1 - n] = (1 - xn) / (1 + xn)
			self.gj2[n] = 1 / (1 + xn)
			self.gj2[self.width - 1 - n] = 1 / (1 + xn)
			self.gj3[n] = (1 - xn) / (1 + xn)
			self.gj3[self.width - 1 - n] = (1 - xn) / (1 + xn)
			xxn = (xnum - 0.5) / xd
			xn = 0.33 * xxn ** 3
			self.fi1[n] = xn
			self.fi1[self.height - 2 - n] = xn
			self.fi2[n] = 1 / (1 + xn)
			self.fi2[self.height - 2 - n] = 1 / (1 + xn)
			self.fi3[n] = (1 - xn) / (1 + xn)
			self.fi3[self.height - 2 - n] = (1 - xn) / (1 + xn)
			self.fj1[n] = xn
			self.fj1[self.width - 2 - n] = xn
			self.fj2[n] = 1 / (1 + xn)
			self.fj2[self.width - 2 - n] = 1 / (1 + xn)
			self.fj3[n] = (1 - xn) / (1 + xn)
			self.fj3[self.width - 2 - n] = (1 - xn) / (1 + xn)

	def update(self, time_step):

		# Pulse Parameters
		t0 = 20
		spread = 8

		# Incident Ez values
		for j in range(1, self.width):
			self.ez_inc[j] = self.ez_inc[j] + 0.5 * (self.hx_inc[j - 1] - self.hx_inc[j])

		# Absorbing Boundary Conditions
		self.ez_inc[0] = self.boundary_low.pop(0)
		self.boundary_low.append(self.ez_inc[1])
		self.ez_inc[self.width - 1] = self.boundary_high.pop(0)
		self.boundary_high.append(self.ez_inc[self.width - 2])

		# Calculate the Dz field
		for j in range(1, self.width):
			for i in range(1, self.height):
				self.dz[i, j] = self.gi3[i] * self.gj3[j] * self.dz[i, j] + self.gi2[i] * self.gj2[j] * 0.5 * (
						self.hy[i, j] - self.hy[i - 1, j] - self.hx[i, j] + self.hx[i, j - 1])

		# Source
		pulse = np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)
		self.ez_inc[3] = pulse
		# Incident Dz values
		for i in range(self.ha, self.hb + 1):
			self.dz[i, self.wa] = self.dz[i, self.wa] + 0.5 * self.hx_inc[self.wa - 1]
			self.dz[i, self.wb] = self.dz[i, self.wb] - 0.5 * self.hx_inc[self.wb]

		# Calculate the Ez field
		for j in range(0, self.width):
			for i in range(0, self.height):
				self.ez[i, j] = self.gaz[i, j] * (self.dz[i, j] - self.iz[i, j])
				self.iz[i, j] = self.iz[i, j] + self.gbz[i, j] * self.ez[i, j]

		# Calculate the Incident Hx
		for j in range(0, self.width - 1):
			self.hx_inc[j] = self.hx_inc[j] + 0.5 * (self.ez_inc[j] - self.ez_inc[j + 1])

		# Calculate the Hx field
		for j in range(0, self.width - 1):
			for i in range(0, self.height - 1):
				curl_e = self.ez[i, j] - self.ez[i, j + 1]
				self.ihx[i, j] = self.ihx[i, j] + curl_e
				self.hx[i, j] = self.fj3[j] * self.hx[i, j] + self.fj2[j] * (
						0.5 * curl_e + self.fi1[i] * self.ihx[i, j])
		# Incident Hx values
		for i in range(self.ha, self.hb + 1):
			self.hx[i, self.wa - 1] = self.hx[i, self.wa - 1] + 0.5 * self.ez_inc[self.wa]
			self.hx[i, self.wb] = self.hx[i, self.wb] - 0.5 * self.ez_inc[self.wb]
		# Calculate the Hy field
		for j in range(0, self.width):
			for i in range(0, self.height - 1):
				curl_e = self.ez[i, j] - self.ez[i + 1, j]
				self.ihy[i, j] = self.ihy[i, j] + curl_e
				self.hy[i, j] = self.fi3[i] * self.hy[i, j] - self.fi2[i] * (
						0.5 * curl_e + self.fj1[j] * self.ihy[i, j])

		# Incident Hy values
		for j in range(self.wa, self.wb + 1):
			self.hy[self.ha - 1, j] = self.hy[self.ha - 1, j] - 0.5 * self.ez_inc[j]
			self.hy[self.hb, j] = self.hy[self.hb, j] + 0.5 * self.ez_inc[j]

		self.train_data = [
			np.abs(self.ez[np.ix_(self.total_field_range, self.total_field_range)]),
			#np.abs(self.ez[np.ix_(self.total_field_range, self.total_field_range)]),
			#self.dz[np.ix_(self.total_field_range, self.total_field_range)],
		   	#self.hx[np.ix_(self.total_field_range, self.total_field_range)],
		   	#self.hy[np.ix_(self.total_field_range, self.total_field_range)],
			self.gaz[np.ix_(self.total_field_range, self.total_field_range)],
		   	self.gbz[np.ix_(self.total_field_range, self.total_field_range)],
		   	#np.outer(self.ez_inc[self.total_field_range], self.ez_inc[self.total_field_range]),
		   	#np.outer(self.hx_inc[self.total_field_range], self.hx_inc[self.total_field_range])
			]


'''
import matplotlib.pyplot as plt
fdtd = Grid(128)
fig2 = plt.figure(figsize=(8, 7))
ax = fig2.add_subplot(1, 1, 1)


for i in range(1000):
	fdtd.update(i)
	plt.gca().cla()
	plt.pcolor(fdtd.ez)
	plt.pause(0.1)
'''

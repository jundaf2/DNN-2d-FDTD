
import numpy as np


# Yee-Grid
class Grid(object):
	def __init__(self, dim):
		# c * delta_t <= delta_x / sqrt(2)   if  delta_x == delta_y
		self.cdtds = 1.0 / np.sqrt(2.0)
		self.vac_imp = 377.0  # vacuum impedance
		self.height = dim
		self.width = dim
		self.source_index = int(self.height / 2) + self.height * int(self.width / 2)
		self.source_jz = 0
		# field quantities are flattened in 1-d
		self.Ez = np.zeros((dim*dim), dtype=float)
		self.Hx = np.zeros((dim*dim), dtype=float)
		self.Hy = np.zeros((dim*dim), dtype=float)
		self.fHxH = np.zeros((dim*dim), dtype=float)
		self.fHxE = np.zeros((dim*dim), dtype=float)
		self.fHyH = np.zeros((dim*dim), dtype=float)
		self.fHyE = np.zeros((dim*dim), dtype=float)
		self.fEzH = np.zeros((dim*dim), dtype=float)
		self.fEzE = np.zeros((dim*dim), dtype=float)

	def init(self):
		for y in range(self.height):
			for x in range(self.width):
				i = y + self.height*x
				self.Ez[i] = 0.0
				self.Hx[i] = 0.0
				self.Hy[i] = 0.0
				self.fHxH[i] = 1.0
				# delta_t / (mu * delta_y) = epsilon * c**2 * delta_t / delta_y = epsilon * c / sqrt(2) = self.cdtds / eta
				self.fHxE[i] = self.cdtds/self.vac_imp
				self.fHyH[i] = 1.0
				# delta_t / (mu * delta_x) = epsilon * c**2 * delta_t / delta_x = epsilon * c / sqrt(2) = self.cdtds / eta
				self.fHyE[i] = self.cdtds/self.vac_imp
				# if conductivity (sigma == 0):
				# 1 / (beta_ij * delta_x) = delta_t / (epsilon * delta_x) = c * delta_t / (c * epsilon * delta_x)
				# = 1 / (sqrt(2) * c * epsilon) = self.cdtds * eta
				self.fEzH[i] = self.cdtds*self.vac_imp
				self.fEzE[i] = 1.0

	def update(self, time):
		"""
		update all field values by Yee's algorithm
		"""
		# i is the 2-d matrix entry of the 1-d field vector
		for x in range(self.width):
			for y in range(self.height-1):
				i = y + x*self.height
				# Update in Hx, Ez is differentiated in vertical(along height)
				self.Hx[i] = self.Hx[i]*self.fHxH[i] - (self.Ez[i + 1] - self.Ez[i])*self.fHxE[i]

		for x in range(self.width-1):
			for y in range(self.height):
				i = y + x*self.height
				# Update in Hy, Ez is differentiated in horizontal(along width)
				self.Hy[i] = self.Hy[i]*self.fHyH[i] + (self.Ez[i + self.height] - self.Ez[i])*self.fHyE[i]

		for x in range(1,self.width-1):
			for y in range(1,self.height-1):
				i = y + x*self.height
				# Update in Ez, Hy is differentiated in horizontal, Hx is differentiated in vertical
				self.Ez[i] = self.Ez[i]*self.fEzE[i] + (self.Hy[i] - self.Hy[i-self.height] - self.Hx[i] + self.Hx[i-1])*self.fEzH[i]

		lamb = 400.  # wavelength
		k_vector = 2 * np.pi / lamb  # wave number, no use because kz always equals to 0
		omega = k_vector  # angle frequency, equals to wave number if assuming the two scales are equal
		attenuation = True  # True => 1, False => 0
		z = 0  # cross profile (x-y plane) position
		if time == 10:
			time = 10
		else:
			time = 0
		self.source_jz = 4000. * np.sin(omega * time + k_vector * z) * np.exp(-0.2 * time * attenuation)

		# add an oscillating point source Jz(x, y) = Jz(i, j)
		self.Ez[self.source_index] += self.source_jz


class Colormap(object):
	def __init__(self):
		"""
		add a MATLAB Jet like colormap
		"""
		self.palette = np.zeros(255*3, dtype=int)
		self.colors = np.array([0, 0, 127, 0, 0, 255, 0, 127, 255, 0, 255, 255, 127, 255, 127, 255, 255, 0,
								255, 127, 0, 255, 0, 0, 127, 0, 0, 255, 255, 255])

	def init(self):
		for c in range(9):
			for i in range(25):
				weight_c0 = (25.0 - i)/25.0
				weight_c1 = i/25.0

				# interpolate between the base colors
				self.palette[c * 3 * 25 + 3 * i] = int((self.colors[c*3] * weight_c0 + self.colors[(c+1)*3] * weight_c1))
				self.palette[c * 3 * 25 + 3 * i + 1] = int((self.colors[c*3 + 1] * weight_c0 + self.colors[(c+1)*3+1] * weight_c1))
				self.palette[c * 3 * 25 + 3 * i + 2] = int((self.colors[c*3 + 2] * weight_c0 + self.colors[(c+1)*3 +2] * weight_c1))


class Simulation(object):
	def __init__(self, DIM = 32):
		self.time = 0.
		self.time_step = 10
		self.grid = Grid(DIM)
		self.colormap = Colormap()
		self.grid.init()
		self.colormap.init()
		self.dataset = []

	def data_gen(self, frame_num):
		del self.dataset[0: 1]  # remove data
		# print("Data_Generating ...")
		for i in range(frame_num):
			self.time += self.time_step
			self.grid.update(self.time)
			intensity = np.abs(self.grid.Ez.reshape([self.grid.width, self.grid.height]))
			self.dataset.append(intensity)







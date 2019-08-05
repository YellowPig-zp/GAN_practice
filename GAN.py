import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

class GAN():
	def __init__(self):
		self.img_rows = 28
		self.img_cols = 28
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)

		optimizer = optimizers.Adam(lr=0.0002, beta_1=0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

		z = layers.Input(shape=(100,))
		img = self.generator(z)

		self.discriminator.trainable = False

		possibility = self.discriminator(img)

		self.combined = models.Model(z, possibility)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

	def build_generator(self):
		noise_shape = (100,)
		model = models.Sequential()
		model.add(layers.Dense(256, input_shape=noise_shape))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(512))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(1024))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.BatchNormalization(momentum=0.8))
		model.add(layers.Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(layers.Reshape(self.img_shape))

		model.summary()

		noise = layers.Input(shape=noise_shape)
		img = model(noise)
		generator = models.Model(inputs=noise, outputs=img)

		return generator

	def build_discriminator(self):
		img_shape = self.img_shape

		model = models.Sequential()

		model.add(layers.Flatten(input_shape=img_shape))
		model.add(layers.Dense(512))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.Dense(256))
		model.add(layers.LeakyReLU(alpha=0.2))
		model.add(layers.Dense(1, activation='sigmoid'))

		img = layers.Input(shape=img_shape)
		possibility = model(img)
		discriminator = models.Model(inputs=img, outputs=possibility)

		return discriminator


	def train(self, epochs, batch_size=128, save_interval=50):
		X_train, y_train = loadlocal_mnist(images_path='./train-images-idx3-ubyte', labels_path='./train-labels-idx1-ubyte')
		X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		X_train = np.reshape(X_train, (X_train.shape[0],)+self.img_shape)

		half_batch = int(batch_size / 2)

		for epoch in range(epochs):
			idx = np.random.choice(X_train.shape[0], half_batch, replace=False)
			imgs = X_train[idx]
			noise = np.random.normal(0, 1, (half_batch, 100))
			gen_imgs = self.generator.predict(noise)

			# ---------------------
			#  Train Discriminator
			# ---------------------
			d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# ---------------------
			#  Train Generator
			# ---------------------
			noise = np.random.normal(0, 1, (batch_size, 100))

			valid_y = np.ones((batch_size, 1))

			g_loss = self.combined.train_on_batch(noise, valid_y)

			print ("epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

			if epoch % save_interval == 0:
				self.save_imgs(epoch)

	def save_imgs(self, epoch):
		r, c = 5, 5
		noise = np.random.normal(0, 1, (r * c, 100))
		gen_imgs = self.generator.predict(noise)

		# Rescale images 0 - 1
		gen_imgs = 0.5 * gen_imgs + 0.5

		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("./gan/images/mnist_%d.png" % epoch)
		plt.close()

if __name__ == '__main__':
	gan = GAN()
	gan.train(epochs=30000, batch_size=32, save_interval=200)



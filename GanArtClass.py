import math
import pathlib
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import PIL
import imageio
import tensorflow as tf
from tensorflow.keras import layers
from GANFunctions import wasserstein_generator_loss, wasserstein_discriminator_loss

class GanArtist:
    def __init__(self, img_height = 360, img_width = 360, noise_dim=100, BUFFER_SIZE=60000, BATCH_SIZE=7, datadir=r'/trainingphotos'):
        self.img_height = img_height
        self.img_width = img_width
        self.noise_dim = noise_dim
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.train_dataset = self.getDataAndCache(datadir=datadir)


    # Essential Functions for structure and training.
    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(self.noise_dim,)))

        model.add(layers.Dense(5 * 5 * 300, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.2))

        model.add(layers.Reshape((5, 5, 300)))
        assert model.output_shape == (None, 5, 5, 300)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(300, (5, 5), strides=(3, 3), padding='same', use_bias=False))
        assert model.output_shape == (None, 15, 15, 300)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.15))

        model.add(layers.Conv2DTranspose(150, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 30, 30, 150)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.15))
        model.add(layers.Conv2DTranspose(75, (5, 5), strides=(3, 3), padding='same', use_bias=False))
        assert model.output_shape == (None, 90, 90, 75)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.15))

        model.add(layers.Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 180, 180, 40)
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Dropout(0.15))
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.img_width, self.img_height, 3)

        return model
    def make_discriminator_model(self):
        model = tf.keras.Sequential()

        model.add(layers.InputLayer(input_shape=(self.img_height, self.img_width, 3)))
        model.add(layers.GaussianDropout(0.04))
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1)) # no activation. logits=true when training
        return model

    @tf.function
    def train_step(self, images):
        noise = tf.random.uniform(shape=[self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = wasserstein_generator_loss(fake_output)
            disc_loss = wasserstein_discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            for batch_of_images in dataset:
                self.train_step(batch_of_images)
            end = time.time()
            print('Time for epoch {} is {} sec'.format(epoch + 1, round(end - start, 3)))
    def getDataAndCache(self, datadir='/trainingphotos'):
        # Load Image Directory
        # data_dir = pathlib.Path(r'C:\Users\William\PycharmProjects\GAN PRODUCTION\GAN production' + str(datadir))
        data_dir = pathlib.Path(datadir)
        image_count = len(list(data_dir.glob('*/*.png'))) + len(list(data_dir.glob('*/*.jpg')))
        print("{} images in the set.".format(image_count))
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            seed=123,
            image_size=(self.img_height, self.img_width),
            labels=None,
            batch_size=image_count - image_count % self.BATCH_SIZE)

        image_batch = next(iter(train_ds))

        # Prepare data
        train_images = (image_batch - 127.5) / 127.5  # Normalize the images to [-1, 1]

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        print("With {} batches of {} images each.".format(len(train_dataset), self.BATCH_SIZE))
        # .cache keeps images in memory during training.
        train_dataset = train_dataset.cache()
        return train_dataset
    def SAVE(self, modeldir=r"C:\Users\William\PycharmProjects\GAN PRODUCTION\GAN production\MODELS", suffix = ""):
        # compile them before being saved
        self.generator.compile(optimizer=self.generator_optimizer)
        self.discriminator.compile(optimizer=self.discriminator_optimizer)
        # Save them
        self.generator.save("{}\generator{}.h5".format(modeldir, suffix))
        self.discriminator.save("{}\discriminator{}.h5".format(modeldir, suffix))
        print("Saved to {}. Epochs: {}".format(modeldir, suffix))

    def LOAD(self, modeldir=r"C:\Users\William\PycharmProjects\GAN PRODUCTION\GAN production\MODELS", suffix=""):
        self.generator = tf.keras.models.load_model("{}/generator{}.h5".format(modeldir, suffix))
        self.discriminator = tf.keras.models.load_model("{}/discriminator{}.h5".format(modeldir, suffix))
        # compile them before using
        self.generator.compile(optimizer=self.generator_optimizer)
        self.discriminator.compile(optimizer=self.discriminator_optimizer)

    # Generating graphics of various kinds <3
    def generate_image(self, filename="image"):

        generated_image = tf.cast(tf.math.round(self.generator(tf.random.uniform([1, self.noise_dim]), training=False) * 127.5 + 127.5), tf.int32)[0]

        plt.imshow(generated_image)
        plt.axis('off')
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save("{}{}.png".format(str(filename), round(time.time())))
        plt.close()
    def generate_gif(self, frames=400, filename="GIF"):
        # Make a GIF of the current generator outputting images with a moving perlin noise
        anim_file = "{}{}.gif".format(filename, round(time.time()))
        perlin = tf.random.uniform(shape=[1, self.noise_dim], minval=0.4, maxval=0.85)
        with imageio.get_writer(anim_file, mode='I') as writer:
            for i in range(frames):
                # Make image
                array_image = np.array(tf.cast(tf.math.round(self.generator(perlin, training=False) * 127.5 + 127.5), tf.uint8)[0])

                # Put image in writer for gif
                writer.append_data(array_image)
                # Change 1 bit, half at a time(so loop gif is 4times the noise worth of frames

                array = np.array(perlin)
                index = i % self.noise_dim
                array[0][index] = (random.uniform(0.1,0.4) if array[0][index] > 0.5 else random.uniform(0.6,0.9))
                perlin = tf.cast(array, tf.float32)
                if i % (frames / 5) == 0: print("{} % complete".format(i * 100 // frames))
            print("Finished. File called: {}".format(anim_file))

    def production(self, preschool=2, rate=2, produce=1, time_allowed=60, gif=False):
        start = time.time()
        preschool = int(preschool)
        rate = int(rate)
        produce = int(produce)
        time_allowed = int(time_allowed)
        # early training period to get it up to scratch
        print("Starting preschool for {} epochs".format(preschool))
        self.train(self.train_dataset, preschool)

        # Production period, pump out the gifs and photos
        for i in range(produce):
            if time.time() - start < time_allowed:
                print("Production lot {} starting. Training for {} epochs.".format(i + 1, rate))
                self.train(self.train_dataset, rate)
                self.generate_image()
                self.generate_image()
                if gif: self.generate_gif()

                print("Production lot {} finished.".format(i + 1))
            else:
                print("Finished due to time constraints.")
                break

    def evolution_gif(self, epochs=[1], frames=[1]):
        if type(epochs) == type(1): epochs = [epochs]
        if type(frames) == type(1): frames = [frames]
        anim_file = "learning{}.gif".format(round(time.time()))
        seed = tf.random.uniform([1, self.noise_dim])

        with imageio.get_writer(anim_file, mode='I') as writer:
            for count, epoch in enumerate(epochs):
                #frame_num = frames[count - 1]

                for i in range(frames[count - 1]):
                    print("Frame {}.".format(i + 1))
                    # Make image, turn into array for the writer
                    array_image = np.array(
                        tf.cast(tf.math.round(self.generator(seed, training=False) * 127.5 + 127.5), tf.uint8)[0])
                    # Put image in writer for gif
                    writer.append_data(array_image)
                    # Train the models a little
                    self.train(self.train_dataset, epoch)
            print("Finished. File called: {}".format(anim_file))

    def GIFS(self, preschool=10000):
        # sections = max(1, int(math.log10(preschool)))
        # frames = [50] * sections
        # weights = [1 / i ** 3 for i in range(1, 1 + len(frames))]
        # SUM = sum(weights)
        # weights = [i / SUM for i in weights]
        # epochs = [max(1, preschool * weights[i] // frames[i]) for i in range(len(frames))]
        # correction_factor = sum([a * b for a, b in zip(epochs, frames)]) / preschool
        # epochs = [max(1, int(i / correction_factor))for i in epochs]
        # print("Doing {} epochs of {} frames for a total of {} training epochs".format(epochs, frames, sum([a * b for a, b in zip(epochs, frames)])))
        # Initial Training w/ GIF
        self.evolution_gif(epochs=[1], frames=[100])
        self.generate_image()
        self.generate_gif()
        # Change dataset and then new evolution gif
        self.train_dataset = self.getDataAndCache(datadir='/flowers1')
        self.evolution_gif(epochs=[1], frames=[100])
        self.generate_image()
        self.generate_gif()
        # Change dataset and then new evolution gif
        self.train_dataset = self.getDataAndCache(datadir='/flowers2')
        self.evolution_gif(epochs=[2], frames=[100])
        self.generate_image()
        self.generate_gif()
        # Change dataset and then new evolution gif
        self.train_dataset = self.getDataAndCache(datadir='/flowers3')
        self.evolution_gif(epochs=[2], frames=[100])
        self.generate_image()
        self.generate_gif()
        # Change dataset and then new evolution gif
        self.train_dataset = self.getDataAndCache(datadir='/flowers1')
        self.evolution_gif(epochs=[1], frames=[100])
        self.generate_image()
        self.generate_gif()
    def shimmers(self):
        # Make a GIF of the current generator outputting images with a moving perlin noise
        filename = "shimmer"
        frames = 20 # 2 second gif that perfectly loops
        anim_file = "{}{}.gif".format(filename, round(time.time()))
        perlin = tf.random.uniform(shape=[1, self.noise_dim], minval=0.05, maxval=0.95)
        #print(perlin)
        bitstochange = random.sample(list(range(self.noise_dim)), k=frames//2)

        with imageio.get_writer(anim_file, mode='I') as writer:
            for i in bitstochange:
                # Make image
                array_image = np.array(tf.cast(tf.math.round(self.generator(perlin, training=False) * 127.5 + 127.5), tf.uint8)[0])
                writer.append_data(array_image)
                array = np.array(perlin)
                index = i
                array[0][index] = array[0][index]/1.3
                perlin = tf.cast(array, tf.float32)
                #print(perlin)

            writer.append_data(np.array(tf.cast(tf.math.round(self.generator(perlin, training=False) * 127.5 + 127.5), tf.uint8)[0]))
            for i in bitstochange:
                # Make image
                array_image = np.array(tf.cast(tf.math.round(self.generator(perlin, training=False) * 127.5 + 127.5), tf.uint8)[0])
                writer.append_data(array_image)
                array = np.array(perlin)
                index = i
                array[0][index] = array[0][index]*1.3
                perlin = tf.cast(array, tf.float32)
            writer.append_data(np.array(tf.cast(tf.math.round(self.generator(perlin, training=False) * 127.5 + 127.5), tf.uint8)[0]))

            print("Finished. File called: {}".format(anim_file))

from GanArtClass import GanArtist
from tensorflow.python.client import device_lib

# Show devices, check they exist! Don't run this with only a cpu!
device_lib.list_local_devices()

# Specify where the photos are. You will need to put them inside a folder, inside the directory you specify.
# Yes, a folder within a folder.
datadir = r'C:\...\ThisFolder\trainingphotos'

# If you want to save or load models, you will need to specify where to save/load them to/from.
modeldir = r"C:\...\ThisFolder\MODELS"


# Create the GAN model
artist = GanArtist(img_height = 360,                # Photo height
                   img_width  = 360,                # Photo width
                   noise_dim  = 100,                # Noise dimensions to use in generator
                   BATCH_SIZE = 7,                  # Number of images to train models in each train step
                   datadir    = datadir) # Where to get photos from?

# Print summaries of the generator and discriminator models.
# Here you can see each layer, convolutions. Look at the total params, thats why you need a GPU. :)
print(artist.generator.summary())
print(artist.discriminator.summary())


# Train the 'artist'. Using the train_dataset created upon model initialisation.
# Second variable is the number of epochs (training steps) to do.
# You will need at least 100 epochs to get anything interesting, likely many, many more.
artist.train(artist.train_dataset, 10)

# Generate a GIF of random noise
# epochs are the number of epochs to train between each frame.
# frames is the number of frames to put in the gif.
# So total number of epochs trained is epochs*frames
artist.evolution_gif(epochs=1, frames = 20)


# Generate a random image from the artists current learning state.
artist.generate_image()

# Generate a GIF using moving random noise as input.
artist.generate_gif()

# Generate a gif which picks one noise input, and each frame slightly changes the noise to get a slowly moving shimmering effect.
# much more subtle than the generator_gif() function.
artist.shimmers()


# Functions to SAVE, or LOD, a model. Make sure its in the model directory specified above.
# If you want to specify a specific model in the directory, or give a specific name, use the suffix argument.
# e.g. if suffix="1", then the models will be called:
# generator1.h5
# discriminator1.h5
# and so when you load them you will need to specify suffix="1" also.
# artist.SAVE(modeldir=modeldir, suffix="")
# artist.LOAD(modeldir=modeldir, suffix="")

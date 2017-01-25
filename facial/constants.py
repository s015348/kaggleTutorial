# Set Dev mode for using small samples
DEV_MODE = True
DEV_TRAIN_LEN = 20
DEV_TEST_LEN = 10

# Constants
# Train data and test data are both in size of 96*96 pixels
IMAGE_SIZE = 96
# Gray color scope
GRAY_255 = 255
# Patch is set in size of 21*21
PATCH_SIZE = 10
# Number of features in pair
FEATURE_NUM = 15
# When calculating correlation of image and patch,
# shift 2 pixels in each step
SEARCH_STEP = 2
# Used in calculating errors by selecting 200 train images
EVALUATION_LEN = 200
# Output timer counting per 100 images
LOG_INTERVAL = 100
# For saving neural net models
SIMPLE_3LAYERS_FILENAME = 'simple_3-layers_model'
LENET5_CNN_FILENAME = 'LeNet5_CNN_model'
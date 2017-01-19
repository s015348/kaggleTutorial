# Set Dev mode for using small samples
DEV_MODE = True

# Constants
# Train data and test data are both in size of 96*96 pixels
IMAGE_SIZE = 96
# Patch is set in size of 21*21
PATCH_SIZE = 10
# Number of features in pair
FEATURE_NUM = 15
# When calculating correlation of image and patch,
# shift 2 pixels in each step
SEARCH_STEP = 2
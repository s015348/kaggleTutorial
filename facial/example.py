import time, datetime
import matplotlib.pyplot as plt
from constants import DEV_MODE
from use_patch import use_image_patches
from utility import clean_data, data_preview, load_data, save_result
from visualization import plot_images


# Count running time
starttime = datetime.datetime.now() 

# Load and preview data
train, test, label = load_data(dev_mode = DEV_MODE)
data_preview(train, test, label)
plot_images(train, label)

# Fill NA with average.
# TODO: remove abnormal samples
label = clean_data(label)

# Train and predict
submission, score = use_image_patches(train, test, label)
plt.show()

save_result(submission)

# Count running time
endtime = datetime.datetime.now()
print("Image patch correlation score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))
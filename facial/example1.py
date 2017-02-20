import time, datetime
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from constants import DEV_MODE
from use_patch import use_image_patches
from utility import clean_data, data_preview, get_label_args, load_data, save_result
from visualization import plot_images, plot_label_distribution


# Count running time
starttime = datetime.datetime.now() 

# Load and preview data
train, test, label = load_data(dev_mode = DEV_MODE)
data_preview(train, test, label)
plot_images(train, label)

# Fill NA with average.
# TODO: remove abnormal samples
label = clean_data(label)
label_min, label_max, label_median, label_var = get_label_args(label)
plot_label_distribution(label)

# Train and predict
submission, score = use_image_patches(train, test, label, label_median, label_var)

save_result(submission)

# Count running time
endtime = datetime.datetime.now()
print("Image patch correlation score: %f, spend %d seconds" %(score, (endtime - starttime).seconds))

plt.show()
# Import necessary libraries
import os            # Operating system functionality
import glob          # File system path manipulation
import numpy as np   # Numerical operations
import ND2Reader     # Not used in this script
import napari        # Interactive multi-dimensional image viewer
import bigfish       # Library for single-molecule fluorescence imaging
import bigfish.stack as stack        # Functions for processing image stacks
import bigfish.detection as detection  # Spot detection algorithms
import bigfish.multistack as multistack  # Functions for multi-channel stacks
import bigfish.plot as plot          # Plotting utilities
import time          # Time tracking
import pandas as pd  # Data manipulation and analysis
import tifffile as tiff  # Reading and writing TIFF files
from skimage import io   # Image processing functions
import plotly.express as px  # Expressive visualization library
from nis2pyr.convertor import convert_nd2_to_pyramidal_ome_tiff

dir1 = "./" ##Input Directory with the ND2 files
dir2 = os.path.join(dir1,'TiffFiles') #New subdirectory where processed files will be saved
dir3 = os.path.join(dir2, 'Max_Projections')  # Directory for max projections
dir4 = os.path.join(dir3, 'FOVs')  # Directory for individual FOVs
dir6 = os.path.join(dir3, "Combined_Files") # Create a directory to save combined files if it doesn't exist
dir7 = os.path.join(dir3, 'Spots') #Save output spot files

# Directories to check and create if they don't exist
dirs_to_create = [dir2, dir3, dir4, dir6]

# Loop through each directory
for directory in dirs_to_create:
    # Check if the directory does not exist
    if not os.path.isdir(directory):
        # If it doesn't, create the directory
        os.mkdir(directory)

# Define parameters
voxelval = 110.3752759382
##radiusval = 250.0
radiusval = 2*voxelval



# Filter ND2 files and remove those used for testing bleaching
files = [f for f in os.listdir(dir1) if f.endswith('.nd2') and 'Bleach' not in f]

# Sort the files
files.sort()

# Iterate over each file in the 'files' list (all ND2 files)
for fil in files:
    # Print the directory and filename being processed
    print(os.path.join(dir1, fil))
    
    # Convert the ND2 file to pyramidal OME-TIFF format
    # Specify the input ND2 file path, output OME-TIFF file path,
    # and the maximum number of pyramid levels (set to 1 in this case)
    convert_nd2_to_pyramidal_ome_tiff(os.path.join(dir1, fil), 
                                    os.path.join(dir2, fil.split(".")[0] + '.tif'),
                                    max_levels=1)
# List all TIFF files in the input directory and sort them
files = sorted([f for f in os.listdir(dir2) if f.endswith('.tif')])

# Define a function to compute maximum intensity projection along the Z-axis
def maximum_intensity_projection(image_stack):
    return np.max(image_stack, axis=0)

# Loop through each TIFF file
for fil in files:
    # Read the OME-TIFF file
    ome_tiff_path = os.path.join(dir2, fil)
    image_stack = tiff.imread(ome_tiff_path)
    
    # Compute the maximum intensity projection
    mip = maximum_intensity_projection(image_stack)
    
    # Save the max projection image
    mip_path = os.path.join(dir3, "MAX_" + os.path.splitext(fil)[0] + ".tif")
    tiff.imsave(mip_path, mip)
    
    # Extract FOV information from the filename
    FOV = os.path.splitext(fil)[0].split("_")
    
    # Create a directory for the FOV if it does not exist
    dir5 = os.path.join(dir4, FOV[1])
    os.makedirs(dir5, exist_ok=True)
    
    # Save each channel of the max projection image as a separate TIFF file
    for chan, channel_image in enumerate(mip, start=1):
        savename = os.path.join(dir5, f"{FOV[0]}_channel_{chan}.tif")
        tiff.imsave(savename, channel_image)

# Get a list of directories in dir3
directories = [name for name in os.listdir(dir3) if os.path.isdir(os.path.join(dir3, name))]
directories.sort()  # Sort the list of directories

# Iterate over each directory
for directory in directories:
    # Create full path to the current directory
    dir_path = os.path.join(dir3, directory)
    
    # Get list of TIFF files in the directory and sort them
    tiff_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.tif')])
    
    # Extract channel information from TIFF filenames
    channels = [f.split('_')[2] for f in tiff_files]
    unique_channels = np.unique(channels)
    
    # Iterate over each unique channel
    for channel in unique_channels:
        # Get indices of TIFF files corresponding to the current channel
        indices = [i for i, c in enumerate(channels) if c == channel]
        
        # Initialize array for multichannel image
        multichannel_image = np.zeros((len(indices), image.shape[0], image.shape[1]))
        
        # Iterate over each TIFF file corresponding to the current channel
        for i, index in enumerate(indices):
            # Read TIFF file
            img = tiff.imread(os.path.join(dir_path, tiff_files[index]))
            # Store image in multichannel array
            multichannel_image[i, :, :] = img
        
        # Define filename for combined multichannel image
        savename = os.path.join(dir6, f"{directory}_channel_{channel}")
        # Save combined multichannel image as TIFF
        tiff.imsave(savename, multichannel_image)

# Record start time
start_time = time.time()

# Threshold values for each channel
thresholds = [18, 18, 18]  # Thresholds for Channels 1, 2, and 3

# Number of cycles for each channel
channel_rounds = [8, 8, 8]

# Number of FOVs to use to detect thresholds
nfovs = 15

# Get list of TIFF files in the combined files directory and sort them
files = sorted([f for f in os.listdir(dir6) if f.endswith('.tif')])

# Compute spot radius in pixels
spot_radius_px = detection.get_object_radius_pixel(
                    voxel_size_nm=(voxelval, voxelval), 
                    object_radius_nm=(radiusval, radiusval), 
                    ndim=2)

# Iterate over each channel
for channel, threshold, rounds in zip(range(3), thresholds, channel_rounds):
    e = 0
    for fil in range(nfovs):
        print(f"--- Start {time.time() - start_time} seconds ---")
        filename = files[fil * len(channels) + channel]
        img = tiff.imread(os.path.join(dir6, filename))
        print(filename)
        # Detect spots
        for t in range(img.shape[0]):
            rna = img[t, :, :]
            # LoG filter
            rna_log = stack.log_filter(rna, sigma=spot_radius_px)
            # Local maximum detection
            mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)
            # Thresholding
            threshold_value = detection.automated_threshold_setting(rna_log, mask)
            if channel == 0:
                ths1[e] = threshold_value
            elif channel == 1:
                ths2[e] = threshold_value
            else:
                ths3[e] = threshold_value
            e += 1
    print(f"Finished thresholding for channel {channel + 1} after {time.time() - start_time} seconds ---")

# Record start time
start_time = time.time()

# Calculate thresholds for each channel
threshs = [2 * np.median(ths) + 40 for ths in [ths1, ths2, ths3]]

# Define directory paths and get list of TIFF files
files = sorted([each for each in os.listdir(dir6) if each.endswith('.tif')])
chan = pd.unique(pd.DataFrame(np.stack(np.char.split(files, sep="_"), axis=0))[2])
nfovs = len(files) // len(chan)

# Iterate over each FOV
for fil in range(nfovs):
    print("--- Start %s seconds ---" % (time.time() - start_time))
    for cc, chan in enumerate(chans[:-1]):
        print("Analysing Channel %s" % cc)
        filename = files[fil * len(chans) + cc]
        img = tiff.imread(os.path.join(dir6, filename))
        base_name = filename.split(".")[0]
        savenamespots = base_name + ".csv"
        savenamespotscl = base_name + "_spotclusters.csv"
        savenameclusters = base_name + "_clusters.csv"
        sp_list, spcl_list, cl_list = [], [], []
        # Detect spots and clusters for each round
        for t in range(img.shape[0] - 1):
            print("Analysing Round %s" % (t + 1))
            rna = img[t + 1, :, :]
            spots = detection.detect_spots(
                images=rna,
                return_threshold=False,
                threshold=threshs[cc],
                voxel_size=(voxelval, voxelval),
                spot_radius=(radiusval, radiusval)
            )
            spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
                image=np.uint16(rna),
                spots=spots,
                voxel_size=(voxelval, voxelval),
                spot_radius=(radiusval, radiusval),
                alpha=0.75,
                beta=0.9,
                gamma=15
            )
            spots_post_clustering, clusters = detection.detect_clusters(
                spots=spots_post_decomposition,
                voxel_size=(int(voxelval), int(voxelval)),
                radius=int(radiusval),
                nb_min_spots=4
            )
            sp_list.append(spots)
            spcl_list.append(spots_post_clustering)
            cl_list.append(clusters)
        sp = pd.DataFrame(np.vstack(sp_list), columns=['Y', 'X', 'round'])
        spcl = pd.DataFrame(np.vstack(spcl_list), columns=['Y', 'X', 'clusterindex', 'round'])
        cl = pd.DataFrame(np.vstack(cl_list), columns=['Y', 'X', 'nspots', 'index', 'round'])
        sp.to_csv(os.path.join(dir6, savenamespots), index=False)
        spcl.to_csv(os.path.join(dir6, savenamespotscl), index=False)
        cl.to_csv(os.path.join(dir6, savenameclusters), index=False)

print("Finished thresholding for channel 1 image %s after %s seconds ---" % (e, time.time() - start_time))

# Define a function to find the nearest square number below a given limit
def nearest_square(limit):
    return int(limit ** 0.5)

# Get list of TIFF files
files = sorted([each for each in os.listdir(dir6) if each.endswith('.tif')])
chan = pd.unique(pd.DataFrame(np.stack(np.char.split(files, sep="_"), axis=0))[2])
nfovs = int(len(files) / len(chan))

# Initialize empty lists to store spot data for each channel
spotscy7, spotscy5, spotscy3 = [], [], []

# Read all spot files for each channel before entering the loop
for fil in range(nfovs):
    for cc in range(len(chans)):
        filename = files[fil * len(chans) + cc]
        cy_spots = pd.read_csv(os.path.join(dir7, filename.split(".")[0] + "_spotclustters.csv"))
        cy_spots['Y'] += fil * gap
        cy_spots['X'] += fil * gap
        if cc == 0:
            spotscy7.append(cy_spots)
        elif cc == 1:
            spotscy5.append(cy_spots)
        elif cc == 2:
            spotscy3.append(cy_spots)

# Concatenate spot data for each channel
spotscy7 = pd.concat(spotscy7)
spotscy5 = pd.concat(spotscy5)
spotscy3 = pd.concat(spotscy3)

# Initialize Napari viewer
viewer = napari.Viewer()

# Add image to Napari viewer
imagelayer = viewer.add_image(np.zeros((gap * totalarea, gap * totalarea), dtype=np.int16))
imagelayer.contrast_limits = (0, 65000)

# Iterate over each gene and add points to the viewer for each channel
for round in range(len(genescy7)):
    cy7 = spotscy7[spotscy7['round'] == round]
    cy5 = spotscy5[spotscy5['round'] == round]
    cy3 = spotscy3[spotscy3['round'] == round]

    viewer.add_points(np.array(cy7)[:, :2], 
                      face_color=color[(len(chans) - 1) * round],
                      size=5,
                      blending='translucent_no_depth', 
                      edge_width=0, 
                      name=genescy7[round])

    viewer.add_points(np.array(cy5)[:, :2], 
                      face_color=color[(len(chans) - 1) * round + 1], 
                      size=5,
                      blending='translucent_no_depth', 
                      edge_width=0, 
                      name=genescy5[round])

    viewer.add_points(np.array(cy3)[:, :2], 
                      face_color=color[(len(chans) - 1) * round + 2], 
                      size=5,
                      blending='translucent_no_depth', 
                      edge_width=0, 
                      name=genescy3[round])

# Concatenate spot data for all channels
allspots = pd.concat([spotscy7, spotscy5, spotscy3])

# Save spot data to CSV files
allspots.to_csv(os.path.join(dir7, "allspots.csv"), index=False)
featuresall = allspots['gene']
featuresall.to_csv(os.path.join(dir7, 'features.csv'), index=False)

# Add points for all spots to the viewer
feat = tuple(np.array(featuresall))
pointlayer = viewer.add_points(np.array(allspots)[:, :2], face_color='white', size=5,
                                blending='translucent_no_depth', edge_width=0, name='All_spots', opacity=0,
                                features=feat)

pointlayer.refresh()

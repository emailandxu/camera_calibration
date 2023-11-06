#!/usr/bin/python
#! -*- encoding: utf-8 -*-

# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python implementation of the bash script written by Romuald Perrot
# Created by @vins31
# Modified by Pierre Moulon
#
# this script is for easy use of OpenMVG
#
# usage : python openmvg.py image_dir output_dir
#
# image_dir is the input directory where images are located
# output_dir is where the project must be saved
#
# if output_dir is not present script will create it
# ffmpeg -i VID_20231023_123406_00_034.mp4 -vf "fps=1" input/%04d.jpg
# docker run --rm -it -v $PWD:$PWD openmvg python3 $PWD/my_sfm_sphere.py $PWD/db/avenue/input $PWD/db/avenue/output

# Indicate the openMVG binary directory
OPENMVG_SFM_BIN = "/opt/openMVG_Build/Linux-x86_64-RELEASE"

# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/opt/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

import os
import subprocess
import sys

if len(sys.argv) < 3:
    print ("Usage %s image_dir output_dir" % sys.argv[0])
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_global")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print ("Using input dir  : ", input_dir)
print ("      output_dir : ", output_dir)

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)

print ("1. Intrinsics analysis")
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-c", "7", "-f", "1"] )
pIntrisics.wait()

print ("2. Compute features")
pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT", "-p", "HIGH"] )
pFeatures.wait()

print ("3. Compute matching pairs")
pPairs = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"), "-i", matches_dir+"/sfm_data.json", "-o" , matches_dir + "/pairs.bin" ] )
pPairs.wait()

print ("4. Compute matches")
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-p", matches_dir+ "/pairs.bin", "-o", matches_dir + "/matches.putative.bin" ] )
pMatches.wait()

print ("5. Filter matches" )
pFiltering = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir+"/sfm_data.json", "-m", matches_dir+"/matches.putative.bin" , "-g" , "a" , "-o" , matches_dir+"/matches.e.bin" ] )
pFiltering.wait()

# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)

print ("6. Do Global reconstruction")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", "INCREMENTAL", "--input_file", matches_dir+"/sfm_data.json", "--match_file", matches_dir+"/matches.e.bin", "--output_dir", reconstruction_dir] )
pRecons.wait()

print ("7. Colorize Structure")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
pRecons.wait()

print ("7.1 to json")
cmd =  [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ConvertSfM_DataFormat"), 
                             "binary",  "-i", reconstruction_dir+"/sfm_data.bin", "-o", reconstruction_dir+"/sfm_data.json",
                            "-V", "-I", "-E" ]
pRecons = subprocess.Popen(cmd)
pRecons.wait()

print ("8. To Cubic")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVGSpherical2Cubic"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(output_dir,"cubic")] )
pRecons.wait()

print("9. To Colamp")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_openMVG2Colmap"),  "-i", output_dir+"/cubic/sfm_data_perspective.bin", "-o", os.path.join(output_dir,"cubic/colmap_sparse")] )
pRecons.wait()

print("10. reorgnize")
from pathlib import Path
out = Path(output_dir)
images_dir = out.joinpath("images")
sparse_dir = out.joinpath("sparse/0")

images_dir.mkdir(exist_ok=True)
sparse_dir.mkdir(parents=True, exist_ok=True)

for png in out.joinpath("cubic").glob("*.png"):
    png.rename(images_dir / png.name)
  
for col in out.joinpath("cubic/colmap_sparse").glob("*"):
    col.rename(sparse_dir / col.name)


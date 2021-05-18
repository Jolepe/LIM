#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# v9: We use the RGB images from Flir and Principal Camera
#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Read paths from txt file
with open ("C:/SetupPython.txt", "r") as myfile:
    data=myfile.readlines()
# Assign variables from the txt file.
for i in range(len(data)):
	val = data[i].split("\"")[1]
	var_str = data[i].split("=")[0]
	vars()[var_str] = val

#Global variables
numImages = 1
MeshFile = "texturedMesh_short.ply"

# Import packages required
import sys, os
import flirimageextractor
import open3d as o3
import copy
import numpy as np
import cv2
sys.path.append(os.path.join(GithubCodes, "0_Utils"))
import allUtils
import Utils3D as u3d
sys.path.append(os.path.join(GithubCodes, "3_thermal3dmodel"))
import Utils4AliceVision as ualice
import UtilsColorcard as ucolc
import runthermal3d as rt3d
import segmentation3d as s3d
import additional_objects as addobj
import metrics_thermal3d as mt3d
import skin_segmentation as skinseg


def SilentMkdir(theDir):
	try:
		os.mkdir(theDir)
	except:
		pass
	return 0
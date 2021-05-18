# Segment the IR image in 2:
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# v9: We use the RGB images from Flir and Principal Camera
#
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Read paths from txt file

print("Process:")
a=1#int(input("Personal (0), Job(1):"))
dir_data=["C:/Users/julio/OneDrive/Escritorio/STANDUP WP2 - LIM/AcquisitionReconstruction/CHROrleans/",
             "C:/Users/julimend/OneDrive/Escritorio/STANDUP WP2 - LIM/AcquisitionReconstruction/CHROrleans/"]

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
from os import path
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage.segmentation import mark_boundaries
import Utils2D as u2d
import time
from skimage import io, feature

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Define paths & inputs
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
'''Julius'''
os.chdir(dir_data[a])
#os.chdir("G:/Mon Drive/8. PhD/3. Experimentos/1. 3Dreconstruction/AcquisitionReconstruction/CHROrleans/") #20200730_803598760
#os.chdir("C:/Users/EvelynG/Desktop/20201113/Outputs/") #Test2
#os.chdir("G:/Mon Drive/8. PhD/3. Experimentos/1. 3Dreconstruction/AcquisitionReconstruction/Pruebas/") #Test003

baseDir = "20200724_751386110"
thermaloutput_path = baseDir + "/13_Thermal3DModels/"
finaloutput_path = baseDir + "/15_Final3DModel/"
meshfilename = "texturedMesh_short.obj"
# PoseID: define
poseId="362574497"
# Marker Image:
poseIdMarker = "NoMarker"
#poseIdMarker = "IMG_20201113_172307" # Test2
#poseIdMarker = "IMG_20201123_105814" #Test003

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Load information from previous folders & prev files
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::
dtViews = ualice.loadCameraInit12(baseDir)
dtViews.to_csv(thermaloutput_path + '/dtViews.csv')
with open(baseDir + "/ScaleFactor.txt") as file:
    ScaleFactor = float(file.readline())
file.close()
model3D = o3.io.read_triangle_mesh(baseDir + "/11_texturing/" + meshfilename)

# flir image
flir_rgb_imagepath = str(np.array(dtViews.path)[dtViews.poseId == poseId][0])
flir_rgb_imagepath = flir_rgb_imagepath.replace("1_RGBImages\/FlirCamera", "2_IRImages")
flir_ir_temp_imagepath = flir_rgb_imagepath.replace(".jpg", ".npy")
#flir_ir_temp_imagepath = '20200804_803598760\\/00__Inputs\\/2_IRImages\\flir_20200804T122534.npy'
IRimageTempData = np.load(flir_ir_temp_imagepath)

aux = flir_rgb_imagepath.split("\\")
img_filename = aux[len(aux)-1]


# Camera position
dir_posematrix = baseDir + "/05_PrepareDenseScene/"
CameraMatrix_ir = np.array([[616.21, 0., 240],
                                [0., 616.21, 320],
                                [0., 0., 1.]])
(K, R, t) = ualice.getKRt(dir_posematrix, poseId)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Projectar marcador:
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_pos_markers(poseIdMarker,model3D):

    poseid_marker = str(np.array(dtViews.poseId)[dtViews.path.str.contains(poseIdMarker)][0])
    auxpath = str(np.array(dtViews.path)[dtViews.path.str.contains(poseIdMarker)][0])
    flir_markerseg_img = auxpath.replace("1_RGBImages/MainCamera", "6_MarkerSeg")
    flir_markerseg_img = flir_markerseg_img.replace(".jpg", ".png")
    mark_segmented = cv2.imread(flir_markerseg_img)

    (K, R, t) = ualice.getKRt(dir_posematrix, poseid_marker)
    (model3DFinal, distances, angles,
     values_allvertices,
     selected_vertices, values_selected_vertices) = u3d.Trans2DValTo3Dmodel(K, R, t, mark_segmented[:,:,0], model3D, ScaleFactor,
                                                    filter_distance=False, max_dist=60,
                                                    filter_angles=False, max_angle=90)
    model3DFinal.paint_uniform_color([0, 0, 0])
    np.asarray(model3DFinal.vertex_colors)[selected_vertices[values_selected_vertices==0], :] = [0, 0, 0]
    np.asarray(model3DFinal.vertex_colors)[selected_vertices[values_selected_vertices==255], :] = [0, 1, 1]
    #o3.io.write_triangle_mesh("XXX.ply",model3DFinal)

    selected_vertices_marker = selected_vertices[values_selected_vertices==255]

    return(selected_vertices_marker)

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# IR segmentation:  Clustering
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def IRsegmentation_clustering(IRimageTempData, plotit=True):
    aux = IRimageTempData.reshape((-1, 1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    vectorized = np.float32(aux)

    Kclus = 2
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, Kclus, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    res = center[label.flatten()]

    result_image_seg = res.reshape((IRimageTempData.shape))

    result_image_seg[result_image_seg >= result_image_seg.max()] = 255
    result_image_seg[result_image_seg <255 ] = 0

    if plotit==True:
        figure_size = 15
        plt.figure(figsize=(figure_size, figure_size))
        plt.subplot(1, 2, 1), plt.imshow(IRimageTempData)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(result_image_seg)
        plt.title('Segmented Image when K = %i' % Kclus), plt.xticks([]), plt.yticks([])
        plt.show()

    return(result_image_seg)


def from_3dmodel_to_2dimage(model3D, R, t, plotit=True):
    import open3d as o3d
    import copy

    projected = u3d.projection3Dto2D(CameraMatrix_ir, R, t, np.array(model3D.vertices))
    projected_img_model3d = np.zeros_like(result_image_seg)
    x = np.array(projected[:, 0]).astype('int')
    y = np.array(projected[:, 1]).astype('int')
    z = np.array(model3D.vertices)[:, 2]
    m0 = np.logical_and(x > 0, y > 0)
    m0 = np.logical_and(m0, x < projected_img_model3d.shape[1])
    m0 = np.logical_and(m0, y < projected_img_model3d.shape[0])

    x = x[m0]
    y = y[m0]
    #z = 255*z[m0.ravel()]/(z[m0.ravel()])
    #projected_img_model3d[y, x] =  z #255 for adding white instead of depth
    projected_img_model3d[y, x] = 255

    projected_img_model3d = cv2.blur(projected_img_model3d,(5,5))
    projected_img_model3d = cv2.GaussianBlur(projected_img_model3d, (5, 5), 0)
    projected_img_model3dv1 = copy.deepcopy(projected_img_model3d)
    projected_img_model3dv1[projected_img_model3dv1 > 0] = 255
    projected_img_model3dv1 = scipy.ndimage.median_filter(projected_img_model3dv1, 5)
    projected_img_model3dv1[projected_img_model3dv1 > 0] = 255

    holes = copy.deepcopy(projected_img_model3dv1)
    cv2.floodFill(holes, None, (0, 0), 255)
    holes = cv2.bitwise_not(holes)
    projected_img_model3dv2 = cv2.bitwise_or(projected_img_model3dv1, holes)
    projected_img_model3dv2[np.where(~np.isnan(projected_img_model3dv2))] = 0
    projected_img_model3dv2[np.where(np.isnan(projected_img_model3dv2))] = 255


#    projected_img_model3dv1 = projected_img_model3d
    if plotit == True:
        fig = plt.figure("Superpixels -- %d segments" % (2))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(projected_img_model3d, projected_img_model3dv2.astype("int"),[255,0,0]))
        plt.axis("off")

    return (projected_img_model3dv1)


def from_3dmodel_to_2dimage_nomarker(model3D, R, t, plotit=False):
    #Requires: selected_vertices_marker
    img_proj = from_3dmodel_to_2dimage(model3D, R, t, plotit=False)
    if poseIdMarker == "NoMarker":
        finalimg = img_proj
    else:
        model3d_reduced = model3D.select_by_index(selected_vertices_marker)
        img_marker = from_3dmodel_to_2dimage(model3d_reduced, R, t, plotit=False)
        finalimg = img_proj - img_marker

    return(finalimg)

def CalcIM_from_segmentation(x0=[0,0,0,0,0,0]):

    rot = x0[3:6] #R2=R
    t2 = x0[0:3]
    R2 = eul2rot(rot)

    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=False)

    mi = u2d.mutual_information_val(result_image_seg, projected_img_model3dv1)
    hx = u2d.mutual_information_val(result_image_seg, result_image_seg)
    hy = u2d.mutual_information_val(projected_img_model3dv1, projected_img_model3dv1)
    nmi = 2*mi/(hx+hy)
    # Maximiz mi  == Minimize -mi
    return(-nmi)

def from_matrix_to_angles(R):
    theta1 = np.arcsin(R[2,0])*(-1)
    #theta2 = np.pi-theta1
    psi1 = np.arctan2(R[2,1]/np.cos(theta1),R[2,2]/np.cos(theta1))
    phi1 = np.arctan2(R[1,0]/np.cos(theta1),R[1,1]/np.cos(theta1))
    return (psi1,theta1,phi1)


def IRimage_contours(IRimageTempData, plotit=True):#plotit=False

    from PIL import Image, ImageFilter
    # # Highlight difference
    #f = 2.5
    #imgIR2 = Image.fromarray((255 / 2) * (1 - np.cos(2 * np.pi * f * IRimageTempData / 255)))
    imgIR2 = Image.fromarray(IRimageTempData)
    # Contours
    #Jolepe: Check the result of this filter, change the kernel of the minfilter and erode
    out = imgIR2.filter(ImageFilter.MinFilter(size=5))
    out2 = np.array(Image.fromarray(np.array(imgIR2) - np.array(out)))
    out2[out2 > np.quantile(out2[out2 > 0], 0.8)] = 255


    # Erode
    #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5));#5,5
    #out3 = cv2.erode(np.array(out2), element)


    if plotit ==True:
        fig, axarr = plt.subplots(1, 3)
        axarr[0].imshow(IRimageTempData);
        axarr[1].set_title('0. Original Img')
        axarr[1].imshow(out2)
        axarr[1].set_title('1. Contours - v0')
        axarr[2].imshow(out2)#out3
        axarr[2].set_title('2. Eroded')

    return(out2)#out3

def model3d_contours(projected_img_model3dv1):

    from PIL import Image, ImageFilter
    a = Image.fromarray(projected_img_model3dv1)
    aa = a.filter(ImageFilter.MinFilter(size=3))
    ab = np.array(a) - np.array(aa)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(ab, kernel, iterations=1)

    return(dilation)

def create_multicontours(result_image_seg):#Julius: check possible errors, plot

    ret, bin_img = cv2.threshold(result_image_seg, result_image_seg.mean(), 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    edges = model3d_contours(bin_img)
    dil1 = cv2.dilate(edges, kernel, iterations=4)

    #Interior positive - Exterior negative
    import copy
    dil2 = copy.deepcopy(dil1)
    dil2[np.logical_and(dil1 == 0, bin_img == 0)] = 3 * (-1)
    dil2[np.logical_and(dil1 == 255, bin_img == 0)]  = 1.5*(-1)
    dil2[np.logical_and(dil1 == 0, bin_img == 255)] = 0.1
    dil2[np.logical_and(dil1 == 255, bin_img == 255)] = 0.3
    dil2[edges == 255] = 1

    return(dil2)


def mi_both_contours(img_contours,model3dproj_contours, plothere=True):

    mixed = np.stack((model3dproj_contours,)*3, axis=-1)
    mixed[:,:,1] = img_contours
    mixed[:,:,2] = np.zeros_like(model3dproj_contours)

    mi_this = u2d.mutual_information_val(np.array(model3dproj_contours), img_contours)
    if plothere == True :
        fig, ax = plt.subplots(1, 1)
        ax.imshow(mixed)
        ax.set_title('MutInfo:' + str(round(mi_this, 4)))
    return(mi_this)


def matching_both_contours(img_contours,model3dproj_contours, plothere=True):

    mixed = np.stack((model3dproj_contours,)*3, axis=-1)
    mixed[:,:,1] = img_contours
    mixed[:,:,2] = np.zeros_like(model3dproj_contours)

    mi_this = u2d.mutual_information_val(np.array(model3dproj_contours), img_contours)
    if plothere == True :
        fig, ax = plt.subplots(1, 1)
        ax.imshow(mixed)
        ax.set_title('MutInfo:' + str(round(mi_this, 4)))
    return(mi_this)

def CalcInd1_from_multiplecontours(x0=[0,0,0,0,0,0]):

    rot = x0[3:6] #R2=R
    R2 = eul2rot(rot)
    #eul2rot([psi1,theta1,phi1])
    t2 = x0[0:3]

    import copy
    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=False)

    projected_img_model3dv1[projected_img_model3dv1 == 0] = 1#Julius: Check values 1 and 2
    projected_img_model3dv1[projected_img_model3dv1 == 255] = 2

    # Maximiz mi  == Minimize -mi
    imgIR_contours_multi = create_multicontours(result_image_seg)
    ind = np.mean(imgIR_contours_multi*projected_img_model3dv1)
    # I want the maximun ind
    return(-ind)


def CalcIM_from_contours(x0=[0,0,0,0,0,0]):

    rot = x0[3:6] #R2=R
    R2 = eul2rot(rot)
    #eul2rot([psi1,theta1,phi1])
    t2 = x0[0:3]
    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=False)

    # Maximiz mi  == Minimize -mi
    model3dproj_contours = model3d_contours(projected_img_model3dv1)
    mi = mi_both_contours(img_contours, model3dproj_contours, plothere=False)#Julius:error where imb_contours is called?

    return(-mi)



def CalcProd_from_segmented(x0=[0,0,0,0,0,0]):

    rot = x0[3:6] #R2=R
    #eul2rot([psi1,theta1,phi1])
    t2 = x0[0:3]

    # Segmented images (2 groups in each image)
    R2 = eul2rot(rot)
    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=False)
    imgIR_segmented = IRsegmentation_clustering(IRimageTempData, plotit=False)
    ret, imgIR_segmentedv1 = cv2.threshold(imgIR_segmented, imgIR_segmented.mean(), 255, cv2.THRESH_BINARY)

    npix_matches = np.sum((imgIR_segmentedv1 / 255)*(projected_img_model3dv1 / 255))

    return(-npix_matches)


def CalcCrossCorr_from_segmentation(x0=[0,0,0,0,0,0]):
    # Requires: model3D, result_image_seg

    print("Using x0:" + str(x0))
    rot = x0[3:6] #R2=R
    t2 = x0[0:3]

    R2 = eul2rot(rot)
    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=False)
    image_templated = cv2.matchTemplate(projected_img_model3dv1, result_image_seg, cv2.TM_CCORR_NORMED)
    out = image_templated[0][0]
    return(-out)

def Calc2Ind_from_segmentation(x0):
    i1 = CalcCrossCorr_from_segmentation(x0)
    i2 = CalcIM_from_segmentation(x0)
    return(-i1*i2)


def CalcProd_from_segmented_v2_t(x0_t):
    newx0 = np.hstack([x0_t,x0[3:6]])
    #print(f'Run CalcProd_from_segmented using {bcolors.WARNING}x0=" + str(newx0) + "{bcolors.ENDC}')
    print("Run CalcProd_from_segmented using *x0* =" + str(newx0))
    return(CalcProd_from_segmented(newx0))

def plot_segmentions_and_both(x0):
    rot = x0[3:6]
    t2 = x0[0:3]
    R2 = eul2rot(rot)
    result_image_seg = IRsegmentation_clustering(IRimageTempData, plotit=False)
    projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D, R2, t2, plotit=True)
    fig, axarr = plt.subplots(1, 3, figsize=(10,5))
    axarr[0].imshow(result_image_seg)
    axarr[0].set_title('ReferencePose (RGB Pose) \n MI: ' + str(round(-CalcIM_from_segmentation(x0), 3)) +
                       '\n CrossProd: ' + str(round(-CalcCrossCorr_from_segmentation(x0), 3)) +
                       '\n IndFromMultiContour: ' + str(round(-CalcInd1_from_multiplecontours(x0), 3))
                       )
    axarr[1].imshow(projected_img_model3dv1)
    axarr[2].imshow(mark_boundaries(projected_img_model3dv1, result_image_seg.astype("int"), color=[255, 0, 0], mode={'thick'}))
    return(fig)


def get_manual_registration(img_filename):
    filename = baseDir + '/11_Texturing/ManualAlignment/'+ img_filename.replace("jpg","out")
    f = open(filename, "r")
    matR = list()
    for i, line in enumerate(f):
        #print(i)
        if i>=3 and i<=5:
            f1 = line
            l1  = np.array([float(x) for x in f1.replace("\n", "").split(" ")])
            if i>=4:
                l1 = l1*(-1)
            #print(l1)
            matR.append(l1)

        if i == 6:
            f1 = line
            l1  = np.array([float(x) for x in f1.replace("\n", "").split(" ")])
            t_ref = l1*[1,-1,-1]
    f.close()

    R_ref = np.array(matR)

    return(R_ref,t_ref)


def save_pose_meshlab_format(K,R,t, fileout):

    file1 = open(fileout, "w")
    file1.write("# Bundle file v0.3\n")
    file1.write("1 0\n")
    file1.write(str(K[0,0]) + " 0 0\n")
    file1.write(str(R[0,0]) + " " + str(R[0,1]) + " " + str(R[0,2]) + "\n")
    file1.write(str(-R[1,0]) + " " + str(-R[1,1]) + " " + str(-R[1,2]) + "\n")
    file1.write(str(-R[2,0]) + " " + str(-R[2,1]) + " " + str(-R[2,2]) + "\n")
    file1.write(str(t[0]) + " " + str(-t[1]) + " " + str(-t[2]) + "\n")
    file1.write("0 0 0")
    file1.close()

    return(True)
#Julius: Implement a cross correlation in polar coordinates (first use paint)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ALG 1: COMPARING SHAPES
#Julius:Improve and create a new function that calculate a new metric.
#Julius: Check function Calc2Ind_from_segmentation
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
t_PSO,t_Grad = 0, 0
from scipy.optimize import minimize
if poseIdMarker!="NoMarker":
    selected_vertices_marker = get_pos_markers(poseIdMarker,model3D)
result_image_seg = IRsegmentation_clustering(IRimageTempData, plotit=False)
projected_img_model3dv1 = from_3dmodel_to_2dimage_nomarker(model3D,R, t, plotit=True)
mi_before = u2d.mutual_information_val(result_image_seg, projected_img_model3dv1)
# CROSS-CORRELATION
# https://dsp.stackexchange.com/questions/28322/python-normalized-cross-correlation-to-measure-similarites-in-2-images
im1=IRimageTempData
sh_row, sh_col = im1.shape
im2=projected_img_model3dv1
def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product
d = 1
correlation = np.zeros_like(im1)
for i in range(d, sh_row - (d + 1)):
    for j in range(d, sh_col - (d + 1)):
        correlation[i, j] = correlation_coefficient(im1[i - d: i + d + 1,j - d: j + d + 1],im2[i - d: i + d + 1,j - d: j + d + 1])

fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(IRimageTempData)#IR
axarr[1].imshow(projected_img_model3dv1)#0' y 1' de la proyección del modelo 3D
axarr[2].imshow(correlation)
#io.imshow(correlation, cmap='gray')
#io.show()

# PLOT
fig, axarr = plt.subplots(1, 3)
axarr[0].imshow(result_image_seg)
axarr[0].set_title('MI w R,t: ' + str(round(mi_before, 4)))
axarr[1].imshow(projected_img_model3dv1)
axarr[2].imshow(mark_boundaries(projected_img_model3dv1, result_image_seg.astype("int"),color=[255,0,0]))

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Initial segmentation - w RGB camera pose. (R, T)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
psi1,theta1,phi1 = rot2eul(R)
x0=[t[0],t[1],t[2],psi1,theta1,phi1]
fig = plot_segmentions_and_both(x0)
fig.savefig(thermaloutput_path + "Registration_Step0_" + poseId + ".jpg")
plt.close(fig)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Optimization w Initial segmentation - using RGB camera pose: (R, T)
#Julius: Takes time to process
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
psi1,theta1,phi1 = rot2eul(R)
x0=[t[0],t[1],t[2],psi1,theta1,phi1]
'''MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
                    'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']'''
#The other methods require a jacobian parameter :https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cobyla']
times={}

mask=IRimageTempData>25#25 skin temperature
IRimageTempData_m=IRimageTempData*mask
img_contours = IRimage_contours(IRimageTempData_m)
IRimageTempData_ft=IRimageTempData_m.flatten()
img_contours_ft=img_contours.flatten()
fig, axarr = plt.subplots(1, 5)
axarr[0].imshow(IRimageTempData)
axarr[1].imshow(projected_img_model3dv1)
axarr[2].imshow(correlation)
axarr[3].imshow(img_contours)
fig=plt.figure()
plt.hist(IRimageTempData_ft,256,[0,256])
plt.show()
fig=plt.figure()
plt.hist(img_contours_ft,256,[0,256])
plt.show()
x_borde=[]
y_borde=[]
fig=plt.figure()
for i in range(img_contours.shape[0]):
    for j in range(img_contours.shape[1]):
        if (img_contours[i][j]==255):
            x_borde.append(i)
            y_borde.append(j)
#Baricenter coordinates
x_bar=round(np.mean(x_borde))
y_bar=round(np.mean(y_borde))
plt.scatter(x_borde,y_borde,marker=',',s=np.ones(len(x_borde)))
#plt.scatter(x_bar,y_bar,marker='+')#plot the baricenter
plt.axis('off')
plt.show()
plt.savefig(baseDir + "filtered_image.jpg")
plt.close(fig)

dst_1 = cv2.imread(baseDir + "filtered_image.jpg",0)#Julius: change to something thinner
mask=dst_1>200#25 skin temperature

fig=plt.figure()
plt.imshow(dst_1)
plt.show()

dst=dst_1*mask

fig=plt.figure()
plt.imshow(dst)
plt.show()

ret, labels, stats, centroids=cv2.connectedComponentsWithStats(dst, connectivity=4)
areas = stats[:,cv2.CC_STAT_AREA][1:]
temporal=stats[:,4]>50#18/05: solo tomar el borde y el fondo
for i in range(len(temporal)):
    if temporal[i]:
        fig = plt.figure()
        plt.imshow(labels == i)
        plt.title("Label "+str(i))
        plt.show()

p10 = np.quantile(areas,0.25) - 0.1*np.quantile(areas,0.25)
p90 = np.quantile(areas,0.75) + 0.1*np.quantile(areas,0.75)
keep_points = np.logical_and(stats[:,cv2.CC_STAT_AREA]>p10, stats[:,cv2.CC_STAT_AREA]<p90)
print(keep_points)
new_labels=np.zeros((labels.shape))
for i in range(len(keep_points)):
    if keep_points[i]:
        new_labels=new_labels+(labels==(i+1))

print("R U Ok?")
fig=plt.figure()
plt.imshow(new_labels)
plt.show()

#To Polar coordinates
r=[]
theta=[]
for i in range(len(x_borde)):
    r_temp=np.sqrt(((x_borde[i]-x_bar)**2)+((y_borde[i]-y_bar)**2))
    r.append(r_temp)
    theta_temp=np.arctan2((y_borde[i]-y_bar),(x_borde[i]-x_bar))
    #if (x_borde[i]<x_bar):
    #    r_temp=r_temp*(-1)
    #theta_temp = np.arcsin((y_bar - y_borde[i]) / r_temp)
    theta.append(theta_temp)
fig=plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r,marker=',')
plt.show()
'''


for i in range(len(MINIMIZE_METHODS)):
    method=MINIMIZE_METHODS[i]
    start = time.time()
    #result = minimize(Calc2Ind_from_segmentation, x0, method='Nelder-Mead', options={'disp': True}, tol=1e-4)#Julius:Try with other methods, Swarp¿?
    #Julius: Try with another functions, the contours ones and the other ones.
    #result = minimize(Calc2Ind_from_segmentation, x0, method=MINIMIZE_METHODS[i], options={'disp': True}, tol=1e-4)
    result = minimize(CalcIM_from_contours, x0, method=MINIMIZE_METHODS[i], options={'disp': True}, tol=1e-4)
    end = time.time()
    t_Gradv1 = (end - start)
    fig = plot_segmentions_and_both(result.x)
    fig.savefig(thermaloutput_path + "outputs_registration/" + "FinalRegistration_" + method + "_" + img_filename )
    plt.close(fig)
    times[method]=t_Gradv1
print(times)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Comparison
#:::::::::::::::::::::::::::::::::::::::::::
x0 = result.x

#Reference vectors
R_ref, t_ref = get_manual_registration(img_filename)
R_ref_vec = rot2eul(R_ref)

# RGB vectors
R_vec = rot2eul(R)

# Vectos from this algorithm
R2_vec = x0[3:6]
t2 = x0[0:3]

# Get metadata for thermal images:
Temp3DModelsPath = baseDir + "/13_Thermal3DModels/"
file3DModelOutputPath = baseDir + "/15_Final3DModel/"
import shelve

temp = shelve.open(Temp3DModelsPath + '/shelve.out')
ListDistances = temp['ListDistances']
ListAngles = temp['ListAngles']
ListPoseIds = temp['ListPoseIds']
ListTemperatures = temp['ListTemperatures']
temp.close()
np_distances = np.vstack(ListDistances)
np_angles = np.vstack(ListAngles)
np_temperatures = np.vstack(ListTemperatures)
poseids = np.vstack(ListPoseIds)

# Average distance:
idx = np.where(poseids==poseId)[0]
selected3D = np.array(model3D.vertices)[np.where(np_temperatures[idx]>0)[1],:]
dif_pos = selected3D.mean(axis=0) - t
avg_dist_to_3dmodel_t = np.power(np.sum(np.power(dif_pos,2)),0.5)*ScaleFactor
dif_pos = selected3D.mean(axis=0) - t2
avg_dist_to_3dmodel_t2 = np.power(np.sum(np.power(dif_pos,2)),0.5)*ScaleFactor
dif_pos = selected3D.mean(axis=0) - t_ref
avg_dist_to_3dmodel_tref = np.power(np.sum(np.power(dif_pos,2)),0.5)*ScaleFactor

# Output:
SilentMkdir(thermaloutput_path + "outputs_registration/")
output = (baseDir,poseId,img_filename,ScaleFactor,
            R_ref_vec[0],R_ref_vec[1],R_ref_vec[2], t_ref[0],t_ref[1],t_ref[2], avg_dist_to_3dmodel_tref,
            R_vec[0],R_vec[1],R_vec[2], t[0],t[1],t[2], avg_dist_to_3dmodel_t,
            R2_vec[0],R2_vec[1],R2_vec[2], t2[0],t2[1],t2[2], avg_dist_to_3dmodel_t2,
            t_PSO,t_Gradv1
          )
with open(thermaloutput_path + "outputs_registration/" + "RegistrationResults_v1_" + img_filename.replace(".jpg","") + ".txt" , 'w') as f:
    for item in output:
        f.write("%s\n" % item)


CameraMatrix_ir = np.array([[616.21, 0., 240],
                            [0., 616.21, 320],
                            [0., 0., 1.]])
fileout = thermaloutput_path + "outputs_registration/PoseFlir_RGBsensor_" + img_filename.replace(".jpg","") + ".out"
save_pose_meshlab_format(K,R,t, fileout)
fileout = thermaloutput_path + "outputs_registration/PoseFlir_IRsensorEstimated_" + img_filename.replace(".jpg","") + ".out"
save_pose_meshlab_format(CameraMatrix_ir,eul2rot(R2_vec),t2, fileout)
fileout = thermaloutput_path + "outputs_registration/PoseFlir_IRManualAlignment_" + img_filename.replace(".jpg","") + ".out"
save_pose_meshlab_format(CameraMatrix_ir,R_ref,t_ref, fileout)

'''

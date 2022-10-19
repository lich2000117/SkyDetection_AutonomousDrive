# Import Packages
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def read_images(img_path):
    """For reading in images from a filepath as graycsale. (From Workshop 8)"""
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    return img

def show(img_L):
    """Display the result"""
    plt.subplots(figsize=(5, 5)) 
    plt.subplot(3,1,1)
    plt.imshow(img_L, cmap='gray')  
    plt.title('Left image')
    plt.axis('off')

### ========= 1. START Use gradient only to define optimal separate line START ==============


def sobel_filter(img_input):
    """Use Sobel filter to get edge map for x and y"""
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.GaussianBlur(img_input, (3, 3), 0)
    gray = cv2.GaussianBlur(src, (3, 3), 0)

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    return abs_grad_x, abs_grad_y, grad 


def Calculate_border(grad_img, t):
    """
    From image up to bottom, iterate along x-axis (width)
    calculate and extract pixel location for first gradient > threshold.
    Like edge detection, detect first edge from sky  

    return: list of y value for borders. index are x value.

    Note: to select a pixel in image :
            image[y,x],  where y=height, x=width
    """
    height = grad_img.shape[0]  # 400 here
    width = grad_img.shape[1]  #800 here for example, with 800x400 shape
    y_list = [] # list of pixels for border between sky and ground
    for x in range(width):
        border_y = height
        for y in range(1, height):
            if (grad_img[y,x]>t):
                border_y = y
                break
        y_list.append(border_y)
    return y_list

def check_no_sky_in_img(divider, img, thresh=2):
    """Check if this img does not contain sky"""
    if (np.array(divider).mean()- img.shape[0])<thresh:
        return False
    return True

def Global_energy_function(img, divider, gamma=2):
    """Simplified energy function to represent the differences of sky region and ground region.
    Takes input of an array (y value of the divider line) of sky and ground respectively.
    
    Computer Energy between sky and ground. (good dividing point will create largest variance in ground gradient array 
    and smallest variance in sky gradient array.).
    
    The larger the output, better the division point choice.
    
    Hyper-parameter: Gamma (Penalty for sky variance.) # coeffibbcient to penalise high gradient in sky. 2 from paper.
    """
    
    # get two regions arrays.
    total_sky_array=[]
    total_ground_array=[]
    total_J = []
    for x in range(len(img[0])):
        # for every column in image
        y_sky = divider[x]
        sky_array = img[:y_sky, x]
        ground_array = img[y_sky:, x]

        # calculate energy
        sky_mean = sky_array.mean()
        ground_mean = ground_array.mean()
        total_J.append(np.abs(ground_mean - ((gamma*sky_mean))))  # maximise this function so we have maxi ground_std and smallest sky_std.
        total_sky_array.append(sky_array)
        total_ground_array.append(ground_array)
    J = np.array(total_J)
    return total_sky_array, total_ground_array, J.mean()
    
    
def optimise_Calculate_border(img, gradient_img, step, t_min, t_max, gamma, msg=True):
    """Find optimal threashold t based on optimising energy function, return array of y-value of pixels.
    input: 
        step size for t.
        t_min and t_max
        msg, display train message
    return: 
        1. b_best: 
            list of pixels' y coordinate to represent border location
            list index: x corrdinate
            For example,
                [2,1,1,1,3,4] -> 6 columns in image, border pixel's y coordinate corresponds to 2,1,1,1,3,4 (1 means the border is to the up)
        2. best_sky_array: 
            array of sky pixel color valus for every column in the image. 
            [array([10,255,0]), array([255,255]), array([0])],  contains array[0] if no pixel is detected as border
        3. best_ground_array:
            same except for capturing ground column pixels.
        4. J_hist:
            list of energy value J, can be used for plotting.
    """
    n = int((t_max-t_min)/step)

    print("Number of Iterations: ",n)
    print("Step size:",step)

    J_max = -1
    b_best= None
    best_t = None
    best_sky_array = []
    best_ground_array = []

    #plotting
    J_hist = []
    out_x = n

    t=t_min
    for k in range(1,n+1):
        t = t + step
        b_temp = Calculate_border(gradient_img, t)
        sky_array, ground_array, J_temp = Global_energy_function(img,b_temp,gamma=2)
        #print(J_temp)
        if ((len(sky_array)==0) or (len(ground_array)==0)):
            # if we cannot find a line in between, break
            print("Threshold too large, cannot find line, break, Iterations: " + str(k))
            J_hist.append([0]*(n-k))
            break;
        if J_temp > J_max:
            if msg:
                print("\n")
                print("Find new best J: " + str(J_temp))
                print("Update Treashold, Current Iteration: "+str(k))
                print("New t: " + str(best_t))
            J_max=J_temp
            b_best=b_temp
            best_t=t
            best_sky_array = sky_array
            best_ground_array = ground_array
        J_hist.append(J_temp)
    print("Best T value: ", best_t)
    return b_best, best_sky_array, best_ground_array, J_hist


### ========= 1. END Use gradient only to define optimal separate line END ================

# Current assumption: img always can be divided by sky and ground
#    What if some sky is not presented? How to detect partial sky?

### ========= 2. START Use Sobel + SkyRegion to detect vertical separate (fake sky) START ================

def Calculate_VerticalDivider(y_divider, grad_img_x, t):
    """
    Calculate vertical line that separate sky and building for example, to detect fake sky.
    """
    height = grad_img_x.shape[0]  # 400 here
    width = grad_img_x.shape[1]  #800 here for example, with 800x400 shape
    vertical_lines_x = [] # store list of x where we find a vertical divider in sky
    y_list = [] # list of pixels for border between sky and ground
    for x in range(width-1):
        # only calculate region in sky.
        for y in range(y_divider[x]):
            if (grad_img_x[y,x]>t):
                vertical_lines_x.append(x)
    if len(vertical_lines_x)==0:
        return []
    hist, bin_edges = np.histogram(vertical_lines_x, bins=1)
    if int(len(vertical_lines_x)/10)>1:
        hist, bin_edges = np.histogram(vertical_lines_x, bins=int(len(vertical_lines_x)/10))
    
    out=vertical_lines_x
    if vertical_lines_x:
        out=[vertical_lines_x[0]]
        x_cur = vertical_lines_x[0]+10 # treat within 10 pixel as a single point
        for i in range(len(vertical_lines_x)):
            x1=vertical_lines_x[i]
            if x1>x_cur:
                x_cur = vertical_lines_x[i]+10
                out.append(vertical_lines_x[i])
    return out


def k_mean_rough_detect(img_in):
    """ 
    rough detect, divide image into two horizontal part to capture sky part as a rectangle.
    Use K-Mean cluster to determine which small region is sky, which is fake sky region that is in fact ground objects.
    For purpose of this algorithm,
    parameters are fixed.

    Return: 
        1. index of y value (vertical height of the image) to tell us how to divide the image into two parts (one above, one down below).
        2. segmented_image, for plotting ONLY.
    """
    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(img_in)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) #criteria
    k = 2 # Choosing number of cluster
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 

    centers = np.uint8(centers) # convert data into 8-bit values 
    segmented_data = centers[labels.flatten()] # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((img_in.shape)) # reshape data into the original image dimensions

    # get sky region's y value to determine which is sky region
    sky_region_y = np.where(labels.flatten()==1)[0][0]
    if (sky_region_y==0):
        #re-assign value if we find 0 as deviding point
        sky_region_y = np.where(labels.flatten()==0)[0][0]
    return sky_region_y, segmented_image


def k_mean_get_sky_ground_color(img_in, sky_region_y):
    """ 
    precise K mean, use 2 clusters for upper image
    and 1 cluster for ground image.

    calculate color difference between two sky clusters and true ground cluster
    (fake sky cluster should be similar to true ground cluster)
    
    Use K-Mean cluster to determine which small region is sky, which is fake sky region that is in fact ground objects.
    For purpose of this algorithm,
    parameters are fixed.

    Return: 
        1. true sky color
        2. true ground color
    """
    
    # select top sky region image
    img = cv2.cvtColor(img_in,cv2.COLOR_GRAY2RGB)
    img1 = img[:sky_region_y,:]

    Z = img1.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    sky_ret,sky_label,sky_center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    sky_center = np.uint8(sky_center)
    sky = sky_center[sky_label.flatten()]
    sky2 = sky.reshape((img1.shape))


    # select bottom ground region
    img2 = cv2.cvtColor(img_in,cv2.COLOR_GRAY2RGB)
    img2 = img2[sky_region_y:,:]
    Z = img2.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    K = 1
    ground_ret,ground_label,ground_center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    ground_center = np.uint8(ground_center)
    ground = ground_center[ground_label.flatten()]
    ground2 = ground.reshape((img2.shape))

    ## Find true sky center which is not similar to ground
    max_dist = 0
    true_sky_color = 0
    true_ground_color = ground_center[0].mean()
    for cluster in sky_center:
        cur_dist = cluster.mean() - true_ground_color
        if cur_dist > max_dist:
            max_dist = cur_dist
            true_sky_color = cluster.mean()

    return true_sky_color, true_ground_color

def check_sky_region_true(img, b_best, x, true_sky_color, true_ground_color):
    """
    check if this column is similar to ground cluster or sky cluster.
    Also known as check_if_sky() in jupyter notebook.
    """
    color = img[b_best[x],x]
    col_mean = color.mean()
    diff_sky = abs(col_mean-true_sky_color)
    diff_ground = abs(col_mean-true_ground_color)
    #print(col_mean, diff_sky, diff_ground)
    if diff_sky < diff_ground:
        # if it is more sky
        return True
    else:
        return False

def remove_fake_sky(img_in, b_best, vertical_lines_x, true_sky_color, true_ground_color, Thresh=0.5):
    """
    Remove fake sky region from two areas defined in previous functions, only keep true sky.
    
    Input:
        vertical_lines_x: x coordinate of how to divide image based on vertical gradient.
        
        Thresh = 0.5 # determine untill how many (percentage) columns we detected are sky, we treat this region as sky
    """
    if not vertical_lines_x:
        # if cannot divide image into different sky parts, return original array
        return b_best

    new_sky_array = []
    vertical_lines_x.append(img_in.shape[1]) # add border of the image as well
    for i in range(len(vertical_lines_x)):
        if i == 0:
            start = 0
        else:
            start = vertical_lines_x[i-1]
        end = vertical_lines_x[i]
        count_true_column = 1 # count how many columns is sky region 
        for x in range(start, end):
            # for every column in our sky region image (every sky column divided by divider we found)
            if (check_sky_region_true(img_in, b_best, x, true_sky_color, true_ground_color)):
                count_true_column += 1
        #print(start,end, count_true_column)
        if count_true_column/(end-start) > Thresh:
            # this region is sky region, add index.
            for x in range(start, end):
                new_sky_array.append(b_best[x])
        else:
            for x in range(start, end):
                new_sky_array.append(0)
    return new_sky_array

def check_if_pixel_in_sky(x,y, b_best):
    """
    Check if given pixel is in the sky region
    x: 0-800
    y: 0-400
    """
#     for x in range(len(b_best)):  # for every column
#         b_best[x]  #lowest border pixel y value
    if y<b_best[x]: return True
    return False
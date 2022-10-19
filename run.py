from detector import *


IMG_PATH = "./images/img1.jpg"
SHOWIMG=False

def get_sky(img_path,SHOWIMG=False):
    ## For single image

    ### ========= 1. START Use gradient only to define optimal separate line START ==============
    img_L= read_images(img_path)
    if SHOWIMG: 
        show(img_L)
        plt.show()

    # 2. get gradient image
    img_in = img_L

    grad_x_img, grad_y_img, grad_full_img = sobel_filter(img_in)
    if SHOWIMG: 
        plt.imshow(grad_full_img, cmap="gray")
        plt.axis("off")
        plt.show()

    # 3. find optimal border to divide sky and ground

    step=0.5
    t_min = 8
    t_max = 60  
    gamma = 2

    b_best, sky_array, ground_array, J_hist = optimise_Calculate_border(img_in, grad_full_img, step,t_min,t_max,gamma,msg=False)
    if SHOWIMG: 
        plt.title("Energy Value vs Threshold Value")
        plt.xlabel("Threshold Value")
        plt.ylabel("Energy")
        plt.plot(np.arange(t_min,t_max,step),J_hist)
        plt.show()

    # 4. show current image
    if SHOWIMG: 
        plt.imshow(img_in, cmap='gray')
        for i in range(len(sky_array)):
            plt.vlines(x=i, ymin=0, ymax=len(sky_array[i]))
            plt.title("Optimal t value to divide")
        plt.axis("off")
        plt.show()

    ### ========= 1. END Use gradient only to define optimal separate line END ================

    # Current assumption: img always can be divided by sky and ground
    #    What if some sky is not presented? How to detect partial sky?

    ### ========= 2. START Use Sobel + SkyRegion + KMeans to detect vertical separate (fake sky) START ================


    # check if we have building or ground object obstruct sky.

    def detectfakeSky(y_divider, grad_img_x, t=70):
        """
        Calculate vertical line that separate sky and building for example.
        """
        height = grad_img_x.shape[0]  # 400 here
        width = grad_img_x.shape[1]  #800 here for example, with 800x400 shape
        vertical_lines_x = [] # store list of x where we find a vertical divider in sky
        y_list = [] # list of pixels for border between sky and ground
        for x in range(width-1):
            # only calculate first pixel to check.
            for y in range(0,1):
                if (grad_img_x[y,x]>t):
                    return True
        return False;

    fakeSky = detectfakeSky(b_best, grad_x_img, t=70)



    if fakeSky:

        # 2.1 get vertical divider
        t = 80 # threshold to determine whether it is a divider or not
        vertical_lines_x = Calculate_VerticalDivider(b_best, grad_x_img, t)


        if SHOWIMG:
            # draw output
            plt.imshow(img_in, cmap='gray')
            plt.plot(range(len(img_in[0])), b_best)
            for i in range(len(vertical_lines_x)):
                plt.vlines(x=vertical_lines_x[i], ymin=0, ymax=50,color="red")
            plt.title("Vertical Divider plot")
            plt.axis("off")
            plt.show()

        ## 2.2 use K-means and color differentiation to detect which region is sky and which is ground (fake sky)

        # determine rough division using first round of k-mean
        sky_region_y, m = k_mean_rough_detect(img_in)
        if SHOWIMG:
            plt.imshow(m)
            plt.axis("off")
            plt.show()

        # 2.3 calculate true sky and ground pixel colours using k-mean
        sky_true_color_array, ground_true_color_array = k_mean_get_sky_ground_color(img_in, sky_region_y)
        if SHOWIMG:
            plt.imshow(img_in, cmap='gray')
            for i in range(len(b_best)):
                plt.vlines(x=i, ymin=0, ymax=b_best[i])

            plt.title("Sky Detection Output (BEFORE REFINE)")
            plt.axis("off")
            plt.show()

        # 2.4
        b_best = remove_fake_sky(img_in, b_best, vertical_lines_x, sky_true_color_array, ground_true_color_array)

    plt.imshow(img_in, cmap="gray")
    for i in range(len(b_best)):
        plt.vlines(x=i, ymin=0, ymax=b_best[i])

    plt.title("Sky Detection Output")
    plt.axis("off")
    plt.show()



get_sky(IMG_PATH, SHOWIMG)
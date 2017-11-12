import numpy as np
import cv2
import sys
from self_driving.optical_flow.python import video
from scipy import misc


def color_seg(img_raw, red_thresh=0, green_thresh=0, blue_thresh=0):
    img_color_mask = np.copy(img_raw)
    red_mask = img_raw[:,:,0] < red_thresh
    green_mask = img_raw[:,:,1] < green_thresh
    rgb_mask = np.logical_or(red_mask, green_mask)
    img_color_mask[rgb_mask] = [0,0,0]
    return img_color_mask


def draw_lines_extrapolate(img, lines, color=[255, 0, 0], thickness=2):
    # Assume lines on left and right have opposite signed slopes
    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0: continue; # Infinite slope
            slope = float(y2-y1) / float(x2-x1)
            if .5 <= abs(slope) < 1.0: # Discard unlikely slopes
                if slope > 0:
                    left_xs.extend([x1, x2])
                    left_ys.extend([y1, y2])
                else:
                    right_xs.extend([x1, x2])
                    right_ys.extend([y1, y2])

    left_fit = np.polyfit(left_xs, left_ys, 1)
    right_fit = np.polyfit(right_xs, right_ys, 1)

    y1 = img.shape[0] # Bottom of image
    y2 = img.shape[0] / 2+ 50 # Middle of view
    x1_left = (y1 - left_fit[1]) / left_fit[0]
    x2_left = (y2 - left_fit[1]) / left_fit[0]
    x1_right = (y1 - right_fit[1]) / right_fit[0]
    x2_right = (y2 - right_fit[1]) / right_fit[0]
    y1 = int(y1); y2 = int(y2);
    x1_left = int(x1_left); x2_left = int(x2_left);
    x1_right = int(x1_right); x2_right = int(x2_right);

    cv2.line(img, (x1_left, y1), (x2_left, y2), color, thickness)
    cv2.line(img, (x1_right, y1), (x2_right, y2), color, thickness)


if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0
    cam = video.create_capture(fn)
    index = 0
    while True:
        ret, img = cam.read()

        if img is None:
            break

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_img = color_seg(rgb, red_thresh=200, green_thresh=150, blue_thresh=0)

        gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

        # Define a kernel size and apply Gaussian smoothing
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)
        ignore_mask_color = 255

        # This time we are defining a four sided polygon to mask
        imshape = img.shape
        vertices = np.array([[(0 + 120, imshape[0]),
                              (imshape[1] / 2 - 15, imshape[0] / 2 + 40),
                              (imshape[1] / 2 + 15, imshape[0] / 2 + 40),
                              (imshape[1] - 50, imshape[0])]],
                            dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 5  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10  # minimum number of pixels making up a line
        max_line_gap = 2  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

        if lines is None:
            continue

        draw_lines_extrapolate(line_image, lines, thickness=8)

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(rgb, 1, line_image, 1, 0)
        misc.imsave(sys.argv[2] + 'frame_%d.png' % index, lines_edges)
        index += 1
    cv2.destroyAllWindows()

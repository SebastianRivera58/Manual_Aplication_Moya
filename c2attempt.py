import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "images/Wellspicture.png"
img = cv2.imread(img_path)
img_copy = img.copy()

# Drawing state variables
drawing_rect = False
start_point = None

rectangles = [] 
circles = []

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, start_point, img_copy
    
    # Left click + drag for rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_rect = True
        start_point = (x, y)
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing_rect:
        # Show preview while dragging
        temp_img = img_copy.copy()
        cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("demo", temp_img)
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Finalize rectangle on img_copy
        drawing_rect = False
        cv2.rectangle(img_copy, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("demo", img_copy)
        
    
    # Right click for circle
    elif event == cv2.EVENT_RBUTTONDOWN:
        r = 7
        cv2.circle(img_copy, (x, y), r, (255, 0, 0), 2)
        circles.append((x,y)) 
        cv2.imshow("demo", img_copy)


def r_squared(x,y):
        # Convert to NumPy arrays
    print (x)
    print(y)
    x = np.array(x)
    y = np.array(y)
    # Fit a linear regression: y = m*x + b
    m, b = np.polyfit(x, y, 1)  # 1 = linear

    # Predicted y values from the linear model
    y_pred = m * x + b

    # Calculate R²
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    print (r_squared)
    return r_squared

def plot_lines(dicts):
    x = []  # Reset x and y for each new plot
    y = []
    green=[]
    blue=[]
    red=[]
    num=0
    # Loop through each dictionary in i
    for j in range(len(dicts[1])):
        num+=0.2
        x.append(num)  # Append index to x
        y.append(dicts[1][j]["mean"]-dicts[1][0]["mean"])  # Append "mean" value to y
        green.append(dicts[1][j]["green"]-dicts[1][0]["green"])
        blue.append(dicts[1][j]["blue"]-dicts[1][0]["blue"])
        red.append(dicts[1][j]["red"]-dicts[1][0]["red"])

    #calculate r squared
    normal_r=r_squared(x[:-1], y[1:])
    green_r=r_squared(x[:-1], green[1:])
    blue_r=r_squared(x[:-1], blue[1:])
    red_r=r_squared(x[:-1], red[1:])

    #plot the graphs
    plt.plot(x[:-1], y[1:], color='black', label='Gray')
    plt.plot(x[:-1], green[1:], color='green', label='Green')
    plt.plot(x[:-1], blue[1:], color='blue', label='Blue')
    plt.plot(x[:-1], red[1:], color='red', label='Red')

    #add the text of r squared
    plt.text(x[1], y[-1], f'{normal_r:.4f}', color='black', fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right')
    
    plt.text(x[1], green[1], f'{green_r:.4f}', color='green', fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right')
    
    plt.text(x[1], blue[1], f'{blue_r:.4f}', color='blue', fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right')
    
    plt.text(x[1], red[1], f'{red_r:.4f}', color='red', fontsize=10, 
         verticalalignment='bottom', horizontalalignment='right')
    print(x)
    print(y)
    # for i in dicts:
    #     x = []  # Reset x and y for each new plot
    #     y = []
        
    #     # Loop through each dictionary in i
    #     for j in range(len(i)):
    #         # if i[j]["Centroid"][1]>150:
    #         #     continue
    #         x.append(j)  # Append index to x
    #         y.append(i[j]["mean"])  # Append "mean" value to y

    #     plt.plot(x[1:], y[1:])  # Plot the line for this particular set of data
    #     print(x)
    #     print(y)

    plt.title("Multiple Lines Plot4")
    plt.xlabel("X axis")
    plt.ylabel("Mean Value")

    # Show the plot
    plt.show()



cv2.imshow("demo", img)
cv2.setMouseCallback("demo", mouse_callback)

print("Left click + drag: Draw rectangle")
print("Right click: Draw circle")
print("Press 'r' to reset, 'q' to quit")


while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset image
        img_copy = img.copy()
        cv2.imshow("demo", img_copy)


print("Saved circles:", circles)


if len(circles) > 1:
    x_coords = [circle[0] for circle in circles]
    y_coords = [circle[1] for circle in circles]
    r_squared(x_coords, y_coords)
else:
    print("Need at least 2 circles to calculate R²")


plot_lines(circles)

cv2.destroyAllWindows()
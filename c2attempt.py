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
blanks = []
Standards = []
Standards2 = []
Standards3 = []
key = cv2.waitKey(1) & 0xFF
user_finish = False

mode = 'select_blank'  # other value: 'normal'
pending_standard = None 
print("Mode:", mode, "- Right click to mark ONE blank, then press Enter to continue.")

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, start_point, img_copy, mode, blanks, Standards, Standards2, Standards3, circles
    
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
        if mode == 'select_blank':
            # mark blank with a distinct color (yellow) and store coordinate
            r = 7
            cv2.circle(img_copy, (x, y), r, (255, 255, 255), 2)  # white
            blanks.append((x, y))
            cv2.imshow("demo", img_copy)
            print("Blank selected, now press Enter to continue.")
            circles.append((x,y))
        elif mode == 'Standards':
            r = 7
            cv2.circle(img_copy, (x, y), r, (255, 0, 0), 2)  # blue
            Standards.append((x, y))
            cv2.imshow("demo", img_copy)
            if len(Standards) == 5:
                print("Standards selected, now press Enter to continue.")
            circles.append((x,y))
        elif mode == 'Standards2':
            r = 7
            cv2.circle(img_copy, (x, y), r, (0, 255, 0), 2)  # Green
            Standards2.append((x, y))
            cv2.imshow("demo", img_copy)
            if len(Standards2) == 5:
                print("Second standards selected, now press Enter to continue.")
            circles.append((x,y))
        elif mode == 'Standards3':
            r = 7
            cv2.circle(img_copy, (x, y), r, (0, 0, 255), 2)  # red
            Standards3.append((x, y))
            cv2.imshow("demo", img_copy)
            if len(Standards3) == 5:
                print("Thrird standards selected, now press Enter to continue.")
            circles.append((x,y))


def get_circle_stats(img, center, radius=7):
    """Return mean BGR and grayscale mean inside a circular ROI centered at center."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)  # filled circle mask
    # Mean for BGR
    b, g, r, _ = cv2.mean(img, mask=mask)
    # Grayscale mean
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = cv2.mean(gray, mask=mask)[0]
    return {"center": center, "radius": radius, "bgr_mean": (b, g, r), "gray_mean": gray_mean, "mask": mask}

def compute_rois(img, centers, radius=7):
    """Compute circle ROI stats for a list of centers. Returns list of dicts."""
    stats = []
    for c in centers:
        stats.append(get_circle_stats(img, c, radius))
    return stats

def r_squared(x,y):
        # Convert to NumPy arrays
    print(x)
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

def get_MPV_corrected(MPVSTD, MPVBLANK):
    """
    MPVSTD: list of roi-dicts (e.g. standards_stats) with keys "bgr_mean" and "gray_mean"
    MPVBLANK: list of roi-dicts (e.g. blanks_stats) with keys "bgr_mean" and "gray_mean"
    Returns: list of dicts, one per standard, each containing:
      - corrected_gray: list of (standard_gray - each_blank_gray)
      - corrected_bgr: list of (standard_bgr - each_blank_bgr) tuples
    """
    # get the blank values
    blank_grays = [b["gray_mean"] for b in MPVBLANK]
    blank_bgrs = [b["bgr_mean"] for b in MPVBLANK]

    corrected_list = []
    for s in MPVSTD:
        s_gray = s["gray_mean"]
        s_bgr = s["bgr_mean"]  # tuple tpe (b,g,r)

        corrected_gray = [s_gray - bg for bg in blank_grays]
        corrected_bgr = [tuple(si - bi for si, bi in zip(s_bgr, bb)) for bb in blank_bgrs]

        corrected_list.append({
            "center": s.get("center"),
            "orig_gray": s_gray,
            "orig_bgr": s_bgr,
            "corrected_gray": corrected_gray,
            "corrected_bgr": corrected_bgr
        })

    return corrected_list

def print_corrected_list(corrected_list, label="MPV corrected"):
    """Pretty-print the list returned by get_MPV_corrected."""
    print(f"{label}:")
    for si, s in enumerate(corrected_list):
        center = s.get("center")
        cxcy = f"{center[0]},{center[1]}" if center else "None"
        # original B,G,R order -> print as Gray, Red, Green, Blue like your sample
        b, g, r = s["orig_bgr"]
        print(f"Blank {si + 1} @ {cxcy} -> {s['orig_gray']:.2f}  {r:.2f}  {g:.2f}  {b:.2f}")
        # per-blank corrected values
        for j, (cg, cb) in enumerate(zip(s["corrected_gray"], s["corrected_bgr"])):
            cb_b, cb_g, cb_r = cb
            print(f"  STD{j + 1}: {cg:.3f}  {cb_r:.2f}  {cb_g:.2f}  {cb_b:.2f}")
    print()

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
        blanks.clear()
        Standards.clear()
        Standards2.clear()
        Standards3.clear()
        circles.clear()
        mode = 'select_blank'
        pending_standard = None
        cv2.imshow("demo", img_copy)
    elif key == 13 or key == 10:
            if mode == 'select_blank':
                if not blanks:
                    print("No blank selected yet. Right-click to add one before pressing Enter.")
                else:
                    if pending_standard:
                        mode = pending_standard
                        print(f"Blank confirmed for {pending_standard}. Now select {pending_standard} and press Enter when done.")
                        pending_standard = None
                    else:
                        mode = 'Standards'
                        print("Blank confirmed. Now select Standards and press Enter when done.")
            elif mode == 'Standards':
                if not Standards:
                    print("No standards selected yet.")
                if len(Standards) < 5:
                    print("Select more wells to continue. Around 5 should do.")
                else:
                   while True:
                        resp = input("Are there replicates? (y/n): ").strip().lower()
                        if resp not in ('y','n'):
                            print("Please type 'y' or 'n'.")
                            continue
                        if resp == 'y':
                            pending_standard = 'Standards2'
                            mode = 'select_blank'
                            print("Proceeding to Standards2 (replicate set 1). Right-click to add, press Enter when done.")
                            break
                        else:
                            mode = 'normal'
                            print("No replicates. Skipping Standards2/Standards3 and proceeding to next steps.")
            elif mode == 'Standards2':
                if not Standards2:
                    print("No standards selected yet.")
                if len(Standards2) < 5:
                    print("Select more wells to continue. Around 5 should do.")
                else:
                # ask about a third replicate set; if yes, request blank first
                    while True:
                        resp = input("Are there more replicates (Standards3)? (y/n): ").strip().lower()
                        if resp not in ('y','n'):
                            print("Please type 'y' or 'n'.")
                            continue
                        if resp == 'y':
                            pending_standard = 'Standards3'
                            mode = 'select_blank'
                            print("Replicates requested. Please select blank for Standards3 (right-click), then press Enter.")
                            break
                        else:
                            mode = 'normal'
                            print("No more replicates. Proceeding to next steps.")
                            break
            elif mode == 'Standards3':
                if not Standards3:
                    print("No standards selected yet.")
                if len(Standards3) < 5:
                    print("Select more wells to continue. Around 5 should do.")
                else:
                    print("Will now be calculating... Beeb Boob SDIYBT")

    
print(f"This is the blanks: {blanks}")
print(f"This is the first Standard: {Standards}")
print(f"This is the Second Standard: {Standards2}")
print(f"This is the Third Standard: {Standards3}")

radius = 7
blanks_stats = compute_rois(img, blanks, radius)
standards_stats = compute_rois(img, Standards, radius)
standards2_stats = compute_rois(img, Standards2, radius)
standards3_stats = compute_rois(img, Standards3, radius)

print("Blank stats:", [s["bgr_mean"] + (s["gray_mean"],) for s in blanks_stats])
print("Standards stats (B,G,R,Gray):", [s["bgr_mean"] + (s["gray_mean"],) for s in standards_stats])
print("Standards stats 2 (B,G,R,Gray):", [s["bgr_mean"] + (s["gray_mean"],) for s in standards2_stats])
print("Standards stats 3 (B,G,R,Gray):", [s["bgr_mean"] + (s["gray_mean"],) for s in standards3_stats])

first_MPVC = get_MPV_corrected(blanks_stats,standards_stats)
print_corrected_list(first_MPVC, "First MPV corrected")

second_MPVC = get_MPV_corrected(blanks_stats,standards2_stats)
print_corrected_list(second_MPVC, "Second MPV corrected")

Third_MPVC = get_MPV_corrected(blanks_stats,standards3_stats)
print_corrected_list(Third_MPVC, "Third MPV corrected")

# if len(circles) > 1:
#     x_coords = [circle[0] for circle in circles]
#     y_coords = [circle[1] for circle in circles]
#     r_squared(x_coords, y_coords)
# else:
#     print("Need at least 2 circles to calculate R²")


cv2.destroyAllWindows()

#Consolas, 'Courier New', monospace
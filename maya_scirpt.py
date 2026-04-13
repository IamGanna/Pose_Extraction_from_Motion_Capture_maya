import maya.cmds as cmds
import math
import numpy as np

# ==========================================
# 1. CORE MATH & ALGORITHMS
# ==========================================

def smooth_data_moving_average(data, window_size=11):
    if len(data) < window_size:
        return data

    smoothed = []
    pad = window_size // 2
    for i in range(len(data)):
        start = max(0, i - pad)
        end = min(len(data), i + pad + 1)
        smoothed.append(sum(data[start:end]) / (end - start))
    return smoothed


def perpendicular_distance(pt, start, end):
    x0, y0 = pt
    x1, y1 = start
    x2, y2 = end

    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    if denominator == 0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    return numerator / denominator


def rdp_algorithm(points, epsilon):
    dmax = 0.0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = rdp_algorithm(points[: index + 1], epsilon)
        right = rdp_algorithm(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[end]]


def find_critical_points(energy_curve, start_frame, velocity_threshold=0.5):
    critical = {"maxima": [], "minima": [], "inflection": []}

    for i in range(1, len(energy_curve) - 1):
        d1 = (energy_curve[i + 1] - energy_curve[i - 1]) / 2.0

        if abs(d1) < velocity_threshold:
            d2 = energy_curve[i + 1] - 2 * energy_curve[i] + energy_curve[i - 1]

            frame = start_frame + i
            if d2 < 0:
                critical["maxima"].append(frame)
            elif d2 > 0:
                critical["minima"].append(frame)
            else:
                critical["inflection"].append(frame)

    for i in range(2, len(energy_curve) - 1):
        d2_curr = energy_curve[i + 1] - 2 * energy_curve[i] + energy_curve[i - 1]
        d2_prev = energy_curve[i] - 2 * energy_curve[i - 1] + energy_curve[i - 2]
        if d2_curr * d2_prev < 0:
            frame = start_frame + i
            if frame not in critical["inflection"]:
                critical["inflection"].append(frame)

    return critical


def filter_minimum_gap(frames, min_gap=5):
    if not frames:
        return frames
    filtered = [frames[0]]
    for f in frames[1:]:
        if f - filtered[-1] >= min_gap:
            filtered.append(f)
    return filtered


def subsample_energy(energy_curve, step=2):
    return energy_curve[::step]


def distance_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


# ==========================================
# 2. MAYA DATA EXTRACTION
# ==========================================
def check_legit(joints, keyframes, start_frame, end_frame):
    ROM_LIMITS = {
        "leftupleg":     {"rx": (-30, 120), "ry": (-45, 45), "rz": (-45, 30)},
        "rightupleg":    {"rx": (-30, 120), "ry": (-45, 45), "rz": (-30, 45)},
        "leftleg":       {"rx": (0, 150),   "ry": (-5, 5),   "rz": (-5, 5)},
        "rightleg":      {"rx": (0, 150),   "ry": (-5, 5),   "rz": (-5, 5)},
        "leftfoot":      {"rx": (-40, 50),  "ry": (-20, 20), "rz": (-30, 30)},
        "rightfoot":     {"rx": (-40, 50),  "ry": (-20, 20), "rz": (-30, 30)},
        "spine":         {"rx": (-30, 40),  "ry": (-30, 30), "rz": (-30, 30)},
        "spine1":        {"rx": (-30, 40),  "ry": (-30, 30), "rz": (-30, 30)},
        "spine2":        {"rx": (-30, 40),  "ry": (-30, 30), "rz": (-30, 30)},
        "head":          {"rx": (-40, 40),  "ry": (-60, 60), "rz": (-40, 40)},
        "neck":          {"rx": (-30, 30),  "ry": (-50, 50), "rz": (-30, 30)},
        "leftarm":       {"rx": (-180, 60), "ry": (-50, 180),"rz": (-90, 90)},
        "rightarm":      {"rx": (-180, 60), "ry": (-180, 50),"rz": (-90, 90)},
        "leftforearm":   {"rx": (0, 160),   "ry": (-5, 5),   "rz": (-90, 90)},
        "rightforearm":  {"rx": (0, 160),   "ry": (-5, 5),   "rz": (-90, 90)},
    }
    frames = set()
    check_frames = set(range(start_frame, end_frame + 1)) - set(keyframes)
    
    for frame in check_frames:
        cmds.currentTime(frame)
        for j in joints:
            name = j.split(":")[-1].lower()
            if name in ROM_LIMITS:
                rot = cmds.getAttr(j + ".rotate")[0]
                limits = ROM_LIMITS[name]
                
                rx_min, rx_max = limits["rx"]
                ry_min, ry_max = limits["ry"]
                rz_min, rz_max = limits["rz"]
                
                if (rot[0] < rx_min or rot[0] > rx_max or
                    rot[1] < ry_min or rot[1] > ry_max or
                    rot[2] < rz_min or rot[2] > rz_max):
                    frames.add(frame)
                    break
    
    return frames

def reconstruction_error(joints, original_positions, keyframes, start_frame, end_frame):
    errors = []
    check_frames = set(range(start_frame, end_frame + 1)) - set(keyframes)

    for frame in check_frames:
        cmds.currentTime(frame)
        idx = frame - start_frame
        original = original_positions[idx]

        frame_error = 0.0
        for i, j in enumerate(joints):
            pos = cmds.xform(j, query=True, worldSpace=True, translation=True)
            orig = original[i * 3: i * 3 + 3]
            frame_error += distance_3d(pos, orig)

        avg_joint_error = frame_error / len(joints)
        errors.append(avg_joint_error)

    if errors:
        mean_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"  Reconstruction error:")
        print(f"    Mean: {mean_error:.3f} units per joint")
        print(f"    Max:  {max_error:.3f} units per joint")
        return mean_error
    return 0.0

def PCA_energy(joints, startframe, endframe):
    out = []
    for frame in range(startframe,endframe+1):
        cmds.currentTime(frame)
        tmp = []
        for j in joints:
            tmp = tmp + cmds.xform(j, query = True , worldSpace = True, translation = True)
        out.append(tmp)
    data = np.array(out)
    mean = np.mean(data,axis = 0)
    centered  = data - mean
    covariance_matrix = np.cov(centered , rowvar= False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    first_component = eigenvectors[:,-1]
    pca_curve = centered @ first_component
    return pca_curve.tolist() , out

def extract_key_poses(joints, epsilon, subsample_step=2, smooth_window=11, min_gap=5):
    start_frame = int(cmds.playbackOptions(query=True, minTime=True))
    end_frame = int(cmds.playbackOptions(query=True, maxTime=True))

    raw_energy,original_pose = PCA_energy(joints, start_frame, end_frame)
    smoothed_energy = smooth_data_moving_average(raw_energy, window_size=smooth_window)

    sampled = subsample_energy(smoothed_energy, step=subsample_step)

    points = [[i * subsample_step, val] for i, val in enumerate(sampled)]
    simplified = rdp_algorithm(points, epsilon)
    rdp_frames = set(start_frame + int(p[0]) for p in simplified)

    critical = find_critical_points(sampled, start_frame, velocity_threshold=0.5)

    critical["maxima"] = [start_frame + f * subsample_step for f in
                          range(len(sampled)) if (start_frame + f * subsample_step)
                          in set(critical["maxima"]) or False]


    critical_sub = {"maxima": [], "minima": [], "inflection": []}
    for i in range(1, len(sampled) - 1):
        d1 = (sampled[i + 1] - sampled[i - 1]) / 2.0
        if abs(d1) < 0.5:
            d2 = sampled[i + 1] - 2 * sampled[i] + sampled[i - 1]
            frame = start_frame + i * subsample_step
            if d2 < 0:
                critical_sub["maxima"].append(frame)
            elif d2 > 0:
                critical_sub["minima"].append(frame)
            else:
                critical_sub["inflection"].append(frame)

    for i in range(2, len(sampled) - 1):
        d2_curr = sampled[i + 1] - 2 * sampled[i] + sampled[i - 1]
        d2_prev = sampled[i] - 2 * sampled[i - 1] + sampled[i - 2]
        if d2_curr * d2_prev < 0:
            frame = start_frame + i * subsample_step
            if frame not in critical_sub["inflection"]:
                critical_sub["inflection"].append(frame)
    critical_sub["maxima"] = filter_minimum_gap(sorted(critical_sub["maxima"]), min_gap)
    critical_sub["minima"] = filter_minimum_gap(sorted(critical_sub["minima"]), min_gap)
    critical_sub["inflection"] = filter_minimum_gap(sorted(critical_sub["inflection"]), min_gap)

    critical_frames = set(
        critical_sub["maxima"] + critical_sub["minima"] + critical_sub["inflection"]
    )

    
    key_frames = sorted(rdp_frames | critical_frames)

    key_frames = [f for f in key_frames if start_frame <= f <= end_frame]


    if start_frame not in key_frames:
        key_frames.insert(0, start_frame)
    if end_frame not in key_frames:
        key_frames.append(end_frame)

    print(f"  Local Maxima (peak poses):  {len(critical_sub['maxima'])} frames")
    print(f"  Local Minima (rest poses):  {len(critical_sub['minima'])} frames")
    print(f"  Inflection (transitions):   {len(critical_sub['inflection'])} frames")
    print(f"  RDP key frames:             {len(rdp_frames)} frames")

    return key_frames, start_frame, end_frame , original_pos


# ==========================================
# 3. MAYA UI & EXECUTION
# ==========================================

def run_extraction_tool(*args):
    joints = cmds.ls(selection=True)
    if not joints:
        cmds.warning("Please select at least one joint or controller!")
        return

    epsilon_val = cmds.floatSliderGrp("rdp_epsilon_slider", query=True, value=True)
    subsample_val = int(cmds.intSliderGrp("subsample_slider", query=True, value=True))
    smooth_val = int(cmds.intSliderGrp("smooth_slider", query=True, value=True))
    gap_val = int(cmds.intSliderGrp("gap_slider", query=True, value=True))

    print("Analyzing motion curve (PCA)...")
    key_frames, start_f, end_f , original_pose = extract_key_poses(
        joints, epsilon_val,
        subsample_step=subsample_val,
        smooth_window=smooth_val,
        min_gap=gap_val
    )

    cmds.undoInfo(openChunk=True)
    try:
        all_frames = set(range(start_f, end_f + 1))
        for frame in all_frames - set(key_frames):
            cmds.cutKey(joints, time=(frame, frame), clear=True)

        # ROM validation — after deletion, check interpolated frames
        print("  Validating ROM constraints...")
        violations = check_legit(joints, key_frames, start_f, end_f)
        if violations:
            print(f"  Found {len(violations)} ROM violations — re-inserting keys...")
            for frame in violations:
                cmds.setKeyframe(joints, time=frame)
            key_frames = sorted(set(key_frames) | set(violations))
        else:
            print("  No ROM violations found.")
        reconstruction_error(joints, original_pose, key_frames, start_f, end_f)
    finally:
        cmds.undoInfo(closeChunk=True)

    total = end_f - start_f + 1
    print(f"Done! Kept {len(key_frames)}/{total} key poses ({100*len(key_frames)/total:.1f}%).")


def show_ui():
    window_name = "PoseExtractorUI"
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)

    cmds.window(window_name, title="Mocap Pose Extractor v2", widthHeight=(380, 280))
    cmds.columnLayout(adjustableColumn=True, rowSpacing=8, columnAttach=("both", 10))

    cmds.text(
        label="Select 5-10 main joints (hips, spine, chest, head, wrists, ankles).",
        wordWrap=True, align="center",
    )

    cmds.separator(height=8, style="none")

    cmds.floatSliderGrp(
        "rdp_epsilon_slider", label="Epsilon (RDP):",
        field=True, minValue=0.1, maxValue=5.0, value=1.2, step=0.1,
    )

    cmds.intSliderGrp(
        "subsample_slider", label="Subsample (step):",
        field=True, minValue=1, maxValue=4, value=2, step=1,
    )

    cmds.intSliderGrp(
        "smooth_slider", label="Smooth window:",
        field=True, minValue=3, maxValue=25, value=11, step=2,
    )

    cmds.intSliderGrp(
        "gap_slider", label="Min gap (frames):",
        field=True, minValue=1, maxValue=15, value=5, step=1,
    )

    cmds.separator(height=8, style="none")

    cmds.button(
        label="Extract Key Poses",
        command=run_extraction_tool, height=40, backgroundColor=(0.2, 0.6, 0.3),
    )

    cmds.text(
        label="Tip: For robot dance — epsilon 1.2, subsample 2, smooth 11, gap 5",
        wordWrap=True, align="center", font="smallObliqueLabelFont",
    )

    cmds.showWindow(window_name)


show_ui()
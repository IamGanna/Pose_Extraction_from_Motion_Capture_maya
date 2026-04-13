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

def calculate_motion_energy(joints, start_frame, end_frame):
    energy_curve = []
    prev_positions = []

    cmds.currentTime(start_frame)
    for j in joints:
        prev_positions.append(
            cmds.xform(j, query=True, worldSpace=True, translation=True)
        )
    energy_curve.append(0.0)

    for frame in range(start_frame + 1, end_frame + 1):
        cmds.currentTime(frame)
        current_energy = 0.0
        current_positions = []

        for i, j in enumerate(joints):
            pos = cmds.xform(j, query=True, worldSpace=True, translation=True)
            current_positions.append(pos)
            current_energy += distance_3d(pos, prev_positions[i])

        energy_curve.append(current_energy)
        prev_positions = current_positions

    return energy_curve
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
    return pca_curve.tolist()

def extract_key_poses(joints, epsilon, subsample_step=2, smooth_window=11, min_gap=5):
    start_frame = int(cmds.playbackOptions(query=True, minTime=True))
    end_frame = int(cmds.playbackOptions(query=True, maxTime=True))

    raw_energy = PCA_energy(joints, start_frame, end_frame)
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

    return key_frames, start_frame, end_frame


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

    print("Analyzing motion curve...")
    key_frames, start_f, end_f = extract_key_poses(
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
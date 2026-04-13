import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter # <-- NEW: For smoothing mocap noise

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Joint:
    name: str
    parent: int
    offset: np.ndarray
    channels: list
    channel_start: int
    children: list

# -----------------------------
# Basic transforms
# -----------------------------
def rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],
                     [0,c,-s,0],
                     [0,s,c,0],
                     [0,0,0,1]], float)

def rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],
                     [0,1,0,0],
                     [-s,0,c,0],
                     [0,0,0,1]], float)

def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],
                     [s,c,0,0],
                     [0,0,1,0],
                     [0,0,0,1]], float)

def trans(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T

# -----------------------------
# BVH parser
# -----------------------------
def parse_bvh(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tokens = []
        for ln in f:
            ln = ln.strip()
            if ln:
                tokens.extend(ln.split())

    i = 0
    joints = []
    stack = []
    channel_cursor = 0

    def add_joint(name, parent):
        nonlocal channel_cursor
        j = Joint(
            name=name,
            parent=parent,
            offset=np.zeros(3, float),
            channels=[],
            channel_start=channel_cursor,
            children=[]
        )
        joints.append(j)
        return len(joints) - 1

    if tokens[i] != "HIERARCHY":
        raise ValueError("Not a BVH: missing HIERARCHY")
    i += 1
    if tokens[i] != "ROOT":
        raise ValueError("Not a BVH: missing ROOT")
    i += 1

    root_name = tokens[i]; i += 1
    root_idx = add_joint(root_name, -1)
    stack.append(root_idx)

    while i < len(tokens) and tokens[i] != "MOTION":
        t = tokens[i]; i += 1

        if t == "{": continue
        if t == "}":
            stack.pop()
            continue

        if t in ("JOINT", "ROOT"):
            name = tokens[i]; i += 1
            parent = stack[-1] if stack else -1
            idx = add_joint(name, parent)
            joints[parent].children.append(idx)
            stack.append(idx)
            continue

        if t == "End":
            if i < len(tokens) and tokens[i] == "Site": i += 1
            depth = 0
            while i < len(tokens):
                if tokens[i] == "{": depth += 1
                elif tokens[i] == "}":
                    depth -= 1
                    if depth <= 0:
                        i += 1
                        break
                i += 1
            continue

        if t == "OFFSET":
            x, y, z = float(tokens[i]), float(tokens[i+1]), float(tokens[i+2])
            i += 3
            joints[stack[-1]].offset = np.array([x, y, z], float)
            continue

        if t == "CHANNELS":
            n = int(tokens[i]); i += 1
            ch = tokens[i:i+n]; i += n
            joints[stack[-1]].channels = ch
            joints[stack[-1]].channel_start = channel_cursor
            channel_cursor += n
            continue

    if i >= len(tokens) or tokens[i] != "MOTION":
        raise ValueError("Missing MOTION section")
    i += 1

    if tokens[i] != "Frames:": raise ValueError("Missing Frames:")
    i += 1
    num_frames = int(tokens[i]); i += 1

    if tokens[i] != "Frame" or tokens[i+1] != "Time:":
        raise ValueError("Missing Frame Time")
    i += 2
    frame_time = float(tokens[i]); i += 1

    total_channels = channel_cursor
    data = []
    for _ in range(num_frames):
        frame_vals = list(map(float, tokens[i:i+total_channels]))
        i += total_channels
        data.append(frame_vals)

    motion = np.array(data, dtype=float)
    return joints, motion, frame_time

# -----------------------------
# FK: Local transform (dataset-aware)
# -----------------------------
def local_transform(joint: Joint, frame_row: np.ndarray):
    start = joint.channel_start
    ch = joint.channels
    pos = np.zeros(3, float)
    R = np.eye(4)
    has_pos = any(c.endswith("position") for c in ch)

    for k, name in enumerate(ch):
        v = frame_row[start + k]
        if name == "Xposition": pos[0] += v
        elif name == "Yposition": pos[1] += v
        elif name == "Zposition": pos[2] += v
        elif name == "Xrotation": R = R @ rot_x(v)
        elif name == "Yrotation": R = R @ rot_y(v)
        elif name == "Zrotation": R = R @ rot_z(v)

    if has_pos:
        t = pos if joint.parent != -1 else (joint.offset + pos)
    else:
        t = joint.offset

    return trans(t) @ R

def fk_world_positions(joints, motion, frame_idx):
    frame = motion[frame_idx]
    world = [np.eye(4) for _ in joints]

    for idx, j in enumerate(joints):
        L = local_transform(j, frame)
        if j.parent == -1:
            world[idx] = L
        else:
            world[idx] = world[j.parent] @ L

    return np.array([w[:3, 3] for w in world], float)

# -----------------------------
# Core logic: Energy & NEW RDP
# -----------------------------
def get_edges(joints):
    return [(j.parent, i) for i, j in enumerate(joints) if j.parent != -1]

def remap(pos):
    return pos[:, [0, 2, 1]]

def find_joint(joints, name):
    for i, j in enumerate(joints):
        if j.name == name: return i
    return None

def energy_curve(joints, motion, targets):
    T = len(motion)
    E = np.zeros(T)
    prev = fk_world_positions(joints, motion, 0)
    for t in range(1, T):
        cur = fk_world_positions(joints, motion, t)
        diff = cur[targets] - prev[targets]
        E[t] = np.linalg.norm(diff, axis=1).sum()
        prev = cur
    return E

def perpendicular_distance(pt, line_start, line_end):
    """Calculates 2D distance from a point to a line segment for RDP."""
    if np.all(line_start == line_end):
        return np.linalg.norm(pt - line_start)
    return np.abs(np.cross(line_end - line_start, line_start - pt)) / np.linalg.norm(line_end - line_start)

def rdp(points, epsilon):
    """Recursive Ramer-Douglas-Peucker algorithm."""
    dmax = 0.0
    index = 0
    end = len(points) - 1
    
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
            
    if dmax > epsilon:
        left_results = rdp(points[:index+1], epsilon)
        right_results = rdp(points[index:], epsilon)
        return np.vstack((left_results[:-1], right_results))
    else:
        return np.vstack((points[0], points[end]))

def extract_keys_rdp(E, epsilon=0.5, smooth_window=15):
    """Smooths the curve and applies RDP to extract keyframes."""
    # 1. Smooth the Mocap Noise
    if len(E) > smooth_window:
        # polyorder 3 is standard for keeping curve shape
        smoothed_E = savgol_filter(E, window_length=smooth_window, polyorder=3)
    else:
        smoothed_E = E
        
    # 2. Prepare 2D points for RDP: [frame_index, energy_value]
    frames = np.arange(len(smoothed_E))
    points = np.column_stack((frames, smoothed_E))
    
    # 3. Run RDP
    simplified_points = rdp(points, epsilon)
    
    # 4. Extract just the frame numbers
    key_frames = simplified_points[:, 0].astype(int).tolist()
    return key_frames

def enforce_gap(keys, min_gap=5):
    out = []
    last = -999
    for k in sorted(keys):
        if k - last >= min_gap:
            out.append(k)
            last = k
    return out

# -----------------------------
# Animation & Error Helpers
# -----------------------------
def animate_keyframes(joints, motion, dt, keys, stride=1):
    edges = get_edges(joints)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-90)

    pos0 = remap(fk_world_positions(joints, motion, keys[0]))
    lines = []
    for (a, b) in edges:
        ln, = ax.plot([pos0[a,0], pos0[b,0]],
                      [pos0[a,1], pos0[b,1]],
                      [pos0[a,2], pos0[b,2]])
        lines.append(ln)

    mins = pos0.min(axis=0)
    maxs = pos0.max(axis=0)
    radius = np.max(maxs - mins) / 2 + 1e-6

    def update(i):
        frame_idx = keys[i % len(keys)]
        pos = remap(fk_world_positions(joints, motion, frame_idx))
        root = pos[0]
        ax.set_xlim(root[0]-radius, root[0]+radius)
        ax.set_ylim(root[1]-radius, root[1]+radius)
        ax.set_zlim(root[2]-radius, root[2]+radius)

        for k, (a, b) in enumerate(edges):
            lines[k].set_data([pos[a,0], pos[b,0]], [pos[a,1], pos[b,1]])
            lines[k].set_3d_properties([pos[a,2], pos[b,2]])

        ax.set_title(f"Key {i+1}/{len(keys)}  (Frame {frame_idx})")
        return lines

    anim = FuncAnimation(fig, update, frames=len(keys),
                         interval=dt*1000*stride, blit=False)
    plt.show()
    return anim

def all_positions(joints, motion):
    return np.stack([remap(fk_world_positions(joints, motion, t)) for t in range(len(motion))], axis=0)

def reconstruct_positions(gt_pos, keys):
    T, J, _ = gt_pos.shape
    recon = np.zeros_like(gt_pos)
    for a, b in zip(keys[:-1], keys[1:]):
        Pa = gt_pos[a]
        Pb = gt_pos[b]
        span = b - a
        for t in range(a, b+1):
            u = 0.0 if span == 0 else (t - a) / span
            recon[t] = (1-u) * Pa + u * Pb
    return recon

def rms_error(gt_pos, recon_pos, idxs):
    diff = gt_pos[:, idxs] - recon_pos[:, idxs]
    per_frame = np.linalg.norm(diff, axis=2).mean(axis=1)
    return float(np.sqrt(np.mean(per_frame**2))), per_frame

def animate_bvh(joints, motion, dt, stride=1):
    edges = get_edges(joints)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=15, azim=-90)

    pos0 = remap(fk_world_positions(joints, motion, 0))
    lines = []
    for (a, b) in edges:
        ln, = ax.plot([pos0[a,0], pos0[b,0]],
                      [pos0[a,1], pos0[b,1]],
                      [pos0[a,2], pos0[b,2]])
        lines.append(ln)

    mins = pos0.min(axis=0)
    maxs = pos0.max(axis=0)
    radius = np.max(maxs - mins) / 2 + 1e-6

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.line.set_color((0,0,0,0)); ax.yaxis.line.set_color((0,0,0,0)); ax.zaxis.line.set_color((0,0,0,0))

    def update(frame):
        idx = (frame * stride) % len(motion)
        pos = remap(fk_world_positions(joints, motion, idx))
        root = pos[0]
        ax.set_xlim(root[0]-radius, root[0]+radius)
        ax.set_ylim(root[1]-radius, root[1]+radius)
        ax.set_zlim(root[2]-radius, root[2]+radius)

        for k, (a, b) in enumerate(edges):
            lines[k].set_data([pos[a,0], pos[b,0]], [pos[a,1], pos[b,1]])
            lines[k].set_3d_properties([pos[a,2], pos[b,2]])
        ax.set_title(f"Frame {idx}")
        return lines

    frames = max(1, len(motion) // stride)
    anim = FuncAnimation(fig, update, frames=frames,
                         interval=dt*1000*stride, blit=False)
    plt.show()
    return anim

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Update this path to your actual Bandai dataset path
    path = "C:\dev\Term project monk studio\Bandai-Namco-Research-Motiondataset\dataset\Bandai-Namco-Research-Motiondataset-1\data\dataset-1_dance-long_normal_001.bvh"
    
    try:
        joints, motion, dt = parse_bvh(path)
    except FileNotFoundError:
        print(f"Error: Could not find BVH file at {path}. Please check the path.")
        exit()

    print(f"Loaded BVH: {len(joints)} joints, {motion.shape[0]} frames.")

    targets = [
        find_joint(joints, "Hips"),
        find_joint(joints, "Chest"),
        find_joint(joints, "Head"),
        find_joint(joints, "Hand_L"),
        find_joint(joints, "Hand_R"),
        find_joint(joints, "Foot_L"),
        find_joint(joints, "Foot_R"),
    ]
    # Filter out any None values if joint names differ slightly
    targets = [t for t in targets if t is not None]

    # Calculate Global Energy Curve
    E = energy_curve(joints, motion, targets)
    
    # ---------------------------------------------------------
    # 🔥 THE UPGRADE: RDP Extraction 
    # Tweak 'epsilon' to change how many keyframes you get.
    # Higher epsilon (e.g., 2.0) = fewer keyframes.
    # Lower epsilon (e.g., 0.2) = more keyframes.
    # ---------------------------------------------------------
    raw_keys = extract_keys_rdp(E, epsilon=0.55, smooth_window=15)
    keys = enforce_gap(raw_keys, min_gap=5)
    
    # Always include the very first and very last frame
    keys = sorted(set([0] + keys + [len(motion)-1]))

    print(f"\nExtracted {len(keys)} Keyframes using RDP:")
    print(keys)

    # Calculate Errors
    gt = all_positions(joints, motion)
    recon = reconstruct_positions(gt, keys)
    
    important = [find_joint(joints, n) for n in ["Head","Hand_L","Hand_R","Foot_L","Foot_R"]]
    important = [i for i in important if i is not None]
    
    err_rms, err_curve = rms_error(gt, recon, important)
    print(f"\nRMS Error (end-effectors): {err_rms:.4f}")
    print(f"Max Frame Error: {float(err_curve.max()):.4f}")

    # Visualize
    # Comment out animate_bvh if you only want to see the keyframes
    # anim = animate_bvh(joints, motion, dt, stride=2) 
    anim_keys = animate_keyframes(joints, motion, dt, keys, stride=4)
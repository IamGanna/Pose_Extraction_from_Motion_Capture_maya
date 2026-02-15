import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

        if t == "{":
            continue
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
            # Skip End Site block
            if i < len(tokens) and tokens[i] == "Site":
                i += 1
            depth = 0
            while i < len(tokens):
                if tokens[i] == "{":
                    depth += 1
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

    if tokens[i] != "Frames:":
        raise ValueError("Missing Frames:")
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
        if name == "Xposition":
            pos[0] += v
        elif name == "Yposition":
            pos[1] += v
        elif name == "Zposition":
            pos[2] += v
        elif name == "Xrotation":
            R = R @ rot_x(v)
        elif name == "Yrotation":
            R = R @ rot_y(v)
        elif name == "Zrotation":
            R = R @ rot_z(v)

    # Key rule:
    # If joint has explicit position channels, DO NOT also add OFFSET (avoid double translation).
    # If it doesn't, use OFFSET.
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
# Animation helpers
# -----------------------------
def get_edges(joints):
    return [(j.parent, i) for i, j in enumerate(joints) if j.parent != -1]

def remap(pos):
    # (x, y, z) -> (x, z, y) for nicer "up" in matplotlib
    return pos[:, [0, 2, 1]]

def find_joint(joints, name):
    for i, j in enumerate(joints):
        if j.name == name:
            return i
            
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
def local_minima(E):
    keys = []
    for t in range(1, len(E)-1):
        if E[t] < E[t-1] and E[t] < E[t+1]:
            keys.append(t)
    return keys
def enforce_gap(keys, min_gap=5):
    out = []
    last = -999
    for k in sorted(keys):
        if k - last >= min_gap:
            out.append(k)
            last = k
    return out


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

    # radius based on first frame size
    mins = pos0.min(axis=0)
    maxs = pos0.max(axis=0)
    radius = np.max(maxs - mins) / 2 + 1e-6

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.line.set_color((0,0,0,0))
    ax.yaxis.line.set_color((0,0,0,0))
    ax.zaxis.line.set_color((0,0,0,0))



    def update(frame):
        idx = (frame * stride) % len(motion)
        pos = remap(fk_world_positions(joints, motion, idx))

        # follow root camera
        root = pos[0]
        ax.set_xlim(root[0]-radius, root[0]+radius)
        ax.set_ylim(root[1]-radius, root[1]+radius)
        ax.set_zlim(root[2]-radius, root[2]+radius)

        for k, (a, b) in enumerate(edges):
            lines[k].set_data([pos[a,0], pos[b,0]],
                              [pos[a,1], pos[b,1]])
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
    path = "Bandai-Namco-Research-Motiondataset/dataset/Bandai-Namco-Research-Motiondataset-1/data/dataset-1_walk_active_001.bvh"
    joints, motion, dt = parse_bvh(path)

    print("joints:", len(joints))
    print("frames:", motion.shape[0], "channels:", motion.shape[1], "dt:", dt)

    # sanity: does anything move?
    p0 = fk_world_positions(joints, motion, 0)
    p10 = fk_world_positions(joints, motion, 10)
    print("max joint displacement frame0->10:", np.max(np.linalg.norm(p10 - p0, axis=1)))

    # check feet motion
    footL = find_joint(joints, "Foot_L")
    footR = find_joint(joints, "Foot_R")
    if footL is not None:
        print("Foot_L move:", np.linalg.norm(p10[footL] - p0[footL]))
    if footR is not None:
        print("Foot_R move:", np.linalg.norm(p10[footR] - p0[footR]))
    targets = [
    find_joint(joints, "Hips"),
    find_joint(joints, "Chest"),
    find_joint(joints, "Head"),
    find_joint(joints, "Hand_L"),
    find_joint(joints, "Hand_R"),
    find_joint(joints, "Foot_L"),
    find_joint(joints, "Foot_R"),
    ]


    anim = animate_bvh(joints, motion, dt, stride=1)
    E = energy_curve(joints, motion, targets)
    keys = enforce_gap(local_minima(E), min_gap=5)

    print("Keyframes:", keys)
    print("Number of keys:", len(keys))

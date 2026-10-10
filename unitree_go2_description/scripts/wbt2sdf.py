#!/usr/bin/env python3
"""Convert the IEEE SMCS 2026 SAR competition Webots worlds (.wbt) to Gazebo SDF.

Usage:
  wbt2sdf.py <world.wbt> [<world.wbt> ...] --out <worlds_dir> [--ceilings]

What is converted exactly:
  Wall, Window, Door, Floor, Table, Cabinet, Radiator, StraightStairs,
  primitive Shapes (Box/Cylinder/Sphere/Capsule/Plane) inside Solid/Pose/Group,
  Victim (STL mesh visual, box collision), OriginMarker (textured 1x1 plane).
Every other Webots object PROTO is approximated by its Webots bounding object
(a box or cylinder), so the navigation footprint is preserved while the visual
detail is not.  Ceilings (planes higher than 2 m) are dropped unless --ceilings
is given, so the world stays visible from above in the Gazebo GUI.

Assets (victim STL meshes and the origin-marker texture) are expected in
<out>/sar_assets/{meshes,textures}; URIs in the SDF are relative to the world file.
"""
import argparse
import math
import os
import re
import sys

import numpy as np

# --------------------------------------------------------------------------- parser

TOKEN_RE = re.compile(r'"(?:[^"\\]|\\.)*"|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|[{}\[\]]')


def tokenize(text):
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith('EXTERNPROTO') or line.lstrip().startswith('hidden '):
            continue
        # strip comments (a '#' outside a string)
        in_str = False
        clean = []
        i = 0
        while i < len(line):
            c = line[i]
            if c == '"' and (i == 0 or line[i - 1] != '\\'):
                in_str = not in_str
            if c == '#' and not in_str:
                break
            clean.append(c)
            i += 1
        out.extend(TOKEN_RE.findall(''.join(clean)))
    return out


class Node:
    def __init__(self, type_, def_name=None):
        self.type = type_
        self.def_name = def_name
        self.fields = {}

    def get(self, name, default=None):
        return self.fields.get(name, default)

    def vec(self, name, default):
        v = self.fields.get(name)
        if isinstance(v, list) and v and all(isinstance(x, float) for x in v):
            return v
        return list(default)

    def num(self, name, default):
        v = self.fields.get(name)
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], float):
            return v[0]
        return default

    def __repr__(self):
        return f'{self.type}({self.def_name}) {list(self.fields)}'


class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0
        self.defs = {}

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    @staticmethod
    def is_num(tok):
        return tok is not None and re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', tok) is not None

    def parse_file(self):
        nodes = []
        while self.peek() is not None:
            nodes.append(self.parse_node())
        return nodes

    def parse_node(self):
        tok = self.next()
        if tok == 'USE':
            return self.defs.get(self.next())
        if tok == 'NULL':
            return None
        def_name = None
        if tok == 'DEF':
            def_name = self.next()
            tok = self.next()
        node = Node(tok, def_name)
        if def_name:
            self.defs[def_name] = node
        assert self.next() == '{', f'expected {{ after {tok}'
        while self.peek() != '}':
            fname = self.next()
            node.fields[fname] = self.parse_value()
        self.next()  # }
        return node

    def parse_value(self):
        tok = self.peek()
        if tok == '[':
            self.next()
            items = []
            while self.peek() != ']':
                items.append(self.parse_value_item())
            self.next()
            # flatten numeric lists ([0.1, 0.2] is parsed as [[0.1], [0.2]])
            if items and all(isinstance(x, list) and len(x) == 1 and isinstance(x[0], float) for x in items):
                return [x[0] for x in items]
            return items
        return self.parse_value_item(single_field=True)

    def parse_value_item(self, single_field=False):
        tok = self.peek()
        if tok.startswith('"'):
            self.next()
            return tok[1:-1]
        if self.is_num(tok):
            nums = []
            while self.is_num(self.peek()):
                nums.append(float(self.next()))
                if not single_field:
                    break
            return nums
        if tok in ('TRUE', 'FALSE'):
            self.next()
            return tok == 'TRUE'
        return self.parse_node()


# --------------------------------------------------------------------------- math

def axis_angle(v):
    x, y, z, a = v
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def compose(pose_a, pose_b):
    """pose_a * pose_b, each pose = (R, t)."""
    Ra, ta = pose_a
    Rb, tb = pose_b
    return (Ra @ Rb, Ra @ tb + ta)


def inverse(pose):
    R, t = pose
    return (R.T, -R.T @ t)


IDENT = (np.eye(3), np.zeros(3))


def node_pose(node):
    t = np.array(node.vec('translation', (0, 0, 0)))
    R = axis_angle(node.vec('rotation', (0, 0, 1, 0)))
    return (R, t)


def local(t=(0, 0, 0), R=None):
    return (np.eye(3) if R is None else R, np.array(t, dtype=float))


def rpy(R):
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(abs(sy) - 1) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def pose_str(pose):
    R, t = pose
    r, p, y = rpy(R)
    return ' '.join(f'{v:.5f}' for v in (t[0], t[1], t[2], r, p, y))


# --------------------------------------------------------------------------- tables

SKIP = {'WorldInfo', 'Viewpoint', 'TexturedBackground', 'Rosbot', 'Robot', 'CeilingLight',
        'CeilingSpotLight', 'DoubleFluorescentLamp', 'Washer', 'CapScrew', 'EyeScrew',
        'ElectricalPlug', 'Connector', 'Receiver', 'Emitter'}

# type: (cx, cy, cz, sx, sy, sz)  local box approximation of the Webots bounding object
BOX = {
    'Armchair': (0, 0, 0.35, 1.0, 1.1, 0.7),
    'Sofa': (0, 0, 0.35, 1.0, 1.8, 0.7),
    'Bed': (0, 0, 0.63, 2.12, 1.75, 1.26),
    'Chair': (-0.01, 0, 0.5, 0.52, 0.5, 1.0),
    'WoodenChair': (-0.01, 0, 0.475, 0.42, 0.4, 0.95),
    'OfficeChair': (-0.02, 0, 0.485, 0.5, 0.5, 0.97),
    'Desk': (0, 0, 0.355, 0.7, 1.2, 0.71),
    'Fridge': (0, 0, 0.9, 0.7, 0.7, 1.8),
    'Oven': (0, 0, 0.34, 0.5, 0.44, 0.68),
    'Sink': (0.1, 0, 0.07, 0.37, 0.44, 0.22),
    'Toilet': (-0.065, 0, 0.51, 0.73, 0.6, 1.02),
    'Bathtube': (-0.03, 0, 0.245, 0.93, 1.9, 0.49),
    'BathroomSink': (0.2, 0, 0.14, 0.44, 0.7, 0.37),
    'LandscapePainting': (0, 0, 0, 0.05, 1.5, 1.0),
    'PortraitPainting': (0, 0, 0, 0.05, 0.82, 1.01),
    'TrashBin': (0, 0, 0.58, 0.8, 0.7, 1.16),
    'PlasticFruitBox': (0, 0, 0.135, 1.0, 0.612, 0.27),
    'TrafficCone': (0, 0, 0.255, 0.5, 0.5, 0.51),
    'WorkBarrier': (0, 0, 0.5, 0.6, 1.2, 1.0),
    'PlatformCart': (0, 0, 0.5, 0.9, 0.5, 1.0),
    'Forklift': (0.2, 0, 0.09, 2.5, 1.0, 1.76),
    'Monitor': (0, 0, 0.2, 0.2, 0.47, 0.4),
    'DesktopComputer': (0, 0, 0.18, 0.36, 0.36, 0.36),
    'Keyboard': (0, 0, 0.005, 0.41, 0.14, 0.01),
    'ComputerMouse': (0, 0, 0.015, 0.082, 0.045, 0.03),
    'OfficeTelephone': (-0.08, 0, 0.06, 0.2, 0.18, 0.12),
    'EmergencyExitSign': (0, 0, -0.04, 0.38, 0.17, 0.081),
    'Book': (0, 0, 0, 0.02, 0.15, 0.2),
    'Clock': (0, 0, 0, 0.05, 0.33, 0.33),
    'FruitBowl': (0, 0, 0.075, 0.3, 0.3, 0.15),
    'Cookware': (0, 0, 0.05, 0.3, 0.17, 0.1),
    'Lid': (0, 0, 0.01, 0.184, 0.184, 0.02),
    'WoodenSpoon': (0, 0, 0.003, 0.28, 0.044, 0.006),
    'Fork': (0, 0, 0.006, 0.19, 0.024, 0.012),
    'Knife': (0, 0, 0.003, 0.22, 0.017, 0.006),
    'Hammer': (0.05, 0, 0, 0.27, 0.065, 0.027),
    'Wrench': (0, 0, 0.003, 0.19, 0.035, 0.005),
    'Screwdriver': (0, 0, 0, 0.21, 0.024, 0.024),
    'PaperBoat': (0, 0, 0.045, 0.34, 0.07, 0.09),
    'RubberDuck': (0, 0, 0.03, 0.1, 0.06, 0.09),
    'LeverValve': (0, 0, 0.15, 0.45, 0.29, 0.5),
    'LJoint': (0, 0, 0, 0.15, 0.08, 0.15),
    'TexturedBoxShape': (0, 0, 0, 0.1, 0.1, 0.1),
    'Barbecue': (0, 0, 0.62, 0.8, 0.9, 1.24),
}
VISUAL_ONLY = {'Carpet': (0, 0, 0.01, 2.4, 1.6, 0.02), 'SquareManhole': None}

# type: (cx, cy, cz, radius, length)
CYL = {
    'FireExtinguisher': (0, 0, 0.4, 0.1, 0.8),
    'FloorLight': (0, 0, 0.86, 0.15, 1.72),
    'PaintBucket': (0, 0, 0.078, 0.092, 0.156),
    'Glass': (0, 0, 0.029, 0.026, 0.056),
    'Wineglass': (0, 0, 0.065, 0.035, 0.13),
    'Can': (0, 0, 0.061, 0.032, 0.122),
    'BeerBottle': (0, 0, 0.115, 0.031, 0.23),
    'WaterBottle': (0, 0, 0.153, 0.044, 0.306),
    'JamJar': (0, 0, 0.058, 0.045, 0.115),
    'HoneyJar': (0, 0, 0.058, 0.045, 0.115),
    'Carafe': (0.036, 0, 0.09, 0.066, 0.18),
    'BunchOfSunFlowers': (0, 0, 0.475, 0.17, 0.95),
}
SPHERE = {'Apple': (0, 0, 0, 0.05), 'Orange': (0, 0, 0, 0.05)}

COLORS = {
    'Wall': (0.85, 0.85, 0.80), 'Door': (0.45, 0.30, 0.18), 'Floor': (0.55, 0.50, 0.45),
    'Table': (0.55, 0.40, 0.25), 'RoundTable': (0.55, 0.40, 0.25), 'Desk': (0.55, 0.40, 0.25),
    'WoodenChair': (0.55, 0.40, 0.25), 'Chair': (0.4, 0.3, 0.2), 'OfficeChair': (0.2, 0.2, 0.2),
    'Cabinet': (0.92, 0.92, 0.88), 'Fridge': (0.92, 0.92, 0.92), 'Oven': (0.3, 0.3, 0.3),
    'Sink': (0.8, 0.8, 0.85), 'BathroomSink': (0.95, 0.95, 0.95), 'Toilet': (0.95, 0.95, 0.95),
    'Bathtube': (0.95, 0.95, 0.95), 'Bed': (0.6, 0.6, 0.8), 'Sofa': (0.3, 0.3, 0.35),
    'Armchair': (0.3, 0.3, 0.35), 'Radiator': (0.9, 0.9, 0.9), 'Victim': (0.9, 0.45, 0.2),
    'OilBarrel': (0.2, 0.3, 0.6), 'WoodenBox': (0.6, 0.45, 0.25), 'WoodenPallet': (0.65, 0.5, 0.3),
    'WoodenPalletStack': (0.65, 0.5, 0.3), 'CardboardBox': (0.75, 0.6, 0.4),
    'PlasticCrate': (0.2, 0.5, 0.8), 'PlasticFruitBox': (0.2, 0.6, 0.3),
    'FireExtinguisher': (0.8, 0.1, 0.1), 'TrafficCone': (1.0, 0.5, 0.0), 'WorkBarrier': (1.0, 0.6, 0.0),
    'Forklift': (0.9, 0.6, 0.1), 'TruckSimple': (0.3, 0.3, 0.35), 'ConveyorBelt': (0.3, 0.3, 0.3),
    'StraightStairs': (0.7, 0.7, 0.7), 'StraightStairsRail': (0.6, 0.6, 0.6), 'TrashBin': (0.2, 0.4, 0.2),
    'PaintBucket': (0.8, 0.8, 0.8), 'Carpet': (0.5, 0.2, 0.2), 'SquareManhole': (0.35, 0.35, 0.35),
    'Worktop': (0.4, 0.4, 0.4), 'Monitor': (0.1, 0.1, 0.1), 'DesktopComputer': (0.2, 0.2, 0.2),
}
APPEARANCE_COLORS = {
    'Roughcast': (0.85, 0.85, 0.82), 'Parquetry': (0.6, 0.42, 0.25), 'Pavement': (0.5, 0.5, 0.5),
    'CementTiles': (0.62, 0.62, 0.6), 'PorcelainChevronTiles': (0.8, 0.8, 0.75), 'Plaster': (0.9, 0.9, 0.88),
    'BrushedAluminium': (0.7, 0.7, 0.72), 'BakelitePlastic': (0.45, 0.3, 0.2), 'Asphalt': (0.25, 0.25, 0.25),
    'RoughConcrete': (0.55, 0.55, 0.55), 'CorrugatedMetal': (0.6, 0.6, 0.62), 'GalvanizedMetal': (0.65, 0.65, 0.67),
    'OsbWood': (0.7, 0.6, 0.4), 'ThreadMetalPlate': (0.5, 0.5, 0.5), 'MetalPipePaint': (0.4, 0.4, 0.4),
    'PaintedWood': (0.9, 0.9, 0.85), 'Carpet': (0.5, 0.2, 0.2), 'BlanketFabric': (0.5, 0.5, 0.7),
    'PerforatedMetal': (0.55, 0.55, 0.55), 'ElectricConduit': (0.6, 0.6, 0.6), 'FlexibleAluminiumDuct': (0.7, 0.7, 0.7),
}
DEFAULT_COLOR = (0.6, 0.6, 0.6)

VICTIM_MARKER_OFFSETS = {
    'man_1': (0, -0.4, 0), 'man_2': (0, -1.3, 0), 'man_3': (0, 1.3, 0),
    'boy_1': (0, -0.2, 0), 'boy_2': (0, 0.7, 0), 'boy_3': (0, -0.2, 0),
    'woman_1': (0, 0.9, 0), 'woman_2': (0.2, 0.7, 0), 'woman_3': (0, -0.3, 0),
    'girl_1': (0, 0.7, 0), 'girl_2': (-0.1, 0.4, 0), 'girl_3': (0, -0.2, 0),
}
ROBOT_STARTS = {'robot1': (-0.375, 0.375, 0.0), 'robot2': (-0.375, 0.0, 0.0)}

CEILING_Z = 2.0

# Floor slabs are drawn 2mm thick at z=0.0015, so their top face sits at 0.0025.
# Lift the origin marker clear of that, or the floor renders over it.
ORIGIN_MARKER_Z = 0.004


# --------------------------------------------------------------------------- conversion

class Prim:
    def __init__(self, kind, pose, color=DEFAULT_COLOR, collide=True, alpha=1.0, **geo):
        self.kind = kind      # box | cylinder | sphere | mesh
        self.pose = pose      # relative to the model
        self.color = color
        self.alpha = alpha
        self.collide = collide
        self.geo = geo


class Model:
    def __init__(self, name, pose, wtype):
        self.name = name
        self.pose = pose
        self.wtype = wtype
        self.prims = []


class Converter:
    def __init__(self, ceilings=False):
        self.ceilings = ceilings
        self.models = []
        self.names = {}
        self.unknown = {}
        self.skipped_ceilings = 0
        self.victims = []      # (name, model, world pose, marker world pos)
        self.origin = None     # world pose of the origin marker
        self.robots = []       # (name, world pos)
        self.notes = []

    def unique(self, base):
        base = re.sub(r'[^A-Za-z0-9_]+', '_', base).strip('_') or 'model'
        n = self.names.get(base, 0)
        self.names[base] = n + 1
        return base if n == 0 else f'{base}_{n}'

    def color_of(self, node):
        c = node.get('color')
        if isinstance(c, list) and len(c) == 3:
            return tuple(c)
        for key in ('appearance', 'doorAppearance', 'primaryAppearance', 'trayAppearance', 'stepAppearance'):
            ap = node.get(key)
            if isinstance(ap, Node):
                bc = ap.get('baseColor')
                if isinstance(bc, list) and len(bc) == 3:
                    return tuple(bc)
                if ap.type in APPEARANCE_COLORS:
                    return APPEARANCE_COLORS[ap.type]
        return COLORS.get(node.type, DEFAULT_COLOR)

    # ---- entry
    def convert(self, nodes):
        for n in nodes:
            if n is None:
                continue
            self.top_level(n)

    def top_level(self, node):
        if node.type in SKIP:
            return
        if node.type == 'Robot' or node.get('controller') is not None and node.type != 'Victim':
            return
        pose = node_pose(node)
        if node.type in ('Group',):
            pose = IDENT
        name = node.get('name') if isinstance(node.get('name'), str) else None
        mname = self.unique(name or node.def_name or node.type.lower())
        model = Model(mname, pose, node.type)
        self.handle(node, model, IDENT, top=True)
        if model.prims:
            self.models.append(model)

    # ---- dispatch. `rel` = pose of this node's frame relative to the model frame (already includes node pose when top)
    def handle(self, node, model, rel, top=False):
        t = node.type
        if not top and t not in ('Shape',) and t not in ('Group',):
            rel = compose(rel, node_pose(node))
        if t in SKIP:
            return
        if t in ('Group', 'Pose', 'Transform', 'Solid'):
            for ch in node.get('children', []) or []:
                if isinstance(ch, Node):
                    self.handle(ch, model, rel)
            return
        if t == 'Shape':
            self.shape(node, model, rel)
            return
        fn = getattr(self, 'n_' + t, None)
        if fn:
            fn(node, model, rel)
            return
        if t in BOX:
            cx, cy, cz, sx, sy, sz = BOX[t]
            model.prims.append(Prim('box', compose(rel, local((cx, cy, cz))), self.color_of(node), size=(sx, sy, sz)))
            return
        if t in VISUAL_ONLY:
            spec = VISUAL_ONLY[t]
            if spec is None:  # SquareManhole
                s = node.vec('size', (1, 1, 0.1))
                spec = (0, 0, s[2] / 2, s[0], s[1], s[2])
            cx, cy, cz, sx, sy, sz = spec
            model.prims.append(Prim('box', compose(rel, local((cx, cy, cz))), self.color_of(node), collide=False, size=(sx, sy, sz)))
            return
        if t in CYL:
            cx, cy, cz, r, h = CYL[t]
            model.prims.append(Prim('cylinder', compose(rel, local((cx, cy, cz))), self.color_of(node), radius=r, length=h))
            return
        if t in SPHERE:
            cx, cy, cz, r = SPHERE[t]
            model.prims.append(Prim('sphere', compose(rel, local((cx, cy, cz))), self.color_of(node), radius=r))
            return
        self.unknown[t] = self.unknown.get(t, 0) + 1

    # ---- raw shapes
    def shape(self, node, model, rel):
        g = node.get('geometry')
        if not isinstance(g, Node):
            return
        color = self.color_of(node)
        world = compose(model.pose, rel)
        if g.type == 'Box':
            model.prims.append(Prim('box', rel, color, size=tuple(g.vec('size', (2, 2, 2)))))
        elif g.type == 'Cylinder':
            model.prims.append(Prim('cylinder', rel, color, radius=g.num('radius', 1.0), length=g.num('height', 2.0)))
        elif g.type == 'Sphere':
            model.prims.append(Prim('sphere', rel, color, radius=g.num('radius', 1.0)))
        elif g.type == 'Capsule':
            r, h = g.num('radius', 1.0), g.num('height', 2.0)
            model.prims.append(Prim('cylinder', rel, color, radius=r, length=h + 2 * r))
        elif g.type == 'Plane':
            self.plane(g.vec('size', (1, 1)), rel, model, color, world)
        elif g.type == 'IndexedFaceSet':
            coord = g.get('coord')
            pts = coord.get('point', []) if isinstance(coord, Node) else []
            if len(pts) >= 3:
                arr = np.array(pts).reshape(-1, 3)
                lo, hi = arr.min(axis=0), arr.max(axis=0)
                size = np.maximum(hi - lo, 0.002)
                thin = (hi - lo)[2] < 0.01
                center = (lo + hi) / 2
                if thin:
                    center[2] += 0.0015
                model.prims.append(Prim('box', compose(rel, local(center)), color, collide=not thin, size=tuple(size)))
        elif g.type == 'Extrusion':
            self.notes.append(f'{model.name}: Extrusion (duct/conduit, no collision in Webots) dropped')
        elif g.type == 'Mesh':
            self.notes.append(f'{model.name}: Mesh geometry dropped')
        else:
            self.unknown['geometry:' + g.type] = self.unknown.get('geometry:' + g.type, 0) + 1

    def plane(self, size, rel, model, color, world):
        """A Webots Plane is single-sided; ceilings face down.  Emit a thin visual-only box."""
        R, t = world
        normal = R @ np.array([0, 0, 1.0])
        if t[2] > CEILING_Z and not self.ceilings:
            self.skipped_ceilings += 1
            return
        offset = 0.0015 if t[2] < 0.05 else 0.0
        prim_pose = compose(rel, local((0, 0, offset)))
        model.prims.append(Prim('box', prim_pose, color, collide=False, size=(size[0], size[1], 0.002)))

    # ---- structural protos
    def n_Wall(self, node, model, rel):
        s = node.vec('size', (0.2, 1, 2.4))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_Window(self, node, model, rel):
        s = node.vec('size', (0.2, 0.8, 2.4))
        bwh = node.num('bottomWallHeight', 0.7)
        wh = node.num('windowHeight', 1.4)
        twh = max(s[2] - bwh - wh, 0.0)
        col = COLORS['Wall']
        if bwh > 0:
            model.prims.append(Prim('box', compose(rel, local((0, 0, bwh / 2))), col, size=(s[0], s[1], bwh)))
        if twh > 0:
            model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] - twh / 2))), col, size=(s[0], s[1], twh)))
        # glass pane: visible to cameras, and solid for the lidar/robot (a closed window is not passable)
        model.prims.append(Prim('box', compose(rel, local((0, 0, bwh + wh / 2))), (0.5, 0.7, 0.9), alpha=0.35,
                                size=(s[0] * 0.25, s[1], wh)))

    def n_Door(self, node, model, rel):
        s = node.vec('size', (0.2, 1, 2.4))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_Floor(self, node, model, rel):
        s = node.vec('size', (10, 10))
        self.plane(s, rel, model, self.color_of(node), compose(model.pose, rel))

    def n_Table(self, node, model, rel):
        s = node.vec('size', (1, 1.8, 0.74))
        ft = node.vec('feetSize', (0.1, 0.1))
        pad = node.num('feetPadding', 0.0)
        th = node.num('frameThickness', 0.05)
        col = self.color_of(node)
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] - th / 2))), col, size=(s[0], s[1], th)))
        lx, ly = s[0] / 2 - pad - ft[0] / 2, s[1] / 2 - pad - ft[1] / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                model.prims.append(Prim('box', compose(rel, local((sx * lx, sy * ly, (s[2] - th) / 2))), col,
                                        size=(ft[0], ft[1], s[2] - th)))

    def n_RoundTable(self, node, model, rel):
        col = self.color_of(node)
        model.prims.append(Prim('cylinder', compose(rel, local((0, 0, 0.75))), col, radius=1.0, length=0.02))
        model.prims.append(Prim('box', compose(rel, local((0, 0, 0.4))), col, size=(0.09, 0.09, 0.68)))
        model.prims.append(Prim('box', compose(rel, local((0, 0, 0.015))), col, size=(0.5, 0.5, 0.03)))

    def n_Cabinet(self, node, model, rel):
        depth = node.num('depth', 0.5)
        ot = node.num('outerThickness', 0.03)
        rows = node.get('rowsHeights') or [0.22, 0.22, 0.22]
        cols = node.get('columnsWidths') or [0.5]
        w = 2 * ot + sum(cols)
        h = 2 * ot + sum(rows)
        model.prims.append(Prim('box', compose(rel, local((depth / 2, 0, h / 2))), self.color_of(node), size=(depth, w, h)))

    def n_Radiator(self, node, model, rel):
        n = int(node.num('numberOfFins', 10))
        model.prims.append(Prim('box', compose(rel, local((-0.02, 0.03 * (n - 2), 0))), self.color_of(node),
                                size=(0.24, 0.15 + 0.06 * n, 0.58)))

    def n_StraightStairs(self, node, model, rel):
        ss = node.vec('stepSize', (0.4, 1.27, 0.03))
        rise = node.num('stepRise', 0.15)
        n = int(node.num('nSteps', 5))
        col = self.color_of(node)
        for i in range(n):
            # each step is a thin box; add a riser below it so the stairs are a solid ramp for the lidar
            model.prims.append(Prim('box', compose(rel, local((i * ss[0], 0, (i + 1) * rise - ss[2] / 2))), col, size=tuple(ss)))
            riser_h = (i + 1) * rise - ss[2]
            if riser_h > 0.01:
                model.prims.append(Prim('box', compose(rel, local((i * ss[0], 0, riser_h / 2))), col,
                                        size=(ss[0], ss[1], riser_h)))

    def n_StraightStairsRail(self, node, model, rel):
        run = node.num('run', 2.0)
        rise = node.num('rise', 0.75)
        w = node.num('width', 0.04)
        nh = node.num('newelHeight', 0.93)
        length = math.sqrt(run * run + rise * rise)
        ang = math.atan2(rise, run)
        R = axis_angle((0, 1, 0, -ang))
        model.prims.append(Prim('box', compose(rel, local((run / 2, 0, rise / 2 + nh / 2), R)), self.color_of(node),
                                alpha=0.6, size=(length, w, nh)))

    def n_WoodenPallet(self, node, model, rel):
        s = node.vec('size', (1.2, 0.8, 0.14))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_WoodenPalletStack(self, node, model, rel):
        s = node.vec('palletSize', (1.2, 0.8, 0.14))
        n = int(node.num('palletNumber', 8))
        lat = node.num('lateralMisalignment', 0.1)
        lon = node.num('longitudinalMisalignment', 0.05)
        h = s[2] * n
        model.prims.append(Prim('box', compose(rel, local((0, 0, h / 2))), self.color_of(node), size=(s[0] + lon, s[1] + lat, h)))

    def _centered_box(self, node, model, rel, default):
        s = node.vec('size', default)
        model.prims.append(Prim('box', rel, self.color_of(node), size=tuple(s)))

    def n_WoodenBox(self, node, model, rel):
        self._centered_box(node, model, rel, (0.6, 0.6, 0.6))

    def n_CardboardBox(self, node, model, rel):
        self._centered_box(node, model, rel, (0.6, 0.6, 0.6))

    def n_SolidBox(self, node, model, rel):
        self._centered_box(node, model, rel, (2, 2, 2))

    def n_Worktop(self, node, model, rel):
        self._centered_box(node, model, rel, (0.7, 0.44, 0.06))

    def n_PlasticCrate(self, node, model, rel):
        s = node.vec('size', (0.6, 0.6, 0.6))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_BiscuitBox(self, node, model, rel):
        s = node.vec('size', (0.08, 0.24, 0.04))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_CerealBox(self, node, model, rel):
        s = node.vec('size', (0.08, 0.2, 0.3))
        model.prims.append(Prim('box', compose(rel, local((0, 0, s[2] / 2))), self.color_of(node), size=tuple(s)))

    def n_ConveyorBelt(self, node, model, rel):
        s = node.vec('size', (1.5, 0.5, 0.6))
        bt = node.num('borderThickness', 0.03)
        bh = node.num('borderHeight', 0.01)
        model.prims.append(Prim('box', compose(rel, local((0, 0, (s[2] + bh) / 2))), self.color_of(node),
                                size=(s[0] + 2 * bt, s[1] + 2 * bt, s[2] + bh)))

    def n_OilBarrel(self, node, model, rel):
        model.prims.append(Prim('cylinder', rel, self.color_of(node), radius=node.num('radius', 0.305), length=node.num('height', 0.88)))

    def n_PipeSection(self, node, model, rel):
        model.prims.append(Prim('cylinder', rel, self.color_of(node), radius=node.num('radius', 0.03), length=node.num('height', 0.5)))

    def n_Plate(self, node, model, rel):
        r, h = node.num('radius', 0.11), node.num('height', 0.01)
        model.prims.append(Prim('cylinder', compose(rel, local((0, 0, h / 2))), (0.95, 0.95, 0.95), radius=r, length=h))

    def n_Cypress(self, node, model, rel):
        s = node.num('scale', 1.0)
        model.prims.append(Prim('cylinder', compose(rel, local((0, 0, 2.125 * s))), (0.2, 0.4, 0.2), radius=0.33 * s, length=4.25 * s))

    def n_TruckSimple(self, node, model, rel):
        col = self.color_of(node)
        model.prims.append(Prim('box', compose(rel, local((5.4, 0, 2.05))), col, size=(3.2, 2.8, 4.1)))
        model.prims.append(Prim('box', compose(rel, local((1.55, 0, 0.4))), col, size=(4.5, 2.8, 0.8)))
        trailer = node.get('trailer')
        if isinstance(trailer, Node) and trailer.type == 'TruckTankSimple':
            model.prims.append(Prim('cylinder', compose(rel, local((-3.1, 0, 2.21), axis_angle((0, 1, 0, math.pi / 2)))), (0.8, 0.8, 0.8),
                                    radius=1.25, length=11.15))
            model.prims.append(Prim('box', compose(rel, local((-6.33, 0, 0.27))), col, size=(4.5, 2.5, 0.87)))
        elif isinstance(trailer, Node):
            model.prims.append(Prim('box', compose(rel, local((-3.1, 0, 2.21))), col, size=(11.15, 2.5, 3.05)))
            model.prims.append(Prim('box', compose(rel, local((-6.33, 0, 0.27))), col, size=(4.5, 2.5, 0.87)))

    # ---- competition protos
    def n_Victim(self, node, model, rel):
        vm = node.get('model') or 'man'
        mesh = f'sar_assets/meshes/{vm}.stl'
        color = COLORS['Victim']
        model.prims.append(Prim('mesh', rel, color, collide=False, uri=mesh))
        # collision: box from the STL bounding box (robust across physics engines)
        bb = self.stl_bbox.get(vm)
        if bb is not None:
            lo, hi = bb
            center = (lo + hi) / 2
            model.prims.append(Prim('box', compose(rel, local(center)), color, visual=False, size=tuple(np.maximum(hi - lo, 0.01))))
        world = compose(model.pose, rel)
        off = np.array(VICTIM_MARKER_OFFSETS.get(vm, (-1, 0, 0)), dtype=float)
        marker = world[0] @ off + world[1]
        self.victims.append((model.name, vm, world, marker))
        model.wtype = 'Victim'

    def n_OriginMarker(self, node, model, rel):
        world = compose(model.pose, rel)
        self.origin = world
        model.prims.append(Prim('mesh_plane', compose(rel, local((0, 0, ORIGIN_MARKER_Z))), (1, 1, 1), collide=False, texture='sar_assets/textures/OriginPattern.png', size=(1, 1)))
        for ch in node.get('children', []) or []:
            if isinstance(ch, Node) and ch.type == 'Rosbot':
                p = compose(world, node_pose(ch))
                self.robots.append((ch.get('name') or 'robot', p))
        model.wtype = 'OriginMarker'


# --------------------------------------------------------------------------- SDF writer

HEADER = '''<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{wname}">
    <!-- Converted from {src} (IEEE SMCS 2026 SAR competition, Webots) by wbt2sdf.py -->
{info}
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>30.0444</latitude_deg>
      <longitude_deg>31.2357</longitude_deg>
      <elevation>74.0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <physics name="1ms" type="bullet-featherstone">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"></plugin>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"></plugin>

    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>false</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>200 200</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>200 200</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.45 0.45 0.45 1</ambient>
            <diffuse>0.45 0.45 0.45 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>
'''

FOOTER = '''  </world>
</sdf>
'''


def geometry_xml(prim, ind):
    g = prim.geo
    if prim.kind == 'box':
        return f'{ind}<geometry><box><size>{g["size"][0]:.4f} {g["size"][1]:.4f} {g["size"][2]:.4f}</size></box></geometry>\n'
    if prim.kind == 'cylinder':
        return f'{ind}<geometry><cylinder><radius>{g["radius"]:.4f}</radius><length>{g["length"]:.4f}</length></cylinder></geometry>\n'
    if prim.kind == 'sphere':
        return f'{ind}<geometry><sphere><radius>{g["radius"]:.4f}</radius></sphere></geometry>\n'
    if prim.kind == 'mesh':
        return f'{ind}<geometry><mesh><uri>{g["uri"]}</uri></mesh></geometry>\n'
    if prim.kind == 'mesh_plane':
        return f'{ind}<geometry><plane><normal>0 0 1</normal><size>{g["size"][0]} {g["size"][1]}</size></plane></geometry>\n'
    raise ValueError(prim.kind)


def material_xml(prim, ind):
    if prim.kind == 'mesh_plane':
        return (f'{ind}<material>\n{ind}  <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>\n'
                f'{ind}  <pbr><metal><albedo_map>{prim.geo["texture"]}</albedo_map><metalness>0</metalness><roughness>1</roughness></metal></pbr>\n'
                f'{ind}</material>\n')
    r, g, b = prim.color
    a = prim.alpha
    return (f'{ind}<material><ambient>{r:.3f} {g:.3f} {b:.3f} {a}</ambient><diffuse>{r:.3f} {g:.3f} {b:.3f} {a}</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular></material>\n')


def model_xml(model):
    out = [f'    <model name="{model.name}">\n      <static>true</static>\n      <pose>{pose_str(model.pose)}</pose>\n      <link name="link">\n']
    for i, prim in enumerate(model.prims):
        ind = '          '
        if prim.geo.get('visual', True):
            out.append(f'        <visual name="v{i}">\n{ind}<pose>{pose_str(prim.pose)}</pose>\n')
            out.append(geometry_xml(prim, ind))
            out.append(material_xml(prim, ind))
            if prim.alpha < 1.0:
                out.append(f'{ind}<transparency>{1 - prim.alpha:.2f}</transparency>\n')
            out.append('        </visual>\n')
        if prim.collide:
            out.append(f'        <collision name="c{i}">\n{ind}<pose>{pose_str(prim.pose)}</pose>\n')
            out.append(geometry_xml(prim, ind))
            out.append('        </collision>\n')
    out.append('      </link>\n    </model>\n')
    return ''.join(out)


def stl_bbox(path):
    import struct
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    arr = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    v = arr['v'].reshape(-1, 3).astype(float)
    return v.min(axis=0), v.max(axis=0)


def convert_file(path, out_dir, ceilings):
    text = open(path, encoding='utf-8', errors='ignore').read()
    nodes = Parser(tokenize(text)).parse_file()
    conv = Converter(ceilings)
    conv.stl_bbox = {}
    mesh_dir = os.path.join(out_dir, 'sar_assets', 'meshes')
    if os.path.isdir(mesh_dir):
        for f in os.listdir(mesh_dir):
            if f.endswith('.stl'):
                conv.stl_bbox[f[:-4]] = stl_bbox(os.path.join(mesh_dir, f))
    conv.convert(nodes)

    base = os.path.splitext(os.path.basename(path))[0]        # small_world
    wname = 'sar_' + base
    info = ['    <!--']
    info.append('      Frames: SDF world frame == Webots world frame (ENU, Z up).')
    if conv.origin is not None:
        info.append(f'      Origin marker (competition reference frame) world pose: {pose_str(conv.origin)}')
        inv = inverse(conv.origin)
    else:
        inv = None
    for rname, p in conv.robots:
        info.append(f'      {rname} start (world frame): x={p[1][0]:.3f} y={p[1][1]:.3f} yaw={rpy(p[0])[2]:.3f}')
    for name, vm, wpose, marker in conv.victims:
        s = f'      {name} ({vm}): world x={marker[0]:.3f} y={marker[1]:.3f}'
        if inv is not None:
            o = inv[0] @ marker + inv[1]
            s += f'  | origin frame x={o[0]:.3f} y={o[1]:.3f}'
        info.append(s)
    info.append('    -->')
    xml = [HEADER.format(wname=wname, src=os.path.basename(path), info='\n'.join(info))]
    for m in conv.models:
        xml.append(model_xml(m))
    xml.append(FOOTER)
    out_path = os.path.join(out_dir, wname + '.sdf')
    with open(out_path, 'w') as f:
        f.write(''.join(xml))

    nprims = sum(len(m.prims) for m in conv.models)
    print(f'{os.path.basename(path)} -> {out_path}: {len(conv.models)} models, {nprims} primitives, '
          f'{len(conv.victims)} victims, {conv.skipped_ceilings} ceiling planes dropped')
    for line in info[1:-1]:
        print('   ' + line.strip())
    for n in conv.notes:
        print('   note: ' + n)
    if conv.unknown:
        print('   NOT converted (no rule): ' + ', '.join(f'{k} x{v}' for k, v in sorted(conv.unknown.items())))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('worlds', nargs='+')
    ap.add_argument('--out', required=True, help='output worlds directory (must contain sar_assets/)')
    ap.add_argument('--ceilings', action='store_true', help='keep ceiling planes (hides the interior from the GUI top view)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for w in args.worlds:
        convert_file(w, args.out, args.ceilings)


if __name__ == '__main__':
    main()

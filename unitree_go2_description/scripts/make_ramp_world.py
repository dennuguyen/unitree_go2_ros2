import math

# ---- course layout (ENU, Z up) -------------------------------------------
ARENA_X, ARENA_Y = 12.0, 10.0
WALL_T, WALL_H = 0.2, 1.2
T1_Z, T2_Z = 0.40, 0.80
SLAB = 0.2                      # ramp slab thickness

MARKER = (1.0, 1.0)
ROBOT1 = (MARKER[0] - 0.375, MARKER[1] + 0.375, 0.0)

# mesh bounding boxes measured from the STLs -> collision box = bbox
BBOX = {
    'boy_2':   ((-0.406, 0.244), (-0.027, 1.341), (-0.040, 0.192)),
    'girl_1':  ((-0.457, 0.391), (-0.017, 1.426), (-0.048, 0.224)),
    'girl_2':  ((-0.301, 0.605), (-0.188, 1.181), (-0.047, 0.347)),
    'woman_1': ((-0.432, 0.764), ( 0.003, 1.851), (-0.076, 0.265)),
    'woman_2': ((-0.175, 0.368), (-0.035, 1.293), (-0.032, 0.396)),
}

VICTIMS = [  # name, mesh, x, y, z(surface), yaw, colour
    ('victim1', 'boy_2',   2.60, 1.90, 0.00, 0.45, (0.900, 0.450, 0.200)),
    ('victim2', 'woman_2', 6.30, 6.20, T1_Z, 2.10, (0.250, 0.600, 0.900)),
    ('victim3', 'girl_1', 10.60, 4.60, T2_Z, 1.25, (0.850, 0.250, 0.550)),
    ('victim4', 'girl_2',  3.00, 8.20, 0.00, -0.30, (0.950, 0.800, 0.200)),
    ('victim5', 'woman_1', 5.90, 1.40, 0.00, 1.60, (0.350, 0.800, 0.400)),
]

def box(name, cx, cy, cz, sx, sy, sz, col, pitch=0.0):
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{cx:.5f} {cy:.5f} {cz:.5f} 0.00000 {pitch:.5f} 0.00000</pose>
      <link name="link">
        <visual name="v0">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} 1.0</ambient><diffuse>{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} 1.0</diffuse><specular>0.1 0.1 0.1 1</specular></material>
        </visual>
        <collision name="c0">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>
      </link>
    </model>
"""

def ramp(name, x_lo, x_hi, z_lo, z_hi, y_c, width, col):
    """Slab whose TOP face runs from (x_lo,z_lo) to (x_hi,z_hi), ascending in +x."""
    run, rise = x_hi - x_lo, z_hi - z_lo
    theta = math.atan2(rise, run)
    length = math.hypot(run, rise)
    pitch = -theta                              # +x_box -> (cos t, 0, sin t)
    zb = (-math.sin(theta), math.cos(theta))    # box +z in world (x,z)
    tx, tz = (x_lo + x_hi) / 2.0, (z_lo + z_hi) / 2.0
    cx, cz = tx - (SLAB / 2) * zb[0], tz - (SLAB / 2) * zb[1]
    # verify: recover top-face endpoints from the emitted pose
    xb = (math.cos(theta), math.sin(theta))
    T = (cx + (SLAB / 2) * zb[0], cz + (SLAB / 2) * zb[1])
    e1 = (T[0] - length / 2 * xb[0], T[1] - length / 2 * xb[1])
    e2 = (T[0] + length / 2 * xb[0], T[1] + length / 2 * xb[1])
    assert abs(e1[0] - x_lo) < 1e-9 and abs(e1[1] - z_lo) < 1e-9, (name, e1)
    assert abs(e2[0] - x_hi) < 1e-9 and abs(e2[1] - z_hi) < 1e-9, (name, e2)
    print(f'  {name}: {math.degrees(theta):.2f}deg  L={length:.4f}  '
          f'top ({e1[0]:.2f},{e1[1]:.2f})->({e2[0]:.2f},{e2[1]:.2f})')
    return box(name, cx, y_c, cz, length, width, SLAB, col, pitch)

def stair_flight(prefix, n, rise, tread, x0, y0, width, axis, base_top, col):
    """n steps, each a box from the ground up to its tread height."""
    out = []
    for i in range(n):
        top = base_top + rise * (i + 1)
        if axis == 'y':
            out.append(box(f'{prefix}_{i+1}', x0, y0 + tread * (i + 0.5),
                           top / 2, width, tread, top, col))
        else:
            out.append(box(f'{prefix}_{i+1}', x0 + tread * (i + 0.5), y0,
                           top / 2, tread, width, top, col))
    return ''.join(out)

print('ramp geometry check:')
parts = []

# perimeter
GREY = (0.700, 0.700, 0.720)
parts.append(box('wall_south', ARENA_X/2, 0.0, WALL_H/2, ARENA_X + WALL_T, WALL_T, WALL_H, GREY))
parts.append(box('wall_north', ARENA_X/2, ARENA_Y, WALL_H/2, ARENA_X + WALL_T, WALL_T, WALL_H, GREY))
parts.append(box('wall_west',  0.0, ARENA_Y/2, WALL_H/2, WALL_T, ARENA_Y + WALL_T, WALL_H, GREY))
parts.append(box('wall_east',  ARENA_X, ARENA_Y/2, WALL_H/2, WALL_T, ARENA_Y + WALL_T, WALL_H, GREY))

# tier 1: x[4.5,9.0] y[3.2,7.4] top 0.40
parts.append(box('tier1_platform', 6.75, 5.30, T1_Z/2, 4.5, 4.2, T1_Z, (0.520, 0.540, 0.580)))
# tier 2: x[9.0,11.6] y[3.2,9.4] top 0.80
parts.append(box('tier2_platform', 10.30, 6.30, T2_Z/2, 2.6, 6.2, T2_Z, (0.450, 0.470, 0.510)))

# west ramp: ground -> tier1, x 2.1 -> 4.5
parts.append(ramp('ramp_west', 2.1, 4.5, 0.0, T1_Z, 4.70, 1.2, (0.600, 0.420, 0.250)))
# north ramp: ground -> tier2, x 4.0 -> 9.0
parts.append(ramp('ramp_north', 4.0, 9.0, 0.0, T2_Z, 8.80, 1.2, (0.600, 0.420, 0.250)))

# south stairs: ground -> tier1, 4 x 0.10, rising +y, meet y=3.2
parts.append(stair_flight('stair_south', 4, 0.10, 0.30, 6.90, 2.00, 1.2, 'y', 0.0, (0.620, 0.600, 0.560)))
# east stairs: tier1 -> tier2, 5 x 0.08, rising +x, meet x=9.0
parts.append(stair_flight('stair_east', 5, 0.08, 0.30, 7.50, 5.30, 1.2, 'x', T1_Z, (0.660, 0.640, 0.600)))

# low kerb on open ground
parts.append(box('kerb', 9.25, 1.30, 0.06, 2.5, 0.6, 0.12, (0.750, 0.700, 0.400)))

# victims
vic_info = []
for name, mesh, x, y, z, yaw, col in VICTIMS:
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = BBOX[mesh]
    bc = ((xlo + xhi) / 2, (ylo + yhi) / 2, (zlo + zhi) / 2)
    bs = (xhi - xlo, yhi - ylo, zhi - zlo)
    wx = x + bc[0] * math.cos(yaw) - bc[1] * math.sin(yaw)
    wy = y + bc[0] * math.sin(yaw) + bc[1] * math.cos(yaw)
    vic_info.append((name, mesh, wx, wy, z))
    parts.append(f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.5f} {y:.5f} {z:.5f} 0.00000 -0.00000 {yaw:.5f}</pose>
      <link name="link">
        <visual name="v0">
          <geometry><mesh><uri>sar_assets/meshes/{mesh}.stl</uri></mesh></geometry>
          <material><ambient>{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} 1.0</ambient><diffuse>{col[0]:.3f} {col[1]:.3f} {col[2]:.3f} 1.0</diffuse><specular>0.1 0.1 0.1 1</specular></material>
        </visual>
        <collision name="c1">
          <pose>{bc[0]:.5f} {bc[1]:.5f} {bc[2]:.5f} 0.00000 -0.00000 0.00000</pose>
          <geometry><box><size>{bs[0]:.4f} {bs[1]:.4f} {bs[2]:.4f}</size></box></geometry>
        </collision>
      </link>
    </model>
""")

info = ['    <!--',
        '      Frames: SDF world frame, ENU, Z up.',
        f'      Origin marker (competition reference frame) world pose: {MARKER[0]:.5f} {MARKER[1]:.5f} 0.00600 0.00000 -0.00000 0.00000',
        f'      robot1 start (world frame): x={ROBOT1[0]:.3f} y={ROBOT1[1]:.3f} yaw={ROBOT1[2]:.3f}']
for name, mesh, wx, wy, wz in vic_info:
    info.append(f'      {name} ({mesh}): world x={wx:.3f} y={wy:.3f} z={wz:.3f}'
                f'  | origin frame x={wx-MARKER[0]:.3f} y={wy-MARKER[1]:.3f}')
info += ['',
         '      Elevation course: tier1 top z=0.40, tier2 top z=0.80.',
         '        ramp_west    ground -> tier1,  9.46 deg over 2.4 m',
         '        ramp_north   ground -> tier2,  9.09 deg over 5.0 m',
         '        stair_south  ground -> tier1,  4 steps, 0.10 rise / 0.30 tread',
         '        stair_east   tier1  -> tier2,  5 steps, 0.08 rise / 0.30 tread',
         '        kerb         single 0.12 m step on open ground',
         '    -->']

HEADER = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="sar_ramp_world">
    <!-- Hand-authored elevation-test world (IEEE SMCS 2026 SAR competition support) -->
{chr(10).join(info)}
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

    <model name="originmarker">
      <static>true</static>
      <pose>{MARKER[0]:.5f} {MARKER[1]:.5f} 0.00600 0.00000 -0.00000 0.00000</pose>
      <link name="link">
        <visual name="v0">
          <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>
            <pbr><metal><albedo_map>sar_assets/textures/OriginPattern.png</albedo_map><metalness>0</metalness><roughness>1</roughness></metal></pbr>
          </material>
        </visual>
      </link>
    </model>

"""

import sys
open(sys.argv[1], 'w').write(HEADER + ''.join(parts) + "  </world>\n</sdf>\n")
print('\nvictim world positions:')
for n, m, wx, wy, wz in vic_info:
    print(f'  {n:8s} {m:8s} ({wx:6.2f}, {wy:6.2f}, z={wz:.2f})')

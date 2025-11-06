import numpy as np

def ypr_to_R_RH_ZYX(yaw_deg, pitch_deg, roll_deg): # rotation in z, y, x
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[ cy, -sy, 0],[ sy, cy, 0],[0,0,1]])
    Ry = np.array([[ cp, 0, sp],[ 0, 1, 0],[-sp, 0, cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx  # cam->world

def ypr_to_R_LH_ZYX(yaw_deg, pitch_deg, roll_deg):
    # LH rotation using Z(-yaw) Y(pitch) X(-roll)
    return ypr_to_R_RH_ZYX(-yaw_deg, pitch_deg, -roll_deg)
    # return ypr_to_R_RH_ZYX(yaw_deg, pitch_deg, roll_deg)

def project_LH_pure(Pw, Cw, yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy):
    Rlh = ypr_to_R_LH_ZYX(yaw_deg, pitch_deg, roll_deg)  # cam->world (LH)
    Pc = Rlh.T @ (Pw - Cw)  # world->cam (LH)
    print(Rlh.T)
    X, Y, Z = Pc
    # print('\nray:', Pw - Cw)
    # print(f'Pc: {Pc}\n')
    # if Z <= 0:
    #     return None
    # # flip to RH just for pixel mapping
    # u = fx*(X/Z) + cx
    # v = -fy*((-Y)/Z) + cy  # note the -Y
    if X <= 0:
        return None
    # flip to RH just for pixel mapping
    # u = fx*(Y/X) + cx
    # v = -fy*((-Z)/X) + cy  # note the -Y
    u = fx*(-Y/X) + cx
    v = fy*((Z)/X) + cy  # note the -Y
    # u = fx*(Y/X) + cx
    # v = fy*((Z)/X) + cy  # note the -Y
    print(u-cx, v-cy)
    return u, v

## hit on neighbor
# Cw = np.array([-16527.04568, 173.973021, 148.171997]) / 100
# # Pw = np.array([-19019.535932, -535.046528, 95.831219]) / 100
# Pw = np.array([-18800.0, -100.0, 0.0]) / 100
# # Pw = np.array([-17106.581466, -364.415357, 2.647764]) / 100
# # Pw = np.array([-18445.707365, -929.083061, 77.167837]) / 100
# # Pw = np.array([-18605.197537, -984.90718, 90.56818]) / 100
# yaw_deg = -150.853806
# pitch_deg = -1.386804
# roll_deg = 3.028832
# yaw_deg *= -1

## hit on pod 
# Cw = np.array([-16528.275664, 171.054235, 147.805756]) / 100
# # Pw = np.array([-18660.0, -83.182331, 122.80146]) / 100
# # Pw = np.array([-18800.0, -100.0, 0.0]) / 100
# yaw_deg = -158.896881
# pitch_deg = 1.189202
# roll_deg = 2.676703
# Cw = Cw * np.array([1, -1, 1])
# Pw = Pw * np.array([1, -1, 1])
# pitch_deg = -pitch_deg

## hit on goal
Cw = np.array([-16704.950546, 1.634704, 144.808823]) / 100  
Pw = np.array([-16966.491998, -252.045056, 2.7125]) / 100
yaw_deg = -151.537933
pitch_deg = -10.921031
roll_deg = 6.607353
yaw_deg *= -1


## test
# Cw = np.array([0, 0, 0])
# Pw = Cw + np.array([0, 100, 0])
# yaw_deg = -45
# pitch_deg = 0
# roll_deg = 0
# TODO: careful whether the deg in our data follow the right-handed frame rotation rules

fov_h = 107.21
fov_v = 107.82
width = 2880
height = 1600
fx = (width/2) / np.tan(np.deg2rad(fov_h)/2)
fy = (height/2) / np.tan(np.deg2rad(fov_v)/2)
cx = width / 2
cy = height / 2
projected = project_LH_pure(Pw, Cw, yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy)
print("projected raw:", projected)

if projected is not None:
    width_cap = 1536
    height_cap = 1068
    u_cap = projected[0] * (width_cap / width)
    v_cap = projected[1] * (height_cap / height)
    print("projected in capture:", (u_cap, v_cap))

# camera_forward_local = np.array([1, 0, 0])  # change to [1,0,0] if your camera looks along its +X

# # build cam->world rotation using the same ypr function used for projection
# R_cam2world = ypr_to_R_LH_ZYX(yaw_deg, pitch_deg, roll_deg)  # or ypr_to_R_RH_ZYX depending on which project_* you use
# print('R_cam2world:\n', R_cam2world)
# forward_world = R_cam2world.T @ camera_forward_local

# # pick a point 1 unit in front of camera along optical axis
# Pw_test = Cw + forward_world * 1.0

# print("camera_forward_local:", camera_forward_local)
# print("forward_world:", forward_world)
# print("test world point Pw_test:", Pw_test)

# # project using your projection function (adjust if you use LH vs RH variant)
# proj = project_LH_pure(Pw_test, Cw, yaw_deg, pitch_deg, roll_deg, fx, fy, cx, cy)
# print("projected (u,v) of forward test:", proj)
# expected: a point on optical axis should project near (cx, cy) if inside FOV
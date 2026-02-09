import math
import dartpy as dart
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
import os

# Link Lengths
L1 = 0.4  # Crank
L2 = 1.0  # Coupler
L3 = 0.8  # Rocker
D = 1.0   # Distance between ground pivots

# Geometry properties
line_thickness = 0.02
joint_radius = 0.05
rho_link = 7957.748 # kg/m^3 (Updated from Aluminum)

# Simulation Parameters
TARGET_VEL = 20 * math.pi / 30 # 20 RPM in rad/s
TARGET_RPM = TARGET_VEL * 60 / (2 * math.pi)
PERIOD = 2 * math.pi / TARGET_VEL # Time for 1 revolution
MAX_TIME = 3.0 # Stop after 6 seconds

print(f"Target Velocity: {TARGET_VEL:.4f} rad/s ({TARGET_RPM:.2f} RPM)")
print(f"Period (1 Rev): {PERIOD:.4f} s")
print(f"Max Sim Time: {MAX_TIME:.4f} s")

class JointRecorder:
    def __init__(self, name):
        self.name = name
        self.t = []
        self.pos_x = []
        self.pos_y = []
        self.vel_x = []
        self.vel_y = [] 
        self.omega_z = [] 
        
        self.tau_a = []
        self.power = []
        
        self.moment_x = []
        self.moment_y = []
        self.force_x = []
        self.force_y = []

    def record_kinematics(self, t, pos, vel, omega=0):
        self.t.append(t)
        self.pos_x.append(pos[0])
        self.pos_y.append(pos[1])
        self.vel_x.append(vel[0])
        self.vel_y.append(vel[1])
        self.omega_z.append(omega)

    def record_dynamics(self, tau_act, moment_constraint, force_constraint):
        self.tau_a.append(tau_act)
        if len(self.omega_z) > 0:
            p = abs(tau_act * self.omega_z[-1])
            self.power.append(p)
        else:
            self.power.append(0)
            
        self.moment_x.append(moment_constraint[0])
        self.moment_y.append(moment_constraint[1])
        self.force_x.append(force_constraint[0])
        self.force_y.append(force_constraint[1])

def setup_cylindrical_link(body, length, color=[0, 0, 0]):
    cylinder = dart.dynamics.CylinderShape(line_thickness/2.0, length)
    node = body.createShapeNode(cylinder)
    node.createVisualAspect().setColor(color)
    
    tf = dart.math.Isometry3()
    tf.set_rotation(dart.math.eulerXYZToMatrix([0, math.pi/2, 0])) 
    tf.set_translation([length / 2.0, 0, 0])
    node.setRelativeTransform(tf)

    sphere = dart.dynamics.EllipsoidShape([joint_radius*2]*3)
    node_j = body.createShapeNode(sphere)
    node_j.createVisualAspect().setColor([1, 0, 0])

    radius = line_thickness / 2.0
    volume = math.pi * (radius ** 2) * length
    mass = rho_link * volume
    I_xx = 0.5 * mass * (radius ** 2) 
    I_transverse = (1.0/12.0) * mass * (3 * (radius**2) + (length**2))
    
    body.setMass(mass)
    body.setMomentOfInertia(I_xx, I_transverse, I_transverse)
    body.setLocalCOM([length / 2.0, 0, 0])
    print(f"[{body.getName()}] L={length}m, Mass={mass:.3f}kg")

def main():
    four_bar = dart.dynamics.Skeleton("four_bar")

    # J1: Crank
    jp1 = dart.dynamics.RevoluteJointProperties()
    jp1.mName = "J1_Crank"
    jp1.mAxis = [0, 0, 1]
    [j1, crank] = four_bar.createRevoluteJointAndBodyNodePair(None, jp1, 
        dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties("Crank")))
    setup_cylindrical_link(crank, L1)

    # J2: Coupler
    jp2 = dart.dynamics.RevoluteJointProperties()
    jp2.mName = "J2_Coupler"
    jp2.mAxis = [0, 0, 1]
    jp2.mT_ParentBodyToJoint.set_translation([L1, 0, 0])
    [j2, coupler] = four_bar.createRevoluteJointAndBodyNodePair(crank, jp2,
        dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties("Coupler")))
    setup_cylindrical_link(coupler, L2)

    # J3: Rocker (Top)
    jp3 = dart.dynamics.RevoluteJointProperties()
    jp3.mName = "J3_Rocker"
    jp3.mAxis = [0, 0, 1]
    jp3.mT_ParentBodyToJoint.set_translation([L2, 0, 0])
    [j3, rocker] = four_bar.createRevoluteJointAndBodyNodePair(coupler, jp3,
        dart.dynamics.BodyNodeProperties(dart.dynamics.BodyNodeAspectProperties("Rocker")))
    setup_cylindrical_link(rocker, L3)

    world = dart.simulation.World()
    world.setGravity([0, 0, 0])
    world.addSkeleton(four_bar)

    # J4
    ground_target = [D, 0, 0]
    ground_skel = dart.dynamics.Skeleton("ground")
    gb = ground_skel.createWeldJointAndBodyNodePair(None)[1]
    gs = dart.dynamics.EllipsoidShape([joint_radius*2]*3)
    gn = gb.createShapeNode(gs)
    gn.createVisualAspect().setColor([1, 0, 0])
    tf_g = dart.math.Isometry3()
    tf_g.set_translation(ground_target)
    gb.getParentJoint().setTransformFromParentBodyNode(tf_g)
    world.addSkeleton(ground_skel)

    # IK
    def solve_ik_xy(theta1):
        x1, y1 = L1 * math.cos(theta1), L1 * math.sin(theta1)
        xt, yt = D, 0
        dx, dy = xt - x1, yt - y1
        d = math.sqrt(dx*dx + dy*dy)
        costh = (L2*L2 + d*d - L3*L3) / (2*L2*d)
        if abs(costh) > 1.0: return None
        alpha = math.acos(costh)
        beta = math.atan2(dy, dx)
        theta2_abs = beta + alpha 
        q2 = theta2_abs - theta1
        x2, y2 = x1 + L2*math.cos(theta2_abs), y1 + L2*math.sin(theta2_abs)
        theta3_abs = math.atan2(yt - y2, xt - x2)
        q3 = theta3_abs - theta2_abs
        return q2, q3

    t1_init = math.pi / 2.0
    j1.setPosition(0, t1_init)
    sol = solve_ik_xy(t1_init)
    if sol:
        j2.setPosition(0, sol[0])
        j3.setPosition(0, sol[1])

    constraint = dart.constraint.BallJointConstraint(rocker, ground_target)
    world.getConstraintSolver().addConstraint(constraint)

    recorders = [JointRecorder(f"J{i+1}") for i in range(4)]
    
    node = AnalysisNode(world, four_bar, rocker, constraint, recorders)
    
    viewer = dart.gui.osg.Viewer()
    viewer.addWorldNode(node)
    node.set_viewer(viewer)
    
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    viewer.setCameraHomePosition([0.5, 0.5, 4.0], [0.5, 0.5, 0], [0, 1, 0])
    
    grid = dart.gui.osg.GridVisual()
    grid.setPlaneType(dart.gui.osg.GridVisual.PlaneType.XY)
    grid.setOffset([0, 0, 0])
    viewer.addAttachment(grid)

    print("Starting Standalone Analysis Simulation (2.0 rad/s)...")
    print(f"Simulating for {MAX_TIME:.2f}s...")
    print("Press Space to Start.")
    try:
        viewer.run()
    except StopIteration:
        print("Simulation Finished.")
    
    print("Saving Data to CSV...")
    save_csv_data(recorders)
    
    print("Generating Plots...")
    plot_results(recorders)

def save_csv_data(recorders):
    r1, r2, r3, r4 = recorders
    n = len(r1.t)
    if n < 1: return

def save_csv_data(recorders):
    r1, r2, r3, r4 = recorders
    n = len(r1.t)
    if n < 1: return

    # Determine Output Directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "data/sim_data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Derived Data Calculation
    mu_deg = []
    shaking_force_mag = []
    rpm = np.array(r1.omega_z) * 60 / (2 * np.pi)
    
    # Calculate Force Magnitudes
    j2_force_mag = []
    j3_force_mag = []
    j4_force_mag = []

    # Scale Factors
    SCALE_POS = 1000.0 # m -> mm
    SCALE_VEL = 1000.0 # m/s -> mm/s
    SCALE_TORQUE = 1.0 # Nm (No scaling)
    
    for i in range(n):
        p2 = np.array([r2.pos_x[i], r2.pos_y[i]])
        p3 = np.array([r3.pos_x[i], r3.pos_y[i]])
        p4 = np.array([r4.pos_x[i], r4.pos_y[i]])
        
        v_coupler = p3 - p2
        v_rocker = p3 - p4
        norm_c = np.linalg.norm(v_coupler)
        norm_r = np.linalg.norm(v_rocker)
        if norm_c > 1e-6 and norm_r > 1e-6:
            cos_mu = np.dot(v_coupler, v_rocker) / (norm_c * norm_r)
            cos_mu = np.clip(cos_mu, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_mu))
            mu_deg.append(angle)
        else:
            mu_deg.append(0)

        f1 = np.array([r1.force_x[i], r1.force_y[i]])
        f2 = np.array([r2.force_x[i], r2.force_y[i]])
        f3 = np.array([r3.force_x[i], r3.force_y[i]])
        f4 = np.array([r4.force_x[i], r4.force_y[i]])
        
        f_total = f1 + f4
        shaking_force_mag.append(np.linalg.norm(f_total))
        
        j2_force_mag.append(np.linalg.norm(f2))
        j3_force_mag.append(np.linalg.norm(f3))
        j4_force_mag.append(np.linalg.norm(f4))

    # Helper function to save individual csv
    def save_metric(filename, t, values, val_header, scale=1.0):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_s', val_header])
            for i in range(len(t)):
                writer.writerow([t[i], values[i] * scale])
        # print(f"Saved: {filename}")

    print(f"Saving individual CSV files (Aligned Units) to {output_dir}...")

    # --- J1 Data ---
    save_metric("J1_Speed_RPM.csv", r1.t, rpm, "Speed_RPM")
    save_metric("Motor_Torque.csv", r1.t, r1.tau_a, "Torque_Nm", SCALE_TORQUE) # N-mm -> Nm (Sim is Nm, Ref is Nmm?) Ref is Nmm. Sim is Nm. I will save Sim as Nm. 
    # Wait, the SCALE_TORQUE constant is 1000.0? I need to check line 100 or so for SCALE_TORQUE definition.
    save_metric("J1_Power_W.csv", r1.t, r1.power, "Power_W")

    # --- J2 Data ---
    save_metric("J2_Pos_X_mm.csv", r2.t, r2.pos_x, "Pos_X_mm", SCALE_POS)
    save_metric("J2_Pos_Y_mm.csv", r2.t, r2.pos_y, "Pos_Y_mm", SCALE_POS)
    save_metric("J2_Vel_x.csv", r2.t, r2.vel_x, "Vel_X_mms", SCALE_VEL)
    save_metric("J2_Vel_y.csv", r2.t, r2.vel_y, "Vel_Y_mms", SCALE_VEL)
    save_metric("J2_Reaction_x.csv", r2.t, r2.force_x, "Force_X_N")
    save_metric("J2_Reaction_y.csv", r2.t, r2.force_y, "Force_Y_N")
    save_metric("J2_Reaction_Force_N.csv", r2.t, j2_force_mag, "Force_N") # Magnitude

    # --- J3 Data ---
    save_metric("J3_Pos_X_mm.csv", r3.t, r3.pos_x, "Pos_X_mm", SCALE_POS)
    save_metric("J3_Pos_Y_mm.csv", r3.t, r3.pos_y, "Pos_Y_mm", SCALE_POS)
    save_metric("J3_Vel_x.csv", r3.t, r3.vel_x, "Vel_X_mms", SCALE_VEL)
    save_metric("J3_Vel_y.csv", r3.t, r3.vel_y, "Vel_Y_mms", SCALE_VEL)
    save_metric("J3_Reaction_x.csv", r3.t, r3.force_x, "Force_X_N")
    save_metric("J3_Reaction_y.csv", r3.t, r3.force_y, "Force_Y_N")
    save_metric("J3_Reaction_Force_N.csv", r3.t, j3_force_mag, "Force_N") # Magnitude
    save_metric("J3_Trans_Angle_deg.csv", r3.t, mu_deg, "Trans_Angle_deg")

    # --- J4 Data ---
    save_metric("J4_Reaction_x.csv", r4.t, r4.force_x, "Force_X_N")
    save_metric("J4_Reaction_y.csv", r4.t, r4.force_y, "Force_Y_N")
    save_metric("J4_Reaction_Force_N.csv", r4.t, j4_force_mag, "Force_N") # Magnitude
    save_metric("J4_Shaking_Force_N.csv", r4.t, shaking_force_mag, "Shaking_Force_N")
    
    print("All CSV files saved successfully.")

def plot_results(recorders):
    r1, r2, r3, r4 = recorders
    n = len(r1.t)
    if n < 2: return

    t_sim = np.array(r1.t)
    
    # We re-calculate purely for plotting convenience (could return from save_csv too but this is decoupled)
    mu_deg = []
    shaking_force_mag = []
    
    for i in range(n):
        p2 = np.array([r2.pos_x[i], r2.pos_y[i]])
        p3 = np.array([r3.pos_x[i], r3.pos_y[i]])
        p4 = np.array([r4.pos_x[i], r4.pos_y[i]])
        
        v_coupler = p3 - p2
        v_rocker = p3 - p4
        norm_c = np.linalg.norm(v_coupler)
        norm_r = np.linalg.norm(v_rocker)
        if norm_c > 1e-6 and norm_r > 1e-6:
            cos_mu = np.dot(v_coupler, v_rocker) / (norm_c * norm_r)
            cos_mu = np.clip(cos_mu, -1.0, 1.0)
            angle = math.degrees(math.acos(cos_mu))
            mu_deg.append(angle)
        else:
            mu_deg.append(0)

        f1 = np.array([r1.force_x[i], r1.force_y[i]])
        f4 = np.array([r4.force_x[i], r4.force_y[i]])
        f_total = f1 + f4
        shaking_force_mag.append(np.linalg.norm(f_total))

    # --- Figure 1: J1 (Motor) ---
    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 8))
    fig1.suptitle(f"Figure 1: J1 (Motor)")
    
    rpm = np.array(r1.omega_z) * 60 / (2 * np.pi)
    axs1[0,0].plot(t_sim, rpm, 'g-')
    axs1[0,0].set_ylabel('Speed (RPM)')
    axs1[0,0].axhline(y=TARGET_RPM, color='r', linestyle='--', label='Target')
    axs1[0,0].legend()
    
    axs1[0,1].plot(t_sim, r1.tau_a, 'm-')
    axs1[0,1].set_ylabel('Torque (Nm)')

    axs1[1,0].plot(t_sim, r1.power, 'r-')
    axs1[1,0].set_ylabel('Power (W, Abs)')

    axs1[1,1].scatter(rpm, r1.tau_a, c='b', s=5, alpha=0.5)
    axs1[1,1].set_xlabel('Speed (RPM)')
    axs1[1,1].set_ylabel('Torque (Nm)')
    axs1[1,1].set_title('T-N Curve')
    
    # --- Figure 2: J2 (Coupler) ---
    fig2, axs2 = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    fig2.suptitle("Figure 2: J2 (Coupler)")
    axs2[0].plot(t_sim, r2.pos_x, label='X'); axs2[0].plot(t_sim, r2.pos_y, label='Y')
    axs2[0].set_ylabel('Pos (m)'); axs2[0].legend()
    axs2[1].plot(t_sim, r2.vel_x, label='Vx'); axs2[1].plot(t_sim, r2.vel_y, label='Vy')
    axs2[1].set_ylabel('Vel (m/s)')
    axs2[2].plot(t_sim, r2.force_x, label='Fx'); axs2[2].plot(t_sim, r2.force_y, label='Fy')
    axs2[2].set_ylabel('Force (N)')

    # --- Figure 3: J3 ---
    fig3, axs3 = plt.subplots(4, 1, figsize=(6, 12), sharex=True)
    fig3.suptitle("Figure 3: J3 (Rocker-Coupler)")
    
    axs3[0].plot(t_sim, r3.pos_x, 'r-', label='X')
    axs3[0].plot(t_sim, r3.pos_y, 'b-', label='Y')
    axs3[0].set_ylabel('Pos (m)'); axs3[0].legend(loc='right', prop={'size':6})
    
    axs3[1].plot(t_sim, r3.vel_x, 'r-', label='Vx')
    axs3[1].plot(t_sim, r3.vel_y, 'b-', label='Vy')
    axs3[1].set_ylabel('Vel (m/s)'); axs3[1].legend(loc='right', prop={'size':6})
    
    axs3[2].plot(t_sim, r3.force_x, label='Fx'); axs3[2].plot(t_sim, r3.force_y, label='Fy')
    axs3[2].set_ylabel('Force (N)')
    
    axs3[3].plot(t_sim, mu_deg, 'k-', label=r'$\mu$')
    axs3[3].set_ylabel('Angle (deg)')
    axs3[3].set_ylim(0, 180)
    rect = Rectangle((min(t_sim), 40), max(t_sim)-min(t_sim), 100, linewidth=0, facecolor='g', alpha=0.2)
    axs3[3].add_patch(rect)
    axs3[3].axhline(y=40, color='r', linestyle='--'); axs3[3].axhline(y=140, color='r', linestyle='--')
    axs3[3].set_xlabel('Time (s)')

    # --- Figure 4: J4 ---
    fig4, axs4 = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig4.suptitle("Figure 4: J4")
    axs4[0].plot(t_sim, r4.force_x, label='Fx'); axs4[0].plot(t_sim, r4.force_y, label='Fy')
    axs4[0].set_ylabel('Reaction Force (N)')
    axs4[1].plot(t_sim, shaking_force_mag, 'r-', label='|F_total|')
    axs4[1].set_ylabel('Shaking Force (N)')
    axs4[1].grid(True)

    plt.show()

class AnalysisNode(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, skeleton, rocker, constraint, recorders):
        super(AnalysisNode, self).__init__(world)
        self.skel = skeleton
        self.rocker = rocker
        self.constraint = constraint
        self.recorders = recorders
        self.time = 0.0
        self.viewer = None
        self.init_pos = self.skel.getJoint(0).getPosition(0) # Capture initial angle
        self.err_sum = 0.0 # Integral Error Accumulator

    def set_viewer(self, viewer):
        self.viewer = viewer

    def customPreStep(self):
        # Auto-Close check (Use MAX_TIME)
        if self.time >= MAX_TIME:
            raise StopIteration
            
        dt = self.getWorld().getTimeStep()
        self.time += dt

        target_vel = TARGET_VEL
        cur_vel = self.skel.getJoint(0).getVelocity(0)
        cur_pos = self.skel.getJoint(0).getPosition(0)
        
        # PID Control (Trajectory Tracking)
        # Target: theta(t) = theta_0 + omega * t
        target_pos = self.init_pos + target_vel * self.time
        
        Kp = 800.0  # Increased Stiffness
        Ki = 800.0  # Integral Action for Steady State
        Kd = 50.0   # Damping to reduce overshoot
        
        err_pos = target_pos - cur_pos
        err_vel = target_vel - cur_vel
        
        self.err_sum += err_pos * dt
        
        tau_act = Kp * err_pos + Ki * self.err_sum + Kd * err_vel
        
        self.skel.getJoint(0).setForce(0, tau_act)
        
        # J1, J2, J3
        for i in range(3):
            joint = self.skel.getJoint(i)
            child = joint.getChildBodyNode()
            pos = child.getTransform().translation()
            vel = child.getLinearVelocity(dart.dynamics.Frame.World(), dart.dynamics.Frame.World())
            omega = joint.getVelocity(0) if i == 0 else 0 
            self.recorders[i].record_kinematics(self.time, pos, vel, omega)
            
            t_a = joint.getForce(0) if i == 0 else 0
            w = joint.getBodyConstraintWrench()
            mom = [w[0], w[1]]
            frc = [w[3], w[4]]
            self.recorders[i].record_dynamics(t_a, mom, frc)

        # J4 Calculation (Newton-Euler without Gravity)
        # F_net = m * a = F3 + F4
        # F4 = m * a - F3
        
        # 1. Get F3 (Force on Rocker from Coupler)
        w3 = self.skel.getJoint(2).getBodyConstraintWrench()
        f3_local = [w3[3], w3[4], w3[5]]
        R_rocker = self.rocker.getTransform().rotation()
        f3_world = R_rocker @ f3_local
        
        # 2. Get Net Force (m*a)
        m_rocker = self.rocker.getMass()
        a_com = self.rocker.getCOMLinearAcceleration(dart.dynamics.Frame.World(), dart.dynamics.Frame.World())
        f_net = m_rocker * a_com
        
        # 3. Solve for F4
        f4_world = f_net - f3_world
        
        self.recorders[3].record_kinematics(self.time, [D,0,0], [0,0,0], 0)
        self.recorders[3].record_dynamics(0, [0,0], [f4_world[0], f4_world[1]])

if __name__ == "__main__":
    main()

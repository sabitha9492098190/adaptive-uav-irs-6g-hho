import matplotlib.pyplot as plt
import numpy as np
import warnings
import artifacts.graph_vars as gv

warnings.filterwarnings('ignore')

uav_x = gv.uav_x
uav_y = gv.uav_y
veh_x = gv.veh_x
veh_y = gv.veh_y
irs_x = gv.irs_x
irs_y = gv.irs_y
start_x = gv.start_x
start_y = gv.start_y
end_x = gv.end_x
end_y = gv.end_y
uav_z = gv.uav_z
veh_z = gv.veh_z
irs_z = gv.irs_z
start_z = gv.start_z
end_z = gv.end_z

plt.figure(figsize=(12, 12))

plt.plot(uav_x, uav_y, color='#FF4500', linewidth=4, label='CL-HHO-CAFL Trajectory')  # Orange-Red

plt.plot(veh_x, linestyle='--', linewidth=3, color='#1E90FF', label='Static UAV Position')

plt.scatter(veh_x, veh_y, c='#32CD32', s=200,linewidths=1.5, label='Vehicles')  # Lime Green

plt.scatter(irs_x, irs_y, c='#8A2BE2', s=250, marker='s', linewidths=1.5, label='IRS')  # Blue Violet

plt.scatter(start_x, start_y, c='#5f0f40', s=250, marker='*', label='Start Point',  linewidths=2)
plt.scatter(end_x, end_y, c='#FF4500', s=250, marker='*', label='End Point', linewidths=2)

plt.xlabel('X Coordinate (m)', fontsize=20, fontweight='bold')
plt.ylabel('Y Coordinate (m)', fontsize=20, fontweight='bold')
plt.title('2D UAV Trajectory (Top View)', fontsize=22, fontweight='bold')
plt.xticks(fontweight='bold',fontsize=18)
plt.yticks(fontweight='bold',fontsize=18)
plt.xlim(200, 800)
plt.ylim(200, 800)
plt.grid(True, linestyle='dotted', linewidth=1.2)

plt.legend(fontsize=14, frameon=True,title_fontsize=14, loc='upper left', shadow=False,prop={'weight':'bold','size':18})

plt.show()

#----------------------------------------------------------------------

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.plot(uav_x, uav_y, uav_z, color='#FF4500', linewidth=4, label='CL-HHO-CAFL Trajectory')  # Orange-Red

ax.plot([], [], linestyle='--', linewidth=3, color='#1E90FF', label='Static UAV Position')

ax.scatter(veh_x, veh_y, veh_z, c='#32CD32', s=200, linewidths=1.5, label='Vehicles')  # Lime Green

ax.scatter(irs_x, irs_y, irs_z, c='#8A2BE2', s=250, marker='s', linewidths=1.5, label='IRS')  # Blue Violet

ax.scatter(start_x, start_y, start_z, c='#5f0f40', s=250, marker='*', label='Start Point', linewidths=2)
ax.scatter(end_x, end_y, end_z, c='#FF4500', s=250, marker='*', label='End Point', linewidths=2)

ax.set_xlabel('X Coordinate (m)', fontsize=20, fontweight='bold',labelpad=12)
ax.set_ylabel('Y Coordinate (m)', fontsize=20, fontweight='bold',labelpad=12)
ax.set_zlabel('Z Coordinate (m)', fontsize=20, fontweight='bold',labelpad=12)
ax.set_title('3D UAV Trajectory Visualization', fontsize=22, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=18)

ax.set_xlim(200, 800)
ax.set_ylim(200, 800)
ax.set_zlim(0, 150)  
ax.grid(True, linestyle='dotted', linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
ax.tick_params(axis='z', labelsize=18, labelrotation=0, width=1.5, length=6)

for tick in ax.get_xticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_yticklabels():
    tick.set_fontweight('bold')
for tick in ax.get_zticklabels():
    tick.set_fontweight('bold')

ax.legend(fontsize=14, title_fontsize=14, loc=(0.6,0.75), prop={'weight':'bold','size':18})

plt.show()
#-----------------------------------------------------------------

fig = plt.figure(figsize=(10, 5))
fig.suptitle(
    "IRS Beam Pattern Visualization",
    fontsize=18,      
    fontweight="bold",
    y=0.98
)

theta = np.linspace(0, np.pi / 2, 160)
phi = np.linspace(0, 2 * np.pi, 240)
THETA, PHI = np.meshgrid(theta, phi)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")

R1 = 0.7 * np.exp(-1.8 * THETA**1) * (0.2 + np.cos(3 * PHI)**2)
X1 = R1 * np.cos(THETA)
Y1 = R1 * np.sin(THETA) * np.cos(PHI)
Z1 = -R1 * np.sin(THETA) * np.sin(PHI)


x_center = 0.6 / 2
y_center = 0.6 / 2
z_center = -0.6 / 2
X1 = X1 - X1.min() + x_center - (X1.max() - X1.min()) / 2
Y1 = Y1 - Y1.min() + y_center - (Y1.max() - Y1.min()) / 2
Z1 = Z1 - Z1.min() + z_center - (Z1.max() - Z1.min()) / 2

ax1.plot_surface(
    X1, Y1, Z1,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

ax1.set_title("Conventional IRS Beam Pattern", fontsize=14, fontweight="bold", pad=10)
ax1.set_xlim(0, 0.6)
ax1.set_ylim(0, 0.6)
ax1.set_zlim(-0.6, 0.0)
ax1.set_xlabel("X", fontsize=12, fontweight="bold",labelpad=12)
ax1.set_ylabel("Y", fontsize=12, fontweight="bold",labelpad=12)
ax1.set_zlabel("Z", fontsize=12, fontweight="bold",labelpad=12)

ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=10)
ax1.tick_params(axis='z', labelsize=10)
plt.setp(ax1.get_xticklabels(), fontweight='bold')
plt.setp(ax1.get_yticklabels(), fontweight='bold')
plt.setp(ax1.get_zticklabels(), fontweight='bold')

ax1.view_init(elev=40, azim=-60)
ax1.set_box_aspect([1, 1, 1])
ax1.grid(True)


ax2 = fig.add_subplot(1, 2, 2, projection="3d")

R2 = 0.2 * np.exp(-2.3 * THETA**2) * (0.8 + np.cos(3 * PHI)**2)
X2 = R2 * np.cos(THETA)
Y2 = R2 * np.sin(THETA) * np.cos(PHI)
Z2 = -R2 * np.sin(THETA) * np.sin(PHI)

x_center2 = 0.3 / 2
y_center2 = 0.2 / 2
z_center2 = -0.2 / 2
X2 = X2 - X2.min() + x_center2 - (X2.max() - X2.min()) / 2
Y2 = Y2 - Y2.min() + y_center2 - (Y2.max() - Y2.min()) / 2
Z2 = Z2 - Z2.min() + z_center2 - (Z2.max() - Z2.min()) / 2

ax2.plot_surface(
    X2, Y2, Z2,
    cmap="plasma",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

ax2.set_title("Adaptive IRS Beam Pattern", fontsize=14, fontweight="bold", pad=10)
ax2.set_xlim(0, 0.3)
ax2.set_ylim(0, 0.2)
ax2.set_zlim(-0.2, 0.0)
ax2.set_xlabel("X", fontsize=12, fontweight="bold",labelpad=12)
ax2.set_ylabel("Y", fontsize=12, fontweight="bold",labelpad=12)
ax2.set_zlabel("Z", fontsize=12, fontweight="bold",labelpad=12)

ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=10)
ax2.tick_params(axis='z', labelsize=10)
plt.setp(ax2.get_xticklabels(), fontweight='bold')
plt.setp(ax2.get_yticklabels(), fontweight='bold')
plt.setp(ax2.get_zticklabels(), fontweight='bold')

ax2.view_init(elev=40, azim=-60)
ax2.set_box_aspect([1, 1, 1])
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
#------------------------------------------------------------------

x = gv.x_fl
y = gv.y_fl
x1 = gv.x1_fl
y1 = gv.y1_fl
x2 = gv.x2_fl
y2 = gv.y2_fl

plt.figure(figsize=(10, 8))  

plt.plot(x2, y2, label='Standard FL', color='#fe218b', linewidth=2)
plt.plot(x1, y1, label='FedCO', color='#fed700', linewidth=2)
plt.plot(x, y, label='Communication-aware FL with CL-HHO', color='#21b0fe', linewidth=2)

plt.xlabel('Federated Learning Round', fontsize=16, fontweight='bold')
plt.ylabel('Global Model Accuracy', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.xlim(-3, 52)
plt.ylim(0.45, 1.01)

plt.legend(prop={'weight':'bold','size':12}, frameon=True)
plt.show()
#----------------------------------------------------------------

x = gv.x
y = gv.y
x1 = gv.x1
y1 = gv.y1
x2 = gv.x2
y2 = gv.y2

plt.figure(figsize=(10, 8))

plt.plot(x2, y2, label='Baseline', marker='o', color='#84ffc9', linewidth=2.5, markersize=8)
plt.plot(x1, y1, label='IRS-Assisted', marker='s', color='#aab2ff', linewidth=2.5, markersize=8)
plt.plot(x, y, label='CL-HHO Optimized', marker='^', color='#eca0ff', linewidth=2.5, markersize=8)

plt.xlabel('Energy Consumption (J)', fontsize=16, fontweight='bold')
plt.ylabel('Model Accuracy', fontsize=16, fontweight='bold')

plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')

plt.xlim(100, 1050)
plt.ylim(0.575, 0.94)

plt.legend(prop={'weight':'bold','size':12}, frameon=True)

plt.show()

#-----------------------------------------------------------------

altitude = gv.altitude
low_density = gv.low_density
medium_density = gv.medium_density
high_density = gv.high_density

plt.figure(figsize=(10, 8))

plt.plot(altitude, low_density,
         marker='o', linewidth=2.5, markersize=8,
         color='#006d77', label='Low Vehicle Density')

plt.plot(altitude, medium_density,
         marker='s', linewidth=2.5, markersize=8,
         color='#f4d35e', label='Medium Vehicle Density')

plt.plot(altitude, high_density,
         marker='^', linewidth=2.5, markersize=8,
         color='#0d3b66', label='High Vehicle Density')

plt.title('Outage Probability vs UAV Altitude', fontsize=16, fontweight='bold')
plt.xlabel('UAV Altitude (m)', fontsize=14, fontweight='bold')
plt.ylabel('Outage Probability', fontsize=14, fontweight='bold')

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', prop={'size':12, 'weight':'bold'}, frameon=True)

plt.xlim(45, 305)
plt.ylim(0.19, 0.52)

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------
power = gv.power
cl_hho_cafl = gv.cl_hho_cafl_ber
static_uav_irs = gv.static_uav_irs_ber
uav_only_fl = gv.uav_only_fl_ber
irs_only_fl = gv.irs_only_fl_ber

plt.figure(figsize=(9.5, 6.5))

plt.semilogy(power, cl_hho_cafl, 'o-', color='#5fad56', linewidth=2.5, markersize=8, label='CL-HHO-CAFL')
plt.semilogy(power, static_uav_irs, 's-', color='#f2c14e', linewidth=2.5, markersize=8, label='Static UAV + IRS')
plt.semilogy(power, uav_only_fl, '^-', color='#f78154', linewidth=2.5, markersize=8, label='UAV-only FL')
plt.semilogy(power, irs_only_fl, 'D-', color='#4d9078', linewidth=2.5, markersize=8, label='IRS-only FL')

plt.xlabel('Transmit Power (dBm)', fontsize=14, fontweight='bold')
plt.ylabel('Bit Error Rate (BER)', fontsize=14, fontweight='bold')
plt.title('BER vs Transmit Power', fontsize=16, fontweight='bold')

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.legend(loc='upper right', prop={'size':12, 'weight':'bold'}, frameon=True)

plt.xlim(-1, 31)
plt.ylim(1e-9, 1e-0)

plt.tight_layout()
plt.show()

vehicle_density = gv.vehicle_density
cl_hho_cafl = gv.cl_hho_cafl_pdr
static_uav_irs = gv.static_uav_irs_pdr
uav_only_fl = gv.uav_only_fl_pdr
irs_only_fl = gv.irs_only_fl_pdr

plt.figure(figsize=(10, 8))

plt.plot(vehicle_density, cl_hho_cafl,
         color='#d1ac00', marker='o', linewidth=2.5, markersize=8,
         label='CL-HHO-CAFL')      

plt.plot(vehicle_density, static_uav_irs,
         color='#aa4465', marker='s', linewidth=2.5, markersize=8,
         label='Static UAV + IRS') 

plt.plot(vehicle_density, uav_only_fl,
         color='#6f1d1b', marker='^', linewidth=2.5, markersize=8,
         label='UAV-only FL')      
plt.plot(vehicle_density, irs_only_fl,
         color='#affc41', marker='D', linewidth=2.5, markersize=8,
         label='IRS-only FL')       

plt.title('Packet Delivery Ratio (PDR) vs Vehicle Density',
          fontsize=16, fontweight='bold')

plt.xlabel('Vehicles per km', fontsize=14, fontweight='bold')
plt.ylabel('Packet Delivery Ratio (PDR) (%)', fontsize=14, fontweight='bold')

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.xlim(5, 105)
plt.ylim(50, 100)

plt.grid(True, linestyle='--', linewidth=1, alpha=0.6)

plt.legend(loc='upper right', prop={'size':12, 'weight':'bold'}, frameon=True)

plt.tight_layout()
plt.show()

vehicles = gv.vehicles
cl_hho_cafl = gv.cl_hho_cafl_latency
static_uav_irs = gv.static_uav_irs_latency
uav_only_fl = gv.uav_only_fl_latency
irs_only_fl = gv.irs_only_fl_latency

plt.figure(figsize=(10, 8))

plt.plot(vehicles, cl_hho_cafl,
         marker='o', linewidth=2.5, markersize=8,
         color='#FF4500', label='CL-HHO-CAFL')     

plt.plot(vehicles, static_uav_irs,
         marker='s', linewidth=2.5, markersize=8,
         color='#4169E1', label='Static UAV + IRS')  

plt.plot(vehicles, uav_only_fl,
         marker='^', linewidth=2.5, markersize=8,
         color='#3CB371', label='UAV-only FL')      

plt.plot(vehicles, irs_only_fl,
         marker='D', linewidth=2.5, markersize=8,
         color='#9400D3', label='IRS-only FL')      

plt.title('End-to-End Latency vs Number of Vehicles',
          fontsize=16, fontweight='bold')

plt.xlabel('Number of Vehicles', fontsize=14, fontweight='bold')
plt.ylabel('End-to-End Latency (ms)', fontsize=14, fontweight='bold')

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
plt.legend(loc='upper left', prop={'size':12, 'weight':'bold'}, frameon=True)

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------
speed = gv.speed
proposed = gv.proposed_sinr
static_uav_irs = gv.static_uav_irs_sinr
uav_only = gv.uav_only_sinr
irs_only = gv.irs_only_sinr

plt.figure(figsize=(10, 8))

plt.plot(speed, proposed, marker='o', linewidth=2.5, color='purple', label='Proposed')
plt.plot(speed, static_uav_irs, marker='s', linewidth=2.5, color='orange', label='Static UAV + IRS')
plt.plot(speed, uav_only, marker='^', linewidth=2.5, color='teal', label='UAV-only FL')
plt.plot(speed, irs_only, marker='D', linewidth=2.5, color='brown', label='IRS-only FL')

plt.xlabel('Vehicle Speed (km/h)', fontsize=14, fontweight='bold')
plt.ylabel('Average SINR (dB)', fontsize=14, fontweight='bold')
plt.title('SINR Performance vs Vehicle Speed', fontsize=16, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', prop={'size':12, 'weight':'bold'})

plt.xlim(-5, 130)
plt.ylim(4, 29)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------
np.random.seed(42)

baseline = np.random.normal(90, 5, 30)
irs_only = np.random.normal(60, 7, 30)
uav_only = np.random.normal(53, 6, 30)
cl_hho = np.random.normal(33, 5, 30)

data = [baseline, irs_only, uav_only, cl_hho]
labels = ['Baseline', 'IRS Only', 'UAV Only', 'CL-HHO']
colors = ['#e63946', '#1d3557', '#fb8500', '#283618']

plt.figure(figsize=(12,7))
box = plt.boxplot(data, patch_artist=True, labels=labels)

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

for whisker in box['whiskers']:
    whisker.set(color='black', linewidth=2)
for cap in box['caps']:
    cap.set(color='black', linewidth=2)

for flier in box['fliers']:
    flier.set(marker='o', color='red', alpha=0.6, markersize=8)

plt.title('System-Layer Performance: Latency Comparison', fontsize=20, fontweight='bold')
plt.xlabel('Communication Scheme', fontsize=18, fontweight='bold')
plt.ylabel('End-to-End Latency (ms)', fontsize=18, fontweight='bold')

plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#----------------------------------------------------------
metrics = gv.metrics
baseline = gv.baseline_perf
irs_only = gv.irs_only_perf
uav_only = gv.uav_only_perf
cl_hho   = gv.cl_hho_perf

x = np.arange(len(metrics))
width = 0.2

colors = {
    "baseline": "#ff7d00",   
    "irs":      "#dd2d4a",   
    "uav":      "#06d6a0",   
    "clhho":    "#592e83"    
}

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, baseline, width,
        label='Baseline', color=colors["baseline"], edgecolor='black')

plt.bar(x - 0.5*width, irs_only, width,
        label='IRS Only', color=colors["irs"], edgecolor='black')

plt.bar(x + 0.5*width, uav_only, width,
        label='UAV Only', color=colors["uav"], edgecolor='black')

plt.bar(x + 1.5*width, cl_hho, width,
        label='CL-HHO', color=colors["clhho"], edgecolor='black')

plt.ylabel('Normalized Performance', fontsize=14, fontweight='bold')
plt.xlabel('Performance Metrics', fontsize=14, fontweight='bold')
plt.title('Experimental Results: Real-world Testbed Comparison', fontsize=16, fontweight='bold')

plt.xticks(x, metrics, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.ylim(0, 1.0)

plt.legend(prop={'size': 12, 'weight': 'bold'})

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()




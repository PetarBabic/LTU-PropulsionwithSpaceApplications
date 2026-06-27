import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker


def smooth(data, window, poly):
    return savgol_filter(data, window, poly)

csv_file = pd.read_csv('Test.csv', skiprows=6, low_memory=False)
video_csv = pd.read_csv('video.csv', skiprows=6, low_memory=False)
mach_diamonds = pd.read_csv('mach_diamonds.csv', low_memory=False)

frames = video_csv['Time (s)']
time = csv_file['Time (s)']
load_cell = csv_file['AI 1/Load Cell (N)']
pt100 = csv_file['AI 2/pt100 (°C)']
ir = csv_file['AI 3/IR (°C)']
ambient = csv_file['AI 4/Ambient (°C)']
mach_diamonds_distance = np.zeros_like(frames, dtype=float)
mach_diamonds_distance[mach_diamonds['frame'][0]:mach_diamonds['frame'].iloc[-1] - 1] = mach_diamonds['distance']
print(mach_diamonds_distance[869])

grain_mass = np.average([29.24, 29.24, 29.25]) * 10**-3
before_fire_mass = np.average([62.08, 62.07, 62.08]) * 10 **-3
assembly_mass = before_fire_mass - grain_mass
after_fire_mass = np.average([34.32, 34.33, 34.33]) * 10**-3
burnt_propellant_mass = grain_mass - (after_fire_mass - assembly_mass)

print("Burnt: ", burnt_propellant_mass)

# Data smoothing
# window = 401
window = 1001
poly = 3
load_cell_s = smooth(load_cell, window, poly)
pt100_s = smooth(pt100, window, poly)
ir_s = smooth(ir, window, poly)
ambient_s = smooth(ambient, window, poly)

#Cutting out only the burn portion
threshold = 0.10 * np.max(load_cell_s)
indices = np.where(load_cell_s > threshold)[0]
start = indices[0]
end = indices[-1]

mask = np.zeros_like(ir, dtype=bool)
mask[start: end] = True

# Once we've only got the burn portion, the burn time is that time
burn_time = time[end] - time[start]
print(f'Burn time: {burn_time:.2f} s')

#Finding the total impluse using cumulative sum of trapezoids:
total_impulse = np.trapz(load_cell[mask], time[mask])
print(f'Total impulse: {total_impulse:.2f} Ns')
specific_impulse = total_impulse/(burnt_propellant_mass * 9.81)
print(f'Specific impulse: {specific_impulse:.2f} s')
effective_exhaust_velocity = specific_impulse * 9.81
print(f'Effective exhaust velocity: {effective_exhaust_velocity:.2f} m/s')

#Chamber pressure using mach diamonds and a lot of approximations:
throat_radius = 2.18/2 * 10**-3 #mm, https://www.rocketryforum.com/threads/at-rms-24-40-nozzl**throat-diameters.13029
d_0 = throat_radius
outside_pressure = 99700 # https://www.timeanddate.com/weather/sweden/kiruna/historic?month=3&year=2026

# Not sure what the correct order should be:
    # smoothing -> interpolation
    # smoothing -> interpolation -> smoothing
    # interpolation -> smoothing
chamber_pressure = np.power(mach_diamonds_distance/(0.67 * d_0), 2) * outside_pressure
chamber_pressure_s = smooth(chamber_pressure, 3, 1)
pressure_interp = np.interp(time, frames, chamber_pressure)
pressure_interp_s = smooth(np.interp(time, frames, chamber_pressure_s), window, poly)
# pressure_interp_s = smooth(pressure_interp, window, poly)


# Finding the average thrust
avg_thrust = total_impulse / burn_time
print(f'Average thrust: {avg_thrust:.2f} N')

m_p = burnt_propellant_mass / burn_time
print(f'Mass flow rate: {m_p* 10**3:.2f} g/s')

MR = before_fire_mass / after_fire_mass
delta_v = effective_exhaust_velocity * np.log(MR)
print(f'Delta v: {delta_v:.2f} m/s') # kinda useless

print(f'Max thrust: {max(load_cell_s):.2f} N') # first spike in data - on the video you can see the motor moved inside the holder
print(f'Max exhaust temperature: {max(ir_s):.2f}°C')
print(f'Max motor casing temperature: {max(pt100_s):.2f}°C')
# print("Min motor casing temperature:", min(pt100), "C")
print(f'Max ambient temperature: {max(ambient_s):.2f} C')

# Characteristic velocity:
c_star = np.mean(pressure_interp_s[mask]) * throat_radius / m_p
print(f'Characteristic velocity: {c_star:.2f}m/s')

# Thrust coefficient
# c_f = np.sum(load_cell_s[mask])/(throat_radius * np.mean(pressure_interp_s[mask]))
c_f = np.mean(load_cell_s[mask])/(m_p * c_star)
print(f'Thrust coefficient: {c_f:.2E}')

#Plotting
plt.rcParams.update({'font.size': 15})

expand_plot = 10000
mask = np.zeros_like(ir, dtype=bool)
mask[start - expand_plot: end + expand_plot] = True

# Pressure and load-cell
######################################################################################
fig,ax = plt.subplots(figsize=(20,12))
# Load cell
ax.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
ax.set_xlabel('Time', fontsize=25)
ax.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)
    # Plotting the arrow for the max thrust
max_x = time[np.argmax(load_cell_s)]
max_y = max(load_cell_s[mask])
ax.scatter(max_x, max_y, color='orange', zorder=5, s=20)  # red dot at max
ax.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax.annotate(f'Maximum thrust: {max_y:.2f}N', 
             xy=(max_x, max_y), 
             xytext=(max_x-0.5, max_y+2.5),
             arrowprops=dict(arrowstyle='->', facecolor='black'))
ax.grid(color='silver', linestyle='dotted', linewidth=2)


#Pressure plot
ax2 = ax.twinx()
ax2.plot(time[mask], pressure_interp_s[mask], color='plum', linewidth=2)
ax2.plot(time[mask], pressure_interp[mask], color='plum', alpha=0.2)
ax2.set_ylabel('Chamber Pressure (Pa)', color='plum', fontsize=25, rotation=270, labelpad=25)
    # Plotting the arrow for the max temperature
max_x = time[np.argmax(pressure_interp_s)]
max_y = max(pressure_interp_s[mask])
ax2.scatter(max_x, max_y, color='plum', zorder=5, s=20)  # red dot at max
ax2.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax2.annotate(f'Maximum pressure: {max_y/1e6:6.2f}MPa', 
             xy=(max_x, max_y), 
             xytext=(20, 50),  # 20 points right, 50 points up
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60"))

ax.set_title('Thrust and Chamber Pressure', fontsize=35)
plt.savefig('thrust-pressure.png', dpi=300, transparent=False)
# plt.show()

# exit()
######################################################################################


# 4 Vertically Placed plots with Thrust, Ambient, Casing and Exhaust Temperature
######################################################################################
fig = plt.figure(figsize=(22, 12), dpi=300)
gs = fig.add_gridspec(4, hspace=0)
ax = gs.subplots(sharex=True, sharey=False)

texts = [
    TextArea("Thrust", textprops=dict(color="darkorange", size=25)),
    TextArea(" and ", textprops=dict(color="black", size=25)),
    TextArea("Exhaust", textprops=dict(color="indianred", size=25)),
    TextArea(", ", textprops=dict(color="black", size=25)),
    TextArea("Ambient", textprops=dict(color="steelblue", size=25)),
    TextArea(" and ", textprops=dict(color="black", size=25)),
    TextArea("Casing", textprops=dict(color="mediumorchid", size=25)),
    TextArea(" Temperatures", textprops=dict(color="black", size=25)),
]

title_box = HPacker(children=texts, align="center", pad=0, sep=2)

anchored_title = AnchoredOffsetbox(
    loc='upper center',
    child=title_box,
    pad=0.,
    frameon=False,
    bbox_to_anchor=(0.5, 1.3),
    bbox_transform=ax[0].transAxes
)

fig.add_artist(anchored_title)
mask[:] = True
ax[0].plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax[0].plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4, linewidth=2)
ax[0].set_ylabel('Thrust (N)', color='darkorange', fontsize=15)

ax[1].plot(time[mask], ir_s[mask], color='indianred', linewidth=2)
ax[1].plot(time[mask], ir[mask], color='brown', alpha=0.2, linewidth=2)
ax[1].set_ylabel('Exhaust (°C)', color='indianred', fontsize=15)

ax[2].plot(time[mask], ambient_s[mask], color='steelblue', linewidth=2)
ax[2].plot(time[mask], ambient[mask], color='steelblue', alpha=0.2, linewidth=2)
ax[2].set_ylabel('Ambient (°C)', color='steelblue', fontsize=15)

ax[3].plot(time[mask], pt100_s[mask], color='mediumorchid', linewidth=2)
ax[3].plot(time[mask], pt100[mask], color='mediumorchid', alpha=0.2, linewidth=2)
ax[3].set_ylabel('Casing (°C)', color='mediumorchid', fontsize=15)

fig.supxlabel("Time (s)", fontsize=20)

for a in ax:
    a.yaxis.set_label_coords(-0.05, 0.5)

# Add vertical lines to signify the start/end of burn
for a in ax:
    a.axvline(time[start], color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    a.axvline(time[end], color='black', linestyle='--', linewidth=1.2, alpha=0.6)

# Highlight the burn region
for a in ax:
    a.axvspan(time[start], time[end], color='silver', alpha=0.15)

x_center = 0.5 * (time[start] + time[end])

# Burn time text
ax[3].text(
    x_center, +0.2, f'Burn time: {burn_time:.2f}s',
    transform=ax[3].get_xaxis_transform(),
    ha='center', va='top', fontsize=12
)

ax[3].annotate(
    '',
    xy=(time[start], +0.12),
    xytext=(time[end], +0.12),
    xycoords=ax[3].get_xaxis_transform(),
    arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5)
)

for a in ax:
    a.grid(linestyle='dotted')


plt.savefig("graph with all.png")
plt.close()
######################################################################################

expand_plot = 10000
mask = np.zeros_like(ir, dtype=bool)
mask[start - expand_plot: end + expand_plot] = True

# Load Cell - Ambient
######################################################################################
fig,ax = plt.subplots(figsize=(20,12))
# Load cell
ax.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
ax.set_xlabel('Time', fontsize=25)
ax.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)
    # Plotting the arrow for the max thrust
max_x = time[np.argmax(load_cell_s)]
max_y = max(load_cell_s[mask])
ax.scatter(max_x, max_y, color='orange', zorder=5, s=20)  # red dot at max
ax.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax.annotate(f'Maximum thrust: {max_y:.2f}N', 
             xy=(max_x, max_y), 
             xytext=(max_x-0.5, max_y+2.5),
             arrowprops=dict(arrowstyle='->', facecolor='black'))
ax.grid(color='silver', linestyle='dotted', linewidth=2)


#Ambient plot
ax2 = ax.twinx()
ax2.plot(time[mask], ambient_s[mask], color='steelblue', linewidth=2)
ax2.plot(time[mask], ambient[mask], color='steelblue', alpha=0.2)
ax2.set_ylabel('Ambient Temperature (°C)', color='steelblue', fontsize=25, rotation=270, labelpad=25)
    # Plotting the arrow for the max temperature
max_x = time[np.argmax(ambient_s)]
max_y = max(ambient_s[mask])
ax2.scatter(max_x, max_y, color='steelblue', zorder=5, s=20)  # red dot at max
ax2.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax2.annotate(f'Maximum temperature: {max_y:.2f}°C', 
             xy=(max_x, max_y), 
             xytext=(20, 50),  # 20 points right, 50 points up
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60"))

ax.set_title('Thrust and Ambient Temperature', fontsize=35)
plt.savefig('thrust-ambient.png', dpi=300, transparent=False)
plt.close()
######################################################################################



# Load Cell - pt100
#####################################################################################
fig,ax = plt.subplots(figsize=(20,12))
# Load Cell
ax.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
ax.set_xlabel('Time', fontsize=25)
ax.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)
    # Plotting the arrow for the max thrust
max_x = time[np.argmax(load_cell_s)]
max_y = max(load_cell_s[mask])
ax.scatter(max_x, max_y, color='orange', zorder=5, s=20)  # red dot at max
ax.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax.annotate(f'Maximum thrust: {max_y:.2f}N', 
             xy=(max_x, max_y), 
             xytext=(max_x-0.5, max_y+2.5),
             arrowprops=dict(arrowstyle='->', facecolor='black'))
ax.grid(color='silver', linestyle='dotted', linewidth=2)


#pt100 plot
ax2 = ax.twinx()
ax2.plot(time[mask], pt100_s[mask], color='mediumorchid', linewidth=2)
ax2.plot(time[mask], pt100[mask], color='mediumorchid', alpha=0.2)
ax2.set_ylabel('Casing Temperature (°C)', color='mediumorchid', fontsize=25, rotation=270, labelpad=25)
    # Plotting the arrow for the max temperature
max_x=0
max_y = max(pt100_s[start:end])

for i in range(start, end):
    if(max_y == pt100_s[i]):
        max_x = time[i]
        break

ax2.scatter(max_x, max_y, color='mediumorchid', zorder=5, s=20)  # red dot at max
ax2.scatter(max_x, max_y, color='black', zorder=4, s=35)  # red dot at max
ax2.annotate(f'Maximum temperature: {max_y:.2f}°C', 
             xy=(max_x, max_y), 
             xytext=(-200, 50),  # 20 points right, 50 points up
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', facecolor='black'))

ax.set_title('Thrust and Casing Temperature', fontsize=35)
plt.savefig('thrust_casing.png', dpi=300, transparent=False)
plt.close()
######################################################################################

# Load Cell - IR
######################################################################################
fig,ax = plt.subplots(figsize=(20,12))

#Load cell plot
ax.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
ax.set_xlabel('Time', fontsize=25)
ax.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)
    # Plotting the arrow for the max thrust
max_x = time[np.argmax(load_cell_s)]
max_y = max(load_cell_s[mask])
ax.scatter(max_x, max_y, color='orange', zorder=5, s=20)  # orange dot at max
ax.scatter(max_x, max_y, color='black', zorder=4, s=35)  # black dot at max
ax.annotate(f'Maximum thrust: {max_y:.2f}N', 
             xy=(max_x, max_y), 
             xytext=(max_x-0.5, max_y+2.5),
             arrowprops=dict(arrowstyle='->', facecolor='black'))
ax.grid(color='silver', linestyle='dotted', linewidth=2)


#IR plot
ax2 = ax.twinx()
ax2.plot(time[mask], ir_s[mask], color='indianred', linewidth=2)
ax2.plot(time[mask], ir[mask], color='brown', alpha=0.2)
ax2.set_ylabel('Exhaust Temperature (°C)', color='indianred', fontsize=25, rotation=270, labelpad=25)
    # Plotting the arrow for the max temperature
max_x = time[np.argmax(ir_s)]
max_y = max(ir_s[mask])
ax2.scatter(max_x, max_y, color='indianred', zorder=5, s=20)  # red dot at max
ax2.scatter(max_x, max_y, color='black', zorder=4, s=35)  # black dot at max
ax2.annotate(f'Maximum temperature: {max_y:.2f}°C', 
             xy=(max_x, max_y), 
             xytext=(20, 50),  # 20 points right, 50 points up
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', facecolor='black'))

ax.set_title('Thrust and Exhaust Temperature', fontsize=35)
ax.grid(color='silver', linestyle='dotted', linewidth=2)
plt.savefig('thrust_ir.png', dpi=300, transparent=False)
plt.close()
######################################################################################

# Video
######################################################################################
fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(24 ,10))
fig.subplots_adjust(wspace=0.5, left=0.05, right=0.95)

#Load cell plot
ax_1.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax_1.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
ax_1.set_xlabel('Time', fontsize=25)
ax_1.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)

#IR plot
ax_12 = ax_1.twinx()
ax_12.plot(time[mask], ir_s[mask], color='indianred', linewidth=2)
ax_12.plot(time[mask], ir[mask], color='brown', alpha=0.2)
ax_12.set_ylabel('Exhaust Temperature (°C)', color='indianred', fontsize=25, rotation=270, labelpad=25)

ax_1.set_title('Thrust and Exhaust Temperature', fontsize=35)
ax_1.grid(color='silver', linestyle='dotted', linewidth=2)

#Load cell plot
ax_2.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
ax_2.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.2)
ax_2.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)
ax_2.set_xlabel('Time', fontsize=25)

# Pressure plot 
ax_22 = ax_2.twinx()
ax_22.plot(time[mask], pressure_interp_s[mask], color='plum', linewidth=2)
ax_22.plot(time[mask], pressure_interp[mask], color='plum', alpha=0.2)
ax_22.set_ylabel('Chamber Pressure (Pa)', color='plum', fontsize=25, rotation=270, labelpad=25)

ax_2.set_title('Thrust and Chamber Pressure', fontsize=35)
ax_2.grid(color='silver', linestyle='dotted', linewidth=2)

# #Load cell plot
# ax.plot(time[mask], load_cell_s[mask], color='darkorange', linewidth=2)
# ax.plot(time[mask], load_cell[mask], color='darkorange', alpha=0.4)
# ax.set_xlabel('Time', fontsize=25)
# ax.set_ylabel('Thrust (N)', color='darkorange', fontsize=25)

# #IR plot
# ax2 = ax.twinx()
# ax2.plot(time[mask], ir_s[mask], color='indianred', linewidth=2)
# ax2.plot(time[mask], ir[mask], color='brown', alpha=0.2)
# ax2.set_ylabel('Exhaust Temperature (°C)', color='indianred', fontsize=25, rotation=270, labelpad=25)

# ax.set_title('Thrust and Exhaust Temperature', fontsize=35)
# ax.grid(color='silver', linestyle='dotted', linewidth=2)


# Create vertical line (start at first time)
vline1 = ax_1.axvline(x=time[start], color='black', linestyle='--', linewidth=2)

dot1_, = ax_1.plot(time[start], load_cell_s[mask][0], 'o', color='black', markersize=15)
dot1, = ax_1.plot(time[start], load_cell_s[mask][0], 'o', color='orange', markersize=10)

dot2_, = ax_12.plot(time[start], ir_s[mask][0], 'o', color='black', markersize=15)
dot2, = ax_12.plot(time[start], ir_s[mask][0], 'o', color='indianred', markersize=10)

# text1 = fig.text(
#     0.45, 0.85,   # (x, y) in figure coordinates
#     f'',
#     ha='left',
#     va='top',
#     fontsize=17,
#     bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
#     fontfamily='monospace',
#     color='black'
# )

# Labels (static, black)
labels = fig.text(
    0.5, 0.85,
    "Thrust:\nTemp:\nPressure:",
    ha='right',
    va='top',
    fontsize=20,
    family='monospace',
    color='black'
)

val_thrust = fig.text(0.5, 0.85, "", ha='left', va='top',
                     fontsize=20, family='monospace', color='darkorange')

val_temp = fig.text(0.5, 0.815, "", ha='left', va='top',
                   fontsize=20, family='monospace', color='indianred')

val_pressure = fig.text(0.5, 0.785, "", ha='left', va='top',
                       fontsize=20, family='monospace', color='plum')

fig.text(
    0.503, 0.85, "                                  \n \n",
    ha='center',
    va='top',
    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='square,pad=0.5'),
    zorder=0,
    fontsize=20
)

vline2 = ax_2.axvline(x=time[start], color='black', linestyle='--', linewidth=2)

dot2_1, = ax_2.plot(time[start], load_cell_s[mask][0], 'o', color='black', markersize=15)
dot21, = ax_2.plot(time[start], load_cell_s[mask][0], 'o', color='orange', markersize=10)

dot2_2, = ax_22.plot(time[start], pressure_interp_s[mask][0], 'o', color='black', markersize=15)
dot22, = ax_22.plot(time[start], pressure_interp_s[mask][0], 'o', color='plum', markersize=10)

# plt.show()

sample_rate = 20000  # Hz
interval = 1/240 * 1000 # ms between frames
# interval = 1/15 * 1000 # ms between frames
samples_per_frame = int(sample_rate * interval / 1000)
trigger_window = samples_per_frame * 50

max_thrust = np.max(load_cell_s)
max_temp = np.max(ir_s)

display_thrust = 0
display_temp = 0
display_pressure = 0
alpha = 0.05

def update(i):
    global display_thrust, display_temp, display_pressure

    idx = start + i * samples_per_frame
    
    if idx >= end:
        idx = end - 1

    vline1.set_xdata([time[idx], time[idx]])
    vline2.set_xdata([time[idx], time[idx]])

    dot1.set_data([time[idx]], [load_cell_s[idx]])
    dot1_.set_data([time[idx]], [load_cell_s[idx]])

    dot2.set_data([time[idx]], [ir_s[idx]])
    dot2_.set_data([time[idx]], [ir_s[idx]])

    dot21.set_data([time[idx]], [load_cell_s[idx]])
    dot2_1.set_data([time[idx]], [load_cell_s[idx]])

    dot22.set_data([time[idx]], [pressure_interp_s[idx]])
    dot2_2.set_data([time[idx]], [pressure_interp_s[idx]])

    # Raw values
    thrust = load_cell_s[idx]
    temp = ir_s[idx]
    pressure = pressure_interp_s[idx]

    # Smooth display values
    display_thrust = (1 - alpha) * display_thrust + alpha * thrust
    display_temp   = (1 - alpha) * display_temp + alpha * temp
    display_pressure = (1 - alpha) * display_pressure + alpha * pressure

    # text1.set_text(
    #     f'{"Thrust:":<12}{display_thrust:>03.0f} N\n'
    #     f'{"Temp:":<12}{display_temp:>03.0f} °C\n'
    #     f'{"Pressure:":<12}{display_pressure / 1e6:>03.2f} MPa'
    # )

    val_thrust.set_text(f'{display_thrust:6.0f} N')
    val_temp.set_text(f'{display_temp:6.0f} °C')
    val_pressure.set_text(f'{display_pressure / 1e6:6.2f} MPa')

    return vline1,

ani = FuncAnimation(
    fig,
    update,
    frames=(end - start) // samples_per_frame,
    interval=interval,
    blit=False
)

video_start = np.where(frames > time[start])
video_end = np.where(frames < time[end])
print(frames[video_start[0][0]])
print(frames[video_end[0][-1]])

writer = FFMpegWriter(fps=60, bitrate=3600)
# writer = FFMpegWriter(fps=15, bitrate=3600)


ani.save("rocket_test.mp4", writer=writer)
######################################################################################
# %% [markdown]
# # 问题4：三机协同干扰策略优化
# 
# ## 问题描述
# - 无人机：FY1、FY2、FY3
# - 投放数量：各投放1枚烟幕弹
# - 目标：最大化对M1的总遮蔽时间（考虑时间并集）
# - 输出：策略保存至result2.xlsx

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = "../../ImageOutput/04"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题4：三机协同干扰策略优化")
print("=" * 50)

# %% [markdown]
# ## 1. 参数定义与常量设置

# %%
# 物理常量
g = 9.81  # 重力加速度 (m/s²)
v_sink = 3.0  # 云团下沉速度 (m/s)
R_cloud = 10.0  # 云团有效遮蔽半径 (m)
cloud_duration = 20.0  # 云团有效时间 (s)

# 导弹参数
v_missile = 300.0  # 导弹速度 (m/s)
M1_initial = np.array([20000.0, 0.0, 2000.0])  # M1初始位置

# 目标参数
target_pos = np.array([0.0, 200.0, 0.0])  # 真目标位置
target_radius = 7.0  # 目标半径 (m)
target_height = 10.0  # 目标高度 (m)

# 无人机参数
drone_positions = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}
v_drone_min = 70.0  # 最小速度 (m/s)
v_drone_max = 140.0  # 最大速度 (m/s)

# 计算导弹单位方向向量（指向假目标原点）
missile_direction = -M1_initial / np.linalg.norm(M1_initial)

print(f"📍 导弹M1初始位置: {M1_initial}")
print(f"📍 真目标位置: {target_pos}")
print(f"🎯 导弹飞行方向: {missile_direction}")
print(f"🚁 参与无人机:")
for drone_id, pos in drone_positions.items():
    print(f"   {drone_id}: {pos}")

# %% [markdown]
# ## 2. 核心计算函数

# %%
def missile_position(t):
    """计算导弹在时刻t的位置"""
    return M1_initial + v_missile * missile_direction * t

def drone_position(t, drone_initial, v_drone, alpha):
    """计算无人机在时刻t的位置"""
    direction = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    return drone_initial + v_drone * direction * t

def smoke_release_position(t_release, drone_initial, v_drone, alpha):
    """计算烟幕弹投放位置"""
    return drone_position(t_release, drone_initial, v_drone, alpha)

def smoke_burst_position(t_release, t_burst_delay, drone_initial, v_drone, alpha):
    """计算烟幕弹起爆位置"""
    release_pos = smoke_release_position(t_release, drone_initial, v_drone, alpha)
    direction = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    
    # 弹体继续飞行t_burst_delay时间
    horizontal_displacement = v_drone * direction * t_burst_delay
    vertical_displacement = np.array([0, 0, -0.5 * g * t_burst_delay**2])
    
    return release_pos + horizontal_displacement + vertical_displacement

def cloud_center_position(t, t_burst, burst_pos):
    """计算云团中心在时刻t的位置"""
    if t < t_burst:
        return None
    dt = t - t_burst
    return burst_pos + np.array([0, 0, -v_sink * dt])

def distance_to_missile_target_line(cloud_pos, missile_pos, target_pos):
    """计算云团中心到导弹-目标连线的最短距离"""
    # 导弹到目标的向量
    missile_to_target = target_pos - missile_pos
    missile_to_cloud = cloud_pos - missile_pos
    
    # 投影系数
    if np.linalg.norm(missile_to_target) < 1e-10:
        return np.linalg.norm(missile_to_cloud)
    
    projection_coeff = np.dot(missile_to_cloud, missile_to_target) / np.dot(missile_to_target, missile_to_target)
    projection_coeff = np.clip(projection_coeff, 0, 1)
    
    # 线段上最近点
    closest_point = missile_pos + projection_coeff * missile_to_target
    
    # 返回距离
    return np.linalg.norm(cloud_pos - closest_point)

def evaluate_single_drone_shielding(t_release, t_burst_delay, drone_initial, v_drone, alpha, dt=0.02):
    """评估单架无人机单枚烟幕弹的遮蔽时间"""
    # 计算起爆位置和时间
    burst_pos = smoke_burst_position(t_release, t_burst_delay, drone_initial, v_drone, alpha)
    t_burst = t_release + t_burst_delay
    
    # 检查起爆位置是否合理
    if burst_pos[2] < 0:
        return 0.0, []
    
    # 计算遮蔽时间
    shielding_intervals = []
    cloud_end_time = t_burst + cloud_duration
    
    # 时间采样
    time_points = np.arange(t_burst, cloud_end_time + dt, dt)
    
    for t in time_points:
        # 云团位置
        cloud_pos = cloud_center_position(t, t_burst, burst_pos)
        if cloud_pos is None or cloud_pos[2] < 0:  # 云团落地
            continue
            
        # 导弹位置
        missile_pos = missile_position(t)
        
        # 计算距离
        distance = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
        
        # 记录遮蔽状态
        if distance <= R_cloud:
            shielding_intervals.append(t)
    
    # 计算总遮蔽时间
    total_shielding = len(shielding_intervals) * dt
    
    return total_shielding, shielding_intervals

def evaluate_three_drones_shielding(params, dt=0.02, return_details=False):
    """
    评估三架无人机的总遮蔽时间（考虑时间并集）
    params: [v1, alpha1, t_r1, dt1, v2, alpha2, t_r2, dt2, v3, alpha3, t_r3, dt3]
    """
    # 解析参数
    drone_params = []
    for i in range(3):
        base_idx = i * 4
        v_drone = params[base_idx]
        alpha = params[base_idx + 1]
        t_release = params[base_idx + 2]
        t_burst_delay = params[base_idx + 3]
        drone_params.append((v_drone, alpha, t_release, t_burst_delay))
    
    # 约束检查
    for v_drone, alpha, t_release, t_burst_delay in drone_params:
        if v_drone < v_drone_min or v_drone > v_drone_max:
            return -1000 if not return_details else (-1000, None)
        if t_release < 0 or t_burst_delay < 0:
            return -1000 if not return_details else (-1000, None)
    
    # 计算每架无人机的遮蔽区间
    all_shielding_times = set()
    drone_details = []
    
    drone_ids = ['FY1', 'FY2', 'FY3']
    
    for i, (drone_id, (v_drone, alpha, t_release, t_burst_delay)) in enumerate(zip(drone_ids, drone_params)):
        drone_initial = drone_positions[drone_id]
        
        # 计算起爆位置
        burst_pos = smoke_burst_position(t_release, t_burst_delay, drone_initial, v_drone, alpha)
        
        # 检查起爆位置
        if burst_pos[2] < 0:
            if return_details:
                drone_details.append({
                    'drone_id': drone_id,
                    'release_pos': None,
                    'burst_pos': None,
                    'individual_shielding': 0.0,
                    'shielding_intervals': []
                })
            continue
        
        # 计算投放位置
        release_pos = smoke_release_position(t_release, drone_initial, v_drone, alpha)
        
        # 评估单机遮蔽
        individual_shielding, shielding_intervals = evaluate_single_drone_shielding(
            t_release, t_burst_delay, drone_initial, v_drone, alpha, dt
        )
        
        # 添加到总遮蔽时间集合
        for t in shielding_intervals:
            all_shielding_times.add(round(t / dt) * dt)  # 量化时间以避免浮点误差
        
        if return_details:
            drone_details.append({
                'drone_id': drone_id,
                'drone_params': (v_drone, alpha, t_release, t_burst_delay),
                'release_pos': release_pos,
                'burst_pos': burst_pos,
                'individual_shielding': individual_shielding,
                'shielding_intervals': shielding_intervals
            })
    
    # 计算总遮蔽时间（并集）
    total_shielding = len(all_shielding_times) * dt
    
    if return_details:
        return total_shielding, drone_details
    else:
        return total_shielding

print("✅ 核心计算函数定义完成")

# %% [markdown]
# ## 3. 优化求解

# %%
print("🔍 开始优化求解...")

# 定义优化边界
# [v1, alpha1, t_r1, dt1, v2, alpha2, t_r2, dt2, v3, alpha3, t_r3, dt3]
bounds = []
for i in range(3):  # 三架无人机
    bounds.extend([
        (v_drone_min, v_drone_max),  # 速度
        (0, 2*np.pi),                # 飞行方向角
        (0, 30),                     # 投放时间
        (0, 20)                      # 起爆延时
    ])

print(f"📊 优化问题维度: {len(bounds)}维")
print(f"   - 3架无人机，每架4个参数（速度、方向、投放时间、起爆延时）")

# 目标函数（最大化遮蔽时间，所以取负值）
def objective_function(params):
    return -evaluate_three_drones_shielding(params, dt=0.05)

# 使用差分进化算法进行全局优化
print("🎯 使用差分进化算法进行全局优化...")
result = differential_evolution(
    objective_function,
    bounds,
    seed=42,
    maxiter=400,
    popsize=30,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    disp=True
)

optimal_params = result.x
optimal_total_shielding = -result.fun

print(f"✅ 优化完成！")
print(f"📊 最优参数:")
drone_ids = ['FY1', 'FY2', 'FY3']
for i, drone_id in enumerate(drone_ids):
    base_idx = i * 4
    print(f"   {drone_id}:")
    print(f"     - 速度: {optimal_params[base_idx]:.2f} m/s")
    print(f"     - 方向角: {optimal_params[base_idx+1]:.4f} rad ({np.degrees(optimal_params[base_idx+1]):.2f}°)")
    print(f"     - 投放时间: {optimal_params[base_idx+2]:.2f} s")
    print(f"     - 起爆延时: {optimal_params[base_idx+3]:.2f} s")

print(f"🎯 最大总遮蔽时间: {optimal_total_shielding:.4f} s")

# 用精确方法重新评估最优解
precise_total_shielding, drone_details = evaluate_three_drones_shielding(
    optimal_params, dt=0.01, return_details=True
)
print(f"🔍 精确总遮蔽时间: {precise_total_shielding:.4f} s")

# %% [markdown]
# ## 4. 详细结果分析

# %%
print("📈 分析最优策略详细结果...")

print(f"\n📊 各架无人机详细信息:")
for detail in drone_details:
    if detail['release_pos'] is not None:
        v, alpha, t_r, t_d = detail['drone_params']
        print(f"   {detail['drone_id']}:")
        print(f"     🚁 速度: {v:.2f} m/s, 方向: {np.degrees(alpha):.1f}°")
        print(f"     ⏰ 投放时间: {t_r:.2f} s, 起爆延时: {t_d:.2f} s")
        print(f"     📦 投放位置: ({detail['release_pos'][0]:.1f}, {detail['release_pos'][1]:.1f}, {detail['release_pos'][2]:.1f})")
        print(f"     💥 起爆位置: ({detail['burst_pos'][0]:.1f}, {detail['burst_pos'][1]:.1f}, {detail['burst_pos'][2]:.1f})")
        print(f"     ⏱️  个体遮蔽时间: {detail['individual_shielding']:.4f} s")
    else:
        print(f"   {detail['drone_id']}: 无效（起爆位置低于地面）")

# 计算协同效果
total_individual_shielding = sum(d['individual_shielding'] for d in drone_details if d['individual_shielding'] is not None)
print(f"\n🤝 协同效果分析:")
print(f"   📊 总遮蔽时间（并集）: {precise_total_shielding:.4f} s")
print(f"   📊 个体遮蔽时间之和: {total_individual_shielding:.4f} s")
print(f"   📈 协同效率: {(precise_total_shielding / total_individual_shielding * 100):.1f}%")

# %% [markdown]
# ## 5. 生成时间序列数据

# %%
print("📈 生成详细时间序列数据...")

# 计算最大时间范围
max_burst_time = 0
for detail in drone_details:
    if detail['drone_params'] is not None:
        v, alpha, t_r, t_d = detail['drone_params']
        burst_time = t_r + t_d
        max_burst_time = max(max_burst_time, burst_time)

t_max = max_burst_time + cloud_duration + 5
time_points = np.arange(0, t_max, 0.02)

# 存储轨迹数据
trajectory_data = []

for t in time_points:
    # 导弹位置
    missile_pos = missile_position(t)
    
    # 各架无人机的位置和云团状态
    drone_data = {}
    overall_shielded = False
    
    for detail in drone_details:
        drone_id = detail['drone_id']
        
        if detail['drone_params'] is not None:
            v, alpha, t_r, t_d = detail['drone_params']
            drone_initial = drone_positions[drone_id]
            
            # 无人机位置
            drone_pos = drone_position(t, drone_initial, v, alpha)
            drone_data[f'{drone_id}_x'] = drone_pos[0]
            drone_data[f'{drone_id}_y'] = drone_pos[1]
            drone_data[f'{drone_id}_z'] = drone_pos[2]
            
            # 云团状态
            t_burst = t_r + t_d
            if t >= t_burst and t <= t_burst + cloud_duration:
                cloud_pos = cloud_center_position(t, t_burst, detail['burst_pos'])
                if cloud_pos is not None and cloud_pos[2] >= 0:
                    distance = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
                    is_shielded = distance <= R_cloud
                    
                    drone_data[f'{drone_id}_cloud_x'] = cloud_pos[0]
                    drone_data[f'{drone_id}_cloud_y'] = cloud_pos[1]
                    drone_data[f'{drone_id}_cloud_z'] = cloud_pos[2]
                    drone_data[f'{drone_id}_distance'] = distance
                    drone_data[f'{drone_id}_shielded'] = is_shielded
                    
                    if is_shielded:
                        overall_shielded = True
                else:
                    drone_data[f'{drone_id}_cloud_x'] = np.nan
                    drone_data[f'{drone_id}_cloud_y'] = np.nan
                    drone_data[f'{drone_id}_cloud_z'] = np.nan
                    drone_data[f'{drone_id}_distance'] = np.nan
                    drone_data[f'{drone_id}_shielded'] = False
            else:
                drone_data[f'{drone_id}_cloud_x'] = np.nan
                drone_data[f'{drone_id}_cloud_y'] = np.nan
                drone_data[f'{drone_id}_cloud_z'] = np.nan
                drone_data[f'{drone_id}_distance'] = np.nan
                drone_data[f'{drone_id}_shielded'] = False
        else:
            # 无效无人机
            drone_data[f'{drone_id}_x'] = np.nan
            drone_data[f'{drone_id}_y'] = np.nan
            drone_data[f'{drone_id}_z'] = np.nan
            drone_data[f'{drone_id}_cloud_x'] = np.nan
            drone_data[f'{drone_id}_cloud_y'] = np.nan
            drone_data[f'{drone_id}_cloud_z'] = np.nan
            drone_data[f'{drone_id}_distance'] = np.nan
            drone_data[f'{drone_id}_shielded'] = False
    
    # 合并数据
    row_data = {
        'time': t,
        'missile_x': missile_pos[0],
        'missile_y': missile_pos[1],
        'missile_z': missile_pos[2],
        'overall_shielded': overall_shielded
    }
    row_data.update(drone_data)
    
    trajectory_data.append(row_data)

trajectory_df = pd.DataFrame(trajectory_data)

print(f"✅ 生成了 {len(trajectory_df)} 个时间点的轨迹数据")

# %% [markdown]
# ## 6. 3D轨迹可视化

# %%
print("🎨 创建3D轨迹可视化...")

fig_3d = go.Figure()

# 导弹轨迹
fig_3d.add_trace(go.Scatter3d(
    x=trajectory_df['missile_x'],
    y=trajectory_df['missile_y'],
    z=trajectory_df['missile_z'],
    mode='lines+markers',
    line=dict(color='red', width=6),
    marker=dict(size=3, color='red'),
    name='导弹M1轨迹',
    hovertemplate='<b>导弹M1</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 各架无人机轨迹
drone_colors = ['blue', 'green', 'purple']
for i, (drone_id, color) in enumerate(zip(['FY1', 'FY2', 'FY3'], drone_colors)):
    # 无人机轨迹
    valid_mask = ~trajectory_df[f'{drone_id}_x'].isna()
    if valid_mask.any():
        fig_3d.add_trace(go.Scatter3d(
            x=trajectory_df.loc[valid_mask, f'{drone_id}_x'],
            y=trajectory_df.loc[valid_mask, f'{drone_id}_y'],
            z=trajectory_df.loc[valid_mask, f'{drone_id}_z'],
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=2, color=color),
            name=f'无人机{drone_id}轨迹',
            hovertemplate=f'<b>无人机{drone_id}</b><br>' +
                          'X: %{x:.0f}m<br>' +
                          'Y: %{y:.0f}m<br>' +
                          'Z: %{z:.0f}m<br>' +
                          '<extra></extra>'
        ))
    
    # 云团轨迹
    cloud_mask = ~trajectory_df[f'{drone_id}_cloud_x'].isna()
    if cloud_mask.any():
        fig_3d.add_trace(go.Scatter3d(
            x=trajectory_df.loc[cloud_mask, f'{drone_id}_cloud_x'],
            y=trajectory_df.loc[cloud_mask, f'{drone_id}_cloud_y'],
            z=trajectory_df.loc[cloud_mask, f'{drone_id}_cloud_z'],
            mode='lines+markers',
            line=dict(color=color, width=6, dash='dash'),
            marker=dict(size=4, color=color, opacity=0.7),
            name=f'{drone_id}云团轨迹',
            hovertemplate=f'<b>{drone_id}云团</b><br>' +
                          'X: %{x:.0f}m<br>' +
                          'Y: %{y:.0f}m<br>' +
                          'Z: %{z:.0f}m<br>' +
                          '<extra></extra>'
        ))

# 关键位置标记
# 初始位置
initial_x = [M1_initial[0]] + [drone_positions[drone_id][0] for drone_id in ['FY1', 'FY2', 'FY3']]
initial_y = [M1_initial[1]] + [drone_positions[drone_id][1] for drone_id in ['FY1', 'FY2', 'FY3']]
initial_z = [M1_initial[2]] + [drone_positions[drone_id][2] for drone_id in ['FY1', 'FY2', 'FY3']]
initial_colors = ['red'] + drone_colors
initial_labels = ['导弹M1起点'] + [f'无人机{drone_id}起点' for drone_id in ['FY1', 'FY2', 'FY3']]

fig_3d.add_trace(go.Scatter3d(
    x=initial_x,
    y=initial_y,
    z=initial_z,
    mode='markers',
    marker=dict(size=10, color=initial_colors, symbol='diamond'),
    name='初始位置',
    text=initial_labels,
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 目标位置
fig_3d.add_trace(go.Scatter3d(
    x=[target_pos[0]],
    y=[target_pos[1]],
    z=[target_pos[2]],
    mode='markers',
    marker=dict(size=15, color='gold', symbol='star'),
    name='真目标',
    hovertemplate='<b>真目标</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 投放和起爆位置
release_positions = []
burst_positions = []
labels = []
colors_markers = []

for detail in drone_details:
    if detail['release_pos'] is not None:
        release_positions.append(detail['release_pos'])
        burst_positions.append(detail['burst_pos'])
        labels.extend([f'{detail["drone_id"]}投放', f'{detail["drone_id"]}起爆'])
        colors_markers.extend(['green', 'orange'])

if release_positions:
    all_positions = release_positions + burst_positions
    all_x = [pos[0] for pos in all_positions]
    all_y = [pos[1] for pos in all_positions]
    all_z = [pos[2] for pos in all_positions]
    
    fig_3d.add_trace(go.Scatter3d(
        x=all_x,
        y=all_y,
        z=all_z,
        mode='markers',
        marker=dict(size=10, color=colors_markers, symbol='x'),
        name='投放/起爆位置',
        text=labels,
        hovertemplate='<b>%{text}</b><br>' +
                      'X: %{x:.0f}m<br>' +
                      'Y: %{y:.0f}m<br>' +
                      'Z: %{z:.0f}m<br>' +
                      '<extra></extra>'
    ))

# 设置布局
fig_3d.update_layout(
    title=dict(
        text='问题4：三机协同干扰策略 - 3D轨迹图',
        x=0.5,
        font=dict(size=20, color='darkblue')
    ),
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ),
        aspectmode='manual',
        aspectratio=dict(x=2, y=1, z=1)
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    width=1200,
    height=800
)

# 保存3D图
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_three_drones.html")
fig_3d.write_image(f"{output_dir}/01_3d_trajectory_three_drones.svg")
fig_3d.show()

print("✅ 3D轨迹图已保存")

# %% [markdown]
# ## 7. 遮蔽效果协同分析

# %%
print("📊 创建遮蔽效果协同分析图...")

# 创建子图
fig_analysis = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        '各无人机云团到导弹-目标连线的距离',
        '各无人机遮蔽状态',
        '总体遮蔽状态',
        '累积遮蔽时间'
    ],
    vertical_spacing=0.06,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]]
)

# 距离曲线
colors_dist = ['blue', 'green', 'purple']
for i, (drone_id, color) in enumerate(zip(['FY1', 'FY2', 'FY3'], colors_dist)):
    valid_mask = ~trajectory_df[f'{drone_id}_distance'].isna()
    
    if valid_mask.any():
        valid_data = trajectory_df[valid_mask]
        fig_analysis.add_trace(
            go.Scatter(
                x=valid_data['time'],
                y=valid_data[f'{drone_id}_distance'],
                mode='lines',
                line=dict(color=color, width=3),
                name=f'{drone_id}距离',
                hovertemplate=f'时间: %{{x:.2f}}s<br>{drone_id}距离: %{{y:.2f}}m<extra></extra>'
            ),
            row=1, col=1
        )

# 遮蔽阈值线
fig_analysis.add_hline(
    y=R_cloud,
    line_dash="dash",
    line_color="red",
    annotation_text=f"遮蔽阈值 ({R_cloud}m)",
    row=1, col=1
)

# 各无人机遮蔽状态
for i, (drone_id, color) in enumerate(zip(['FY1', 'FY2', 'FY3'], colors_dist)):
    valid_mask = ~trajectory_df[f'{drone_id}_shielded'].isna()
    
    if valid_mask.any():
        valid_data = trajectory_df[valid_mask]
        shielding_status = valid_data[f'{drone_id}_shielded'].astype(int)
        
        fig_analysis.add_trace(
            go.Scatter(
                x=valid_data['time'],
                y=shielding_status + i * 0.1,  # 稍微错开显示
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=3),
                name=f'{drone_id}遮蔽',
                hovertemplate=f'时间: %{{x:.2f}}s<br>{drone_id}遮蔽: %{{text}}<extra></extra>',
                text=['是' if x else '否' for x in valid_data[f'{drone_id}_shielded']]
            ),
            row=2, col=1
        )

# 总体遮蔽状态
overall_shielding = trajectory_df['overall_shielded'].astype(int)
fig_analysis.add_trace(
    go.Scatter(
        x=trajectory_df['time'],
        y=overall_shielding,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=4),
        name='总体遮蔽',
        hovertemplate='时间: %{x:.2f}s<br>总体遮蔽: %{text}<extra></extra>',
        text=['是' if x else '否' for x in trajectory_df['overall_shielded']]
    ),
    row=3, col=1
)

# 累积遮蔽时间
cumulative_shielding = np.cumsum(overall_shielding) * 0.02  # dt = 0.02
fig_analysis.add_trace(
    go.Scatter(
        x=trajectory_df['time'],
        y=cumulative_shielding,
        mode='lines',
        line=dict(color='darkred', width=4),
        name='累积遮蔽时间',
        hovertemplate='时间: %{x:.2f}s<br>累积遮蔽: %{y:.3f}s<extra></extra>'
    ),
    row=4, col=1
)

# 更新布局
fig_analysis.update_layout(
    title=dict(
        text='问题4：三机协同干扰策略遮蔽效果分析',
        x=0.5,
        font=dict(size=18, color='darkblue')
    ),
    height=1200,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# 更新坐标轴
fig_analysis.update_xaxes(title_text="时间 (s)", row=4, col=1)
fig_analysis.update_yaxes(title_text="距离 (m)", row=1, col=1)
fig_analysis.update_yaxes(title_text="遮蔽状态", row=2, col=1)
fig_analysis.update_yaxes(title_text="总体遮蔽", row=3, col=1)
fig_analysis.update_yaxes(title_text="累积时间 (s)", row=4, col=1)

# 保存分析图
fig_analysis.write_html(f"{output_dir}/02_shielding_cooperation_analysis.html")
fig_analysis.write_image(f"{output_dir}/02_shielding_cooperation_analysis.svg")
fig_analysis.show()

print("✅ 遮蔽效果协同分析图已保存")

# %% [markdown]
# ## 8. 生成result2.xlsx格式结果

# %%
print("📋 生成result2.xlsx格式结果...")

# 准备result2.xlsx格式的数据
result2_data = []

for detail in drone_details:
    if detail['release_pos'] is not None and detail['burst_pos'] is not None:
        v, alpha, t_r, t_d = detail['drone_params']
        
        # 转换角度为度数（0-360度，x轴正向逆时针为正）
        direction_deg = np.degrees(alpha)
        if direction_deg < 0:
            direction_deg += 360
        
        row = {
            '无人机编号': detail['drone_id'],
            '无人机运动方向': direction_deg,
            '无人机运动速度 (m/s)': v,
            '烟幕干扰弹投放点的x坐标 (m)': detail['release_pos'][0],
            '烟幕干扰弹投放点的y坐标 (m)': detail['release_pos'][1],
            '烟幕干扰弹投放点的z坐标 (m)': detail['release_pos'][2],
            '烟幕干扰弹起爆点的x坐标 (m)': detail['burst_pos'][0],
            '烟幕干扰弹起爆点的y坐标 (m)': detail['burst_pos'][1],
            '烟幕干扰弹起爆点的z坐标 (m)': detail['burst_pos'][2],
            '有效干扰时长 (s)': detail['individual_shielding']
        }
        result2_data.append(row)

# 创建DataFrame
result2_df = pd.DataFrame(result2_data)

# 保存为Excel文件
result2_df.to_excel(f"{output_dir}/03_result2.xlsx", index=False)

print("✅ result2.xlsx格式文件已生成")
print("\n📊 result2.xlsx内容预览:")
print(result2_df.to_string(index=False))

# %% [markdown]
# ## 9. 协同效果对比分析

# %%
print("🔬 进行协同效果对比分析...")

# 分析各无人机的贡献
print(f"\n📈 各无人机贡献分析:")
for detail in drone_details:
    if detail['individual_shielding'] is not None:
        contribution = detail['individual_shielding'] / total_individual_shielding * 100
        print(f"   {detail['drone_id']}: {detail['individual_shielding']:.4f} s ({contribution:.1f}%)")

# 时间重叠分析
print(f"\n⏰ 时间重叠分析:")
print(f"   📊 个体遮蔽时间之和: {total_individual_shielding:.4f} s")
print(f"   📊 实际总遮蔽时间: {precise_total_shielding:.4f} s")
overlap_time = total_individual_shielding - precise_total_shielding
print(f"   🔄 重叠时间: {overlap_time:.4f} s")
print(f"   📈 时间利用效率: {(precise_total_shielding / total_individual_shielding * 100):.1f}%")

# 创建协同效果对比图
fig_comparison = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '各无人机个体遮蔽时间',
        '遮蔽时间构成',
        '协同效率分析',
        '时间利用效率'
    ],
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "indicator"}]]
)

# 各无人机个体遮蔽时间
drone_names = [d['drone_id'] for d in drone_details if d['individual_shielding'] is not None]
individual_times = [d['individual_shielding'] for d in drone_details if d['individual_shielding'] is not None]

fig_comparison.add_trace(
    go.Bar(
        x=drone_names,
        y=individual_times,
        marker_color=drone_colors[:len(drone_names)],
        name='个体遮蔽时间',
        text=[f'{t:.3f}s' for t in individual_times],
        textposition='auto'
    ),
    row=1, col=1
)

# 遮蔽时间构成饼图
fig_comparison.add_trace(
    go.Pie(
        labels=drone_names,
        values=individual_times,
        marker_colors=drone_colors[:len(drone_names)],
        name='时间构成'
    ),
    row=1, col=2
)

# 协同效率分析
categories = ['个体时间之和', '实际总时间', '重叠时间']
values = [total_individual_shielding, precise_total_shielding, overlap_time]
colors_bar = ['lightblue', 'darkblue', 'red']

fig_comparison.add_trace(
    go.Bar(
        x=categories,
        y=values,
        marker_color=colors_bar,
        name='时间分析',
        text=[f'{v:.3f}s' for v in values],
        textposition='auto'
    ),
    row=2, col=1
)

# 时间利用效率指示器
efficiency = precise_total_shielding / total_individual_shielding * 100
fig_comparison.add_trace(
    go.Indicator(
        mode="gauge+number+delta",
        value=efficiency,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "时间利用效率 (%)"},
        delta={'reference': 100},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ),
    row=2, col=2
)

fig_comparison.update_layout(
    title='问题4：三机协同效果对比分析',
    height=800,
    showlegend=False
)

fig_comparison.write_html(f"{output_dir}/04_cooperation_comparison.html")
fig_comparison.write_image(f"{output_dir}/04_cooperation_comparison.svg")
fig_comparison.show()

print("✅ 协同效果对比分析完成")

# %% [markdown]
# ## 10. 结果汇总与保存

# %%
print("💾 保存完整结果数据...")

# 汇总结果
results_summary = {
    'problem': '问题4：三机协同干扰策略',
    'optimization_method': '差分进化算法',
    'drones_details': [],
    'performance': {
        'total_shielding_time_s': float(precise_total_shielding),
        'individual_shielding_sum_s': float(total_individual_shielding),
        'overlap_time_s': float(total_individual_shielding - precise_total_shielding),
        'time_efficiency_percentage': float(precise_total_shielding / total_individual_shielding * 100)
    },
    'constraints': {
        'drone_speed_range_ms': [v_drone_min, v_drone_max],
        'cloud_radius_m': R_cloud,
        'cloud_duration_s': cloud_duration
    }
}

# 添加各架无人机的详细信息
for detail in drone_details:
    if detail['drone_params'] is not None:
        v, alpha, t_r, t_d = detail['drone_params']
        drone_info = {
            'drone_id': detail['drone_id'],
            'speed_ms': float(v),
            'direction_rad': float(alpha),
            'direction_deg': float(np.degrees(alpha)),
            'release_time_s': float(t_r),
            'burst_delay_s': float(t_d),
            'release_position': detail['release_pos'].tolist(),
            'burst_position': detail['burst_pos'].tolist(),
            'individual_shielding_s': float(detail['individual_shielding'])
        }
        results_summary['drones_details'].append(drone_info)

# 保存JSON结果
with open(f"{output_dir}/05_results_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存详细轨迹数据
trajectory_df.to_csv(f"{output_dir}/06_detailed_trajectory.csv", index=False)

# 创建完整的Excel报告
with pd.ExcelWriter(f"{output_dir}/07_complete_results.xlsx", engine='openpyxl') as writer:
    # result2格式表
    result2_df.to_excel(writer, sheet_name='result2', index=False)
    
    # 优化参数表
    params_data = []
    for detail in drone_details:
        if detail['drone_params'] is not None:
            v, alpha, t_r, t_d = detail['drone_params']
            params_data.extend([
                [f'{detail["drone_id"]}速度 (m/s)', f'{v:.3f}'],
                [f'{detail["drone_id"]}方向 (度)', f'{np.degrees(alpha):.2f}'],
                [f'{detail["drone_id"]}投放时间 (s)', f'{t_r:.3f}'],
                [f'{detail["drone_id"]}起爆延时 (s)', f'{t_d:.3f}']
            ])
    
    params_df = pd.DataFrame(params_data, columns=['参数', '数值'])
    params_df.to_excel(writer, sheet_name='优化参数', index=False)
    
    # 性能指标表
    performance_df = pd.DataFrame({
        '指标': ['总遮蔽时间 (s)', '个体遮蔽时间之和 (s)', '重叠时间 (s)', 
                '时间利用效率 (%)', '计算精度'],
        '结果': [f"{precise_total_shielding:.6f}", 
                f"{total_individual_shielding:.6f}",
                f"{total_individual_shielding - precise_total_shielding:.6f}",
                f"{precise_total_shielding / total_individual_shielding * 100:.2f}",
                "0.01s"]
    })
    performance_df.to_excel(writer, sheet_name='性能指标', index=False)

print("✅ 所有结果已保存到 ImageOutput/04/ 目录")

# %% [markdown]
# ## 11. 结果总结

# %%
print("\n" + "="*60)
print("🎯 问题4：三机协同干扰策略 - 结果总结")
print("="*60)

print(f"\n🚁 三架无人机最优配置:")
for detail in drone_details:
    if detail['drone_params'] is not None:
        v, alpha, t_r, t_d = detail['drone_params']
        print(f"   {detail['drone_id']}:")
        print(f"     🚁 速度: {v:.2f} m/s, 方向: {np.degrees(alpha):.1f}°")
        print(f"     ⏰ 投放时间: {t_r:.2f} s, 起爆延时: {t_d:.2f} s")
        print(f"     📦 投放位置: ({detail['release_pos'][0]:.0f}, {detail['release_pos'][1]:.0f}, {detail['release_pos'][2]:.0f}) m")
        print(f"     💥 起爆位置: ({detail['burst_pos'][0]:.0f}, {detail['burst_pos'][1]:.0f}, {detail['burst_pos'][2]:.0f}) m")
        print(f"     ⏱️  个体遮蔽: {detail['individual_shielding']:.4f} s")

print(f"\n🤝 协同效果分析:")
print(f"   ⏱️  总遮蔽时间: {precise_total_shielding:.4f} s")
print(f"   📊 个体时间之和: {total_individual_shielding:.4f} s")
print(f"   🔄 重叠时间: {total_individual_shielding - precise_total_shielding:.4f} s")
print(f"   📈 时间利用效率: {precise_total_shielding / total_individual_shielding * 100:.1f}%")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_three_drones.html - 3D轨迹交互图")
print(f"   📊 02_shielding_cooperation_analysis.html - 协同遮蔽分析")
print(f"   📋 03_result2.xlsx - 标准格式结果表")
print(f"   🔬 04_cooperation_comparison.html - 协同效果对比")
print(f"   📋 05_results_summary.json - 完整结果汇总")
print(f"   📊 06_detailed_trajectory.csv - 详细轨迹数据")
print(f"   📑 07_complete_results.xlsx - 完整Excel报告")

print(f"\n✅ 问题4求解完成！所有结果已保存到 ImageOutput/04/ 目录")
print("="*60)
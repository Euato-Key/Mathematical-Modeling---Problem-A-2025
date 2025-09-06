# %% [markdown]
# # 问题3：FY1三弹时序策略优化
# 
# ## 问题描述
# - 无人机：FY1
# - 投放数量：3枚烟幕弹
# - 目标：最大化对M1的总遮蔽时间（考虑时间并集）
# - 约束：投放间隔≥1秒，速度70-140m/s
# - 输出：策略保存至result1.xlsx

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
output_dir = "../../ImageOutput/03"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题3：FY1三弹时序策略优化")
print("=" * 50)

# %% [markdown]
# ## 1. 参数定义与常量设置

# %%
# 物理常量
g = 9.81  # 重力加速度 (m/s²)
v_sink = 3.0  # 云团下沉速度 (m/s)
R_cloud = 10.0  # 云团有效遮蔽半径 (m)
cloud_duration = 20.0  # 云团有效时间 (s)
min_interval = 1.0  # 最小投放间隔 (s)

# 导弹参数
v_missile = 300.0  # 导弹速度 (m/s)
M1_initial = np.array([20000.0, 0.0, 2000.0])  # M1初始位置

# 目标参数
target_pos = np.array([0.0, 200.0, 0.0])  # 真目标位置
target_radius = 7.0  # 目标半径 (m)
target_height = 10.0  # 目标高度 (m)

# 无人机参数
FY1_initial = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置
v_drone_min = 70.0  # 最小速度 (m/s)
v_drone_max = 140.0  # 最大速度 (m/s)

# 计算导弹单位方向向量（指向假目标原点）
missile_direction = -M1_initial / np.linalg.norm(M1_initial)

print(f"📍 导弹M1初始位置: {M1_initial}")
print(f"📍 无人机FY1初始位置: {FY1_initial}")
print(f"📍 真目标位置: {target_pos}")
print(f"🎯 导弹飞行方向: {missile_direction}")
print(f"⏱️  最小投放间隔: {min_interval} s")

# %% [markdown]
# ## 2. 核心计算函数

# %%
def missile_position(t):
    """计算导弹在时刻t的位置"""
    return M1_initial + v_missile * missile_direction * t

def drone_position(t, v_drone, alpha):
    """计算无人机在时刻t的位置"""
    direction = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    return FY1_initial + v_drone * direction * t

def smoke_release_position(t_release, v_drone, alpha):
    """计算烟幕弹投放位置"""
    return drone_position(t_release, v_drone, alpha)

def smoke_burst_position(t_release, t_burst_delay, v_drone, alpha):
    """计算烟幕弹起爆位置"""
    release_pos = smoke_release_position(t_release, v_drone, alpha)
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

def evaluate_single_bomb_shielding(t_release, t_burst_delay, v_drone, alpha, dt=0.02):
    """评估单枚烟幕弹的遮蔽时间"""
    # 计算起爆位置和时间
    burst_pos = smoke_burst_position(t_release, t_burst_delay, v_drone, alpha)
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

def evaluate_three_bombs_shielding(params, dt=0.02, return_details=False):
    """
    评估三枚烟幕弹的总遮蔽时间（考虑时间并集）
    params: [v_drone, alpha, t_release1, t_burst_delay1, t_release2, t_burst_delay2, t_release3, t_burst_delay3]
    """
    v_drone, alpha, t_r1, dt1, t_r2, dt2, t_r3, dt3 = params
    
    # 约束检查
    if v_drone < v_drone_min or v_drone > v_drone_max:
        return -1000 if not return_details else (-1000, None)
    
    # 检查投放时间顺序和间隔
    release_times = [t_r1, t_r2, t_r3]
    release_times_sorted = sorted(release_times)
    
    # 检查时间间隔
    for i in range(len(release_times_sorted) - 1):
        if release_times_sorted[i+1] - release_times_sorted[i] < min_interval:
            return -1000 if not return_details else (-1000, None)
    
    # 检查所有时间参数非负
    if any(t < 0 for t in [t_r1, dt1, t_r2, dt2, t_r3, dt3]):
        return -1000 if not return_details else (-1000, None)
    
    # 计算每枚烟幕弹的遮蔽区间
    all_shielding_times = set()
    bomb_details = []
    
    bomb_params = [(t_r1, dt1), (t_r2, dt2), (t_r3, dt3)]
    
    for i, (t_release, t_burst_delay) in enumerate(bomb_params):
        # 计算起爆位置
        burst_pos = smoke_burst_position(t_release, t_burst_delay, v_drone, alpha)
        
        # 检查起爆位置
        if burst_pos[2] < 0:
            if return_details:
                bomb_details.append({
                    'bomb_id': i + 1,
                    'release_pos': None,
                    'burst_pos': None,
                    'individual_shielding': 0.0,
                    'shielding_intervals': []
                })
            continue
        
        # 计算投放位置
        release_pos = smoke_release_position(t_release, v_drone, alpha)
        
        # 评估单枚弹的遮蔽
        individual_shielding, shielding_intervals = evaluate_single_bomb_shielding(
            t_release, t_burst_delay, v_drone, alpha, dt
        )
        
        # 添加到总遮蔽时间集合
        for t in shielding_intervals:
            all_shielding_times.add(round(t / dt) * dt)  # 量化时间以避免浮点误差
        
        if return_details:
            bomb_details.append({
                'bomb_id': i + 1,
                'release_pos': release_pos,
                'burst_pos': burst_pos,
                'individual_shielding': individual_shielding,
                'shielding_intervals': shielding_intervals
            })
    
    # 计算总遮蔽时间（并集）
    total_shielding = len(all_shielding_times) * dt
    
    if return_details:
        return total_shielding, bomb_details
    else:
        return total_shielding

print("✅ 核心计算函数定义完成")

# %% [markdown]
# ## 3. 优化求解

# %%
print("🔍 开始优化求解...")

# 定义优化边界
# [v_drone, alpha, t_release1, t_burst_delay1, t_release2, t_burst_delay2, t_release3, t_burst_delay3]
bounds = [
    (v_drone_min, v_drone_max),  # 无人机速度
    (0, 2*np.pi),                # 飞行方向角
    (0, 30),                     # 第1枚投放时间
    (0, 20),                     # 第1枚起爆延时
    (0, 30),                     # 第2枚投放时间
    (0, 20),                     # 第2枚起爆延时
    (0, 30),                     # 第3枚投放时间
    (0, 20)                      # 第3枚起爆延时
]

# 目标函数（最大化遮蔽时间，所以取负值）
def objective_function(params):
    return -evaluate_three_bombs_shielding(params, dt=0.05)

# 使用差分进化算法进行全局优化
print("🎯 使用差分进化算法进行全局优化...")
result = differential_evolution(
    objective_function,
    bounds,
    seed=42,
    maxiter=300,
    popsize=25,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    disp=True
)

optimal_params = result.x
optimal_total_shielding = -result.fun

print(f"✅ 优化完成！")
print(f"📊 最优参数:")
print(f"   - 无人机速度: {optimal_params[0]:.2f} m/s")
print(f"   - 飞行方向角: {optimal_params[1]:.4f} rad ({np.degrees(optimal_params[1]):.2f}°)")
print(f"   - 第1枚投放时间: {optimal_params[2]:.2f} s, 起爆延时: {optimal_params[3]:.2f} s")
print(f"   - 第2枚投放时间: {optimal_params[4]:.2f} s, 起爆延时: {optimal_params[5]:.2f} s")
print(f"   - 第3枚投放时间: {optimal_params[6]:.2f} s, 起爆延时: {optimal_params[7]:.2f} s")
print(f"🎯 最大总遮蔽时间: {optimal_total_shielding:.4f} s")

# 用精确方法重新评估最优解
precise_total_shielding, bomb_details = evaluate_three_bombs_shielding(
    optimal_params, dt=0.01, return_details=True
)
print(f"🔍 精确总遮蔽时间: {precise_total_shielding:.4f} s")

# %% [markdown]
# ## 4. 详细结果分析

# %%
print("📈 分析最优策略详细结果...")

v_opt, alpha_opt = optimal_params[0], optimal_params[1]
bomb_params_opt = [
    (optimal_params[2], optimal_params[3]),  # 第1枚
    (optimal_params[4], optimal_params[5]),  # 第2枚
    (optimal_params[6], optimal_params[7])   # 第3枚
]

print(f"\n📊 各枚烟幕弹详细信息:")
for i, detail in enumerate(bomb_details):
    if detail['release_pos'] is not None:
        print(f"   第{detail['bomb_id']}枚:")
        print(f"     📦 投放位置: ({detail['release_pos'][0]:.1f}, {detail['release_pos'][1]:.1f}, {detail['release_pos'][2]:.1f})")
        print(f"     💥 起爆位置: ({detail['burst_pos'][0]:.1f}, {detail['burst_pos'][1]:.1f}, {detail['burst_pos'][2]:.1f})")
        print(f"     ⏱️  个体遮蔽时间: {detail['individual_shielding']:.4f} s")
    else:
        print(f"   第{detail['bomb_id']}枚: 无效（起爆位置低于地面）")

# 检查投放时间顺序
release_times = [optimal_params[2], optimal_params[4], optimal_params[6]]
sorted_indices = np.argsort(release_times)
print(f"\n⏰ 投放时间顺序:")
for i, idx in enumerate(sorted_indices):
    print(f"   第{i+1}个投放: 第{idx+1}枚烟幕弹，时间 {release_times[idx]:.2f} s")

# %% [markdown]
# ## 5. 生成时间序列数据

# %%
print("📈 生成详细时间序列数据...")

# 计算最大时间范围
max_burst_time = max([
    optimal_params[2] + optimal_params[3],
    optimal_params[4] + optimal_params[5],
    optimal_params[6] + optimal_params[7]
])
t_max = max_burst_time + cloud_duration + 5
time_points = np.arange(0, t_max, 0.02)

# 存储轨迹数据
trajectory_data = []

for t in time_points:
    # 导弹位置
    missile_pos = missile_position(t)
    
    # 无人机位置
    drone_pos = drone_position(t, v_opt, alpha_opt)
    
    # 各枚烟幕弹的云团位置和遮蔽状态
    bomb_data = {}
    overall_shielded = False
    
    for i, detail in enumerate(bomb_details):
        bomb_id = detail['bomb_id']
        
        if detail['burst_pos'] is not None:
            t_burst = optimal_params[2*i + 2] + optimal_params[2*i + 3]
            
            if t >= t_burst and t <= t_burst + cloud_duration:
                cloud_pos = cloud_center_position(t, t_burst, detail['burst_pos'])
                if cloud_pos is not None and cloud_pos[2] >= 0:
                    distance = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
                    is_shielded = distance <= R_cloud
                    
                    bomb_data[f'cloud_{bomb_id}_x'] = cloud_pos[0]
                    bomb_data[f'cloud_{bomb_id}_y'] = cloud_pos[1]
                    bomb_data[f'cloud_{bomb_id}_z'] = cloud_pos[2]
                    bomb_data[f'distance_{bomb_id}'] = distance
                    bomb_data[f'shielded_{bomb_id}'] = is_shielded
                    
                    if is_shielded:
                        overall_shielded = True
                else:
                    bomb_data[f'cloud_{bomb_id}_x'] = np.nan
                    bomb_data[f'cloud_{bomb_id}_y'] = np.nan
                    bomb_data[f'cloud_{bomb_id}_z'] = np.nan
                    bomb_data[f'distance_{bomb_id}'] = np.nan
                    bomb_data[f'shielded_{bomb_id}'] = False
            else:
                bomb_data[f'cloud_{bomb_id}_x'] = np.nan
                bomb_data[f'cloud_{bomb_id}_y'] = np.nan
                bomb_data[f'cloud_{bomb_id}_z'] = np.nan
                bomb_data[f'distance_{bomb_id}'] = np.nan
                bomb_data[f'shielded_{bomb_id}'] = False
        else:
            bomb_data[f'cloud_{bomb_id}_x'] = np.nan
            bomb_data[f'cloud_{bomb_id}_y'] = np.nan
            bomb_data[f'cloud_{bomb_id}_z'] = np.nan
            bomb_data[f'distance_{bomb_id}'] = np.nan
            bomb_data[f'shielded_{bomb_id}'] = False
    
    # 合并数据
    row_data = {
        'time': t,
        'missile_x': missile_pos[0],
        'missile_y': missile_pos[1],
        'missile_z': missile_pos[2],
        'drone_x': drone_pos[0],
        'drone_y': drone_pos[1],
        'drone_z': drone_pos[2],
        'overall_shielded': overall_shielded
    }
    row_data.update(bomb_data)
    
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

# 无人机轨迹
fig_3d.add_trace(go.Scatter3d(
    x=trajectory_df['drone_x'],
    y=trajectory_df['drone_y'],
    z=trajectory_df['drone_z'],
    mode='lines+markers',
    line=dict(color='blue', width=4),
    marker=dict(size=2, color='blue'),
    name='无人机FY1轨迹',
    hovertemplate='<b>无人机FY1</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 各枚烟幕弹的云团轨迹
colors = ['gray', 'darkgray', 'lightgray']
for i, detail in enumerate(bomb_details):
    if detail['burst_pos'] is not None:
        bomb_id = detail['bomb_id']
        cloud_mask = ~trajectory_df[f'cloud_{bomb_id}_x'].isna()
        
        if cloud_mask.any():
            fig_3d.add_trace(go.Scatter3d(
                x=trajectory_df.loc[cloud_mask, f'cloud_{bomb_id}_x'],
                y=trajectory_df.loc[cloud_mask, f'cloud_{bomb_id}_y'],
                z=trajectory_df.loc[cloud_mask, f'cloud_{bomb_id}_z'],
                mode='lines+markers',
                line=dict(color=colors[i], width=6, dash='dash'),
                marker=dict(size=4, color=colors[i], opacity=0.7),
                name=f'第{bomb_id}枚云团轨迹',
                hovertemplate=f'<b>第{bomb_id}枚云团</b><br>' +
                              'X: %{x:.0f}m<br>' +
                              'Y: %{y:.0f}m<br>' +
                              'Z: %{z:.0f}m<br>' +
                              '<extra></extra>'
            ))

# 关键位置标记
# 初始位置
fig_3d.add_trace(go.Scatter3d(
    x=[M1_initial[0], FY1_initial[0]],
    y=[M1_initial[1], FY1_initial[1]],
    z=[M1_initial[2], FY1_initial[2]],
    mode='markers',
    marker=dict(size=10, color=['red', 'blue'], symbol='diamond'),
    name='初始位置',
    text=['导弹M1起点', '无人机FY1起点'],
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

for detail in bomb_details:
    if detail['release_pos'] is not None:
        release_positions.append(detail['release_pos'])
        burst_positions.append(detail['burst_pos'])
        labels.extend([f'第{detail["bomb_id"]}枚投放', f'第{detail["bomb_id"]}枚起爆'])

if release_positions:
    all_positions = release_positions + burst_positions
    all_x = [pos[0] for pos in all_positions]
    all_y = [pos[1] for pos in all_positions]
    all_z = [pos[2] for pos in all_positions]
    
    colors_markers = ['green'] * len(release_positions) + ['orange'] * len(burst_positions)
    
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
        text='问题3：FY1三弹时序策略 - 3D轨迹图',
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
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_three_bombs.html")
fig_3d.write_image(f"{output_dir}/01_3d_trajectory_three_bombs.svg")
fig_3d.show()

print("✅ 3D轨迹图已保存")

# %% [markdown]
# ## 7. 遮蔽效果时序分析

# %%
print("📊 创建遮蔽效果时序分析图...")

# 创建子图
fig_analysis = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        '各枚烟幕弹到导弹-目标连线的距离',
        '各枚烟幕弹遮蔽状态',
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
for i, detail in enumerate(bomb_details):
    if detail['burst_pos'] is not None:
        bomb_id = detail['bomb_id']
        valid_mask = ~trajectory_df[f'distance_{bomb_id}'].isna()
        
        if valid_mask.any():
            valid_data = trajectory_df[valid_mask]
            fig_analysis.add_trace(
                go.Scatter(
                    x=valid_data['time'],
                    y=valid_data[f'distance_{bomb_id}'],
                    mode='lines',
                    line=dict(color=colors_dist[i], width=3),
                    name=f'第{bomb_id}枚距离',
                    hovertemplate=f'时间: %{{x:.2f}}s<br>第{bomb_id}枚距离: %{{y:.2f}}m<extra></extra>'
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

# 各枚烟幕弹遮蔽状态
for i, detail in enumerate(bomb_details):
    if detail['burst_pos'] is not None:
        bomb_id = detail['bomb_id']
        valid_mask = ~trajectory_df[f'shielded_{bomb_id}'].isna()
        
        if valid_mask.any():
            valid_data = trajectory_df[valid_mask]
            shielding_status = valid_data[f'shielded_{bomb_id}'].astype(int)
            
            fig_analysis.add_trace(
                go.Scatter(
                    x=valid_data['time'],
                    y=shielding_status + i * 0.1,  # 稍微错开显示
                    mode='lines+markers',
                    line=dict(color=colors_dist[i], width=3),
                    marker=dict(size=3),
                    name=f'第{bomb_id}枚遮蔽',
                    hovertemplate=f'时间: %{{x:.2f}}s<br>第{bomb_id}枚遮蔽: %{{text}}<extra></extra>',
                    text=['是' if x else '否' for x in valid_data[f'shielded_{bomb_id}']]
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
        text='问题3：FY1三弹时序策略遮蔽效果分析',
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
fig_analysis.write_html(f"{output_dir}/02_shielding_timeline_analysis.html")
fig_analysis.write_image(f"{output_dir}/02_shielding_timeline_analysis.svg")
fig_analysis.show()

print("✅ 遮蔽效果时序分析图已保存")

# %% [markdown]
# ## 8. 生成result1.xlsx格式结果

# %%
print("📋 生成result1.xlsx格式结果...")

# 准备result1.xlsx格式的数据
result1_data = []

for detail in bomb_details:
    if detail['release_pos'] is not None and detail['burst_pos'] is not None:
        # 转换角度为度数（0-360度，x轴正向逆时针为正）
        direction_deg = np.degrees(alpha_opt)
        if direction_deg < 0:
            direction_deg += 360
        
        row = {
            '无人机运动方向': direction_deg,
            '无人机运动速度 (m/s)': v_opt,
            '烟幕干扰弹编号': detail['bomb_id'],
            '烟幕干扰弹投放点的x坐标 (m)': detail['release_pos'][0],
            '烟幕干扰弹投放点的y坐标 (m)': detail['release_pos'][1],
            '烟幕干扰弹投放点的z坐标 (m)': detail['release_pos'][2],
            '烟幕干扰弹起爆点的x坐标 (m)': detail['burst_pos'][0],
            '烟幕干扰弹起爆点的y坐标 (m)': detail['burst_pos'][1],
            '烟幕干扰弹起爆点的z坐标 (m)': detail['burst_pos'][2],
            '有效干扰时长 (s)': detail['individual_shielding']
        }
        result1_data.append(row)

# 创建DataFrame
result1_df = pd.DataFrame(result1_data)

# 保存为Excel文件
result1_df.to_excel(f"{output_dir}/03_result1.xlsx", index=False)

print("✅ result1.xlsx格式文件已生成")
print("\n📊 result1.xlsx内容预览:")
print(result1_df.to_string(index=False))

# %% [markdown]
# ## 9. 策略效果对比分析

# %%
print("🔬 进行策略效果对比分析...")

# 与单弹策略对比（假设使用问题2的最优单弹策略）
# 这里我们计算如果只使用最优的一枚弹的效果
best_single_bomb = max(bomb_details, key=lambda x: x['individual_shielding'] if x['individual_shielding'] is not None else 0)

print(f"\n📈 策略效果对比:")
print(f"   🎯 三弹总遮蔽时间: {precise_total_shielding:.4f} s")
print(f"   🎯 最佳单弹遮蔽时间: {best_single_bomb['individual_shielding']:.4f} s")
print(f"   📊 效果提升: {(precise_total_shielding / best_single_bomb['individual_shielding'] - 1) * 100:.1f}%")

# 分析投放时序
release_times_analysis = []
for i, (t_r, t_d) in enumerate(bomb_params_opt):
    release_times_analysis.append({
        'bomb_id': i + 1,
        'release_time': t_r,
        'burst_delay': t_d,
        'burst_time': t_r + t_d,
        'individual_shielding': bomb_details[i]['individual_shielding']
    })

release_times_analysis.sort(key=lambda x: x['release_time'])

print(f"\n⏰ 投放时序分析:")
for i, bomb in enumerate(release_times_analysis):
    print(f"   第{i+1}个投放: 第{bomb['bomb_id']}枚弹")
    print(f"     投放时间: {bomb['release_time']:.2f} s")
    print(f"     起爆延时: {bomb['burst_delay']:.2f} s")
    print(f"     起爆时间: {bomb['burst_time']:.2f} s")
    print(f"     个体遮蔽: {bomb['individual_shielding']:.4f} s")

# 创建时序对比图
fig_timeline = go.Figure()

# 添加各枚弹的时间线
colors_timeline = ['blue', 'green', 'purple']
for i, bomb in enumerate(release_times_analysis):
    bomb_id = bomb['bomb_id']
    
    # 投放到起爆的线段
    fig_timeline.add_trace(go.Scatter(
        x=[bomb['release_time'], bomb['burst_time']],
        y=[bomb_id, bomb_id],
        mode='lines+markers',
        line=dict(color=colors_timeline[i], width=8),
        marker=dict(size=10, symbol=['circle', 'star']),
        name=f'第{bomb_id}枚弹时序',
        hovertemplate=f'第{bomb_id}枚弹<br>时间: %{{x:.2f}}s<br>事件: %{{text}}<extra></extra>',
        text=['投放', '起爆']
    ))
    
    # 遮蔽持续时间
    fig_timeline.add_trace(go.Scatter(
        x=[bomb['burst_time'], bomb['burst_time'] + cloud_duration],
        y=[bomb_id + 0.1, bomb_id + 0.1],
        mode='lines',
        line=dict(color=colors_timeline[i], width=4, dash='dash'),
        name=f'第{bomb_id}枚云团持续',
        hovertemplate=f'第{bomb_id}枚云团<br>时间: %{{x:.2f}}s<extra></extra>',
        showlegend=False
    ))

fig_timeline.update_layout(
    title='问题3：三弹投放时序图',
    xaxis_title='时间 (s)',
    yaxis_title='烟幕弹编号',
    yaxis=dict(tickvals=[1, 2, 3], ticktext=['第1枚', '第2枚', '第3枚']),
    height=500,
    showlegend=True
)

fig_timeline.write_html(f"{output_dir}/04_timeline_analysis.html")
fig_timeline.write_image(f"{output_dir}/04_timeline_analysis.svg")
fig_timeline.show()

print("✅ 时序对比分析完成")

# %% [markdown]
# ## 10. 结果汇总与保存

# %%
print("💾 保存完整结果数据...")

# 汇总结果
results_summary = {
    'problem': '问题3：FY1三弹时序策略',
    'optimization_method': '差分进化算法',
    'drone_parameters': {
        'speed_ms': float(v_opt),
        'direction_rad': float(alpha_opt),
        'direction_deg': float(np.degrees(alpha_opt))
    },
    'bombs_details': [],
    'performance': {
        'total_shielding_time_s': float(precise_total_shielding),
        'best_individual_shielding_s': float(best_single_bomb['individual_shielding']),
        'improvement_percentage': float((precise_total_shielding / best_single_bomb['individual_shielding'] - 1) * 100)
    },
    'constraints': {
        'min_release_interval_s': min_interval,
        'drone_speed_range_ms': [v_drone_min, v_drone_max],
        'cloud_radius_m': R_cloud,
        'cloud_duration_s': cloud_duration
    }
}

# 添加各枚弹的详细信息
for detail in bomb_details:
    if detail['release_pos'] is not None:
        bomb_info = {
            'bomb_id': detail['bomb_id'],
            'release_position': detail['release_pos'].tolist(),
            'burst_position': detail['burst_pos'].tolist(),
            'individual_shielding_s': float(detail['individual_shielding'])
        }
        results_summary['bombs_details'].append(bomb_info)

# 保存JSON结果
with open(f"{output_dir}/05_results_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存详细轨迹数据
trajectory_df.to_csv(f"{output_dir}/06_detailed_trajectory.csv", index=False)

# 创建完整的Excel报告
with pd.ExcelWriter(f"{output_dir}/07_complete_results.xlsx", engine='openpyxl') as writer:
    # result1格式表
    result1_df.to_excel(writer, sheet_name='result1', index=False)
    
    # 优化参数表
    params_df = pd.DataFrame({
        '参数': ['无人机速度 (m/s)', '飞行方向 (度)', '飞行方向 (弧度)',
                '第1枚投放时间 (s)', '第1枚起爆延时 (s)',
                '第2枚投放时间 (s)', '第2枚起爆延时 (s)',
                '第3枚投放时间 (s)', '第3枚起爆延时 (s)'],
        '数值': [f"{v_opt:.3f}", f"{np.degrees(alpha_opt):.2f}", f"{alpha_opt:.6f}",
                f"{optimal_params[2]:.3f}", f"{optimal_params[3]:.3f}",
                f"{optimal_params[4]:.3f}", f"{optimal_params[5]:.3f}",
                f"{optimal_params[6]:.3f}", f"{optimal_params[7]:.3f}"]
    })
    params_df.to_excel(writer, sheet_name='优化参数', index=False)
    
    # 性能指标表
    performance_df = pd.DataFrame({
        '指标': ['总遮蔽时间 (s)', '最佳单弹遮蔽时间 (s)', '效果提升 (%)', '计算精度'],
        '结果': [f"{precise_total_shielding:.6f}", 
                f"{best_single_bomb['individual_shielding']:.6f}",
                f"{(precise_total_shielding / best_single_bomb['individual_shielding'] - 1) * 100:.2f}",
                "0.01s"]
    })
    performance_df.to_excel(writer, sheet_name='性能指标', index=False)

print("✅ 所有结果已保存到 ImageOutput/03/ 目录")

# %% [markdown]
# ## 11. 结果总结

# %%
print("\n" + "="*60)
print("🎯 问题3：FY1三弹时序策略 - 结果总结")
print("="*60)

print(f"\n📊 最优策略参数:")
print(f"   🚁 无人机速度: {v_opt:.2f} m/s")
print(f"   🧭 飞行方向: {np.degrees(alpha_opt):.1f}°")

print(f"\n💣 三枚烟幕弹配置:")
for i, detail in enumerate(bomb_details):
    if detail['release_pos'] is not None:
        t_r, t_d = bomb_params_opt[i]
        print(f"   第{detail['bomb_id']}枚:")
        print(f"     ⏰ 投放时间: {t_r:.2f} s, 起爆延时: {t_d:.2f} s")
        print(f"     📦 投放位置: ({detail['release_pos'][0]:.0f}, {detail['release_pos'][1]:.0f}, {detail['release_pos'][2]:.0f}) m")
        print(f"     💥 起爆位置: ({detail['burst_pos'][0]:.0f}, {detail['burst_pos'][1]:.0f}, {detail['burst_pos'][2]:.0f}) m")
        print(f"     ⏱️  个体遮蔽: {detail['individual_shielding']:.4f} s")

print(f"\n🎯 性能指标:")
print(f"   ⏱️  总遮蔽时间: {precise_total_shielding:.4f} s")
print(f"   🔍 最佳单弹时间: {best_single_bomb['individual_shielding']:.4f} s")
print(f"   📈 效果提升: {(precise_total_shielding / best_single_bomb['individual_shielding'] - 1) * 100:.1f}%")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_three_bombs.html - 3D轨迹交互图")
print(f"   📊 02_shielding_timeline_analysis.html - 遮蔽时序分析")
print(f"   📋 03_result1.xlsx - 标准格式结果表")
print(f"   ⏰ 04_timeline_analysis.html - 投放时序图")
print(f"   📋 05_results_summary.json - 完整结果汇总")
print(f"   📊 06_detailed_trajectory.csv - 详细轨迹数据")
print(f"   📑 07_complete_results.xlsx - 完整Excel报告")

print(f"\n✅ 问题3求解完成！所有结果已保存到 ImageOutput/03/ 目录")
print("="*60)
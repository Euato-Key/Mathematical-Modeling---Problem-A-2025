# %% [markdown]
# # 问题5：五机多弹综合防御策略优化
# 
# ## 问题描述
# - 无人机：FY1、FY2、FY3、FY4、FY5
# - 投放数量：每架最多3枚烟幕弹
# - 目标导弹：M1、M2、M3
# - 目标：最大化对三枚导弹的总遮蔽时间
# - 输出：策略保存至result3.xlsx

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
output_dir = "../../ImageOutput/05"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题5：五机多弹综合防御策略优化")
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
missile_positions = {
    'M1': np.array([20000.0, 0.0, 2000.0]),
    'M2': np.array([19000.0, 600.0, 2100.0]),
    'M3': np.array([18000.0, -600.0, 1900.0])
}

# 目标参数
target_pos = np.array([0.0, 200.0, 0.0])  # 真目标位置
target_radius = 7.0  # 目标半径 (m)
target_height = 10.0  # 目标高度 (m)

# 无人机参数
drone_positions = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}
v_drone_min = 70.0  # 最小速度 (m/s)
v_drone_max = 140.0  # 最大速度 (m/s)
max_bombs_per_drone = 3  # 每架无人机最多投放弹数

# 计算各导弹单位方向向量（指向假目标原点）
missile_directions = {}
for missile_id, pos in missile_positions.items():
    missile_directions[missile_id] = -pos / np.linalg.norm(pos)

print(f"📍 导弹初始位置:")
for missile_id, pos in missile_positions.items():
    print(f"   {missile_id}: {pos}")
print(f"📍 真目标位置: {target_pos}")
print(f"🚁 参与无人机:")
for drone_id, pos in drone_positions.items():
    print(f"   {drone_id}: {pos}")

# %% [markdown]
# ## 2. 核心计算函数

# %%
def missile_position(t, missile_id):
    """计算导弹在时刻t的位置"""
    initial_pos = missile_positions[missile_id]
    direction = missile_directions[missile_id]
    return initial_pos + v_missile * direction * t

def drone_position(t, drone_id, v_drone, alpha):
    """计算无人机在时刻t的位置"""
    initial_pos = drone_positions[drone_id]
    direction = np.array([np.cos(alpha), np.sin(alpha), 0.0])
    return initial_pos + v_drone * direction * t

def smoke_release_position(t_release, drone_id, v_drone, alpha):
    """计算烟幕弹投放位置"""
    return drone_position(t_release, drone_id, v_drone, alpha)

def smoke_burst_position(t_release, t_burst_delay, drone_id, v_drone, alpha):
    """计算烟幕弹起爆位置"""
    release_pos = smoke_release_position(t_release, drone_id, v_drone, alpha)
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

def evaluate_single_bomb_shielding(t_release, t_burst_delay, drone_id, v_drone, alpha, missile_id, dt=0.02):
    """评估单枚烟幕弹对单枚导弹的遮蔽时间"""
    # 计算起爆位置和时间
    burst_pos = smoke_burst_position(t_release, t_burst_delay, drone_id, v_drone, alpha)
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
        missile_pos = missile_position(t, missile_id)
        
        # 计算距离
        distance = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
        
        # 记录遮蔽状态
        if distance <= R_cloud:
            shielding_intervals.append(t)
    
    # 计算总遮蔽时间
    total_shielding = len(shielding_intervals) * dt
    
    return total_shielding, shielding_intervals

def merge_time_intervals(intervals_list):
    """合并时间区间列表，返回并集的总时长"""
    if not intervals_list:
        return 0.0
    
    # 将所有时间点合并并排序
    all_times = set()
    for intervals in intervals_list:
        all_times.update(intervals)
    
    if not all_times:
        return 0.0
    
    # 计算并集时长
    return len(all_times) * 0.02  # dt = 0.02

def evaluate_comprehensive_strategy(params, dt=0.02, return_details=False):
    """
    评估五机多弹综合策略
    params: 每架无人机2个参数(v_drone, alpha) + 每枚弹3个参数(use_bomb, t_release, t_burst_delay)
    总共: 5*2 + 5*3*3 = 10 + 45 = 55个参数
    """
    # 解析参数
    drone_params = {}
    bomb_params = {}
    
    param_idx = 0
    
    # 解析无人机参数
    for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        v_drone = params[param_idx]
        alpha = params[param_idx + 1]
        drone_params[drone_id] = (v_drone, alpha)
        param_idx += 2
    
    # 解析烟幕弹参数
    for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        bomb_params[drone_id] = []
        for bomb_idx in range(max_bombs_per_drone):
            use_bomb = params[param_idx] > 0.5  # 二值化
            t_release = params[param_idx + 1]
            t_burst_delay = params[param_idx + 2]
            bomb_params[drone_id].append((use_bomb, t_release, t_burst_delay))
            param_idx += 3
    
    # 约束检查
    for drone_id, (v_drone, alpha) in drone_params.items():
        if v_drone < v_drone_min or v_drone > v_drone_max:
            return -1000 if not return_details else (-1000, None)
    
    # 检查投放间隔约束
    for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        release_times = []
        for use_bomb, t_release, t_burst_delay in bomb_params[drone_id]:
            if use_bomb and t_release >= 0 and t_burst_delay >= 0:
                release_times.append(t_release)
        
        release_times.sort()
        for i in range(len(release_times) - 1):
            if release_times[i + 1] - release_times[i] < 1.0:  # 最小间隔1秒
                return -1000 if not return_details else (-1000, None)
    
    # 计算每枚导弹的遮蔽时间
    missile_shielding_intervals = {missile_id: [] for missile_id in ['M1', 'M2', 'M3']}
    bomb_details = []
    
    for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
        v_drone, alpha = drone_params[drone_id]
        
        for bomb_idx, (use_bomb, t_release, t_burst_delay) in enumerate(bomb_params[drone_id]):
            if not use_bomb or t_release < 0 or t_burst_delay < 0:
                if return_details:
                    bomb_details.append({
                        'drone_id': drone_id,
                        'bomb_idx': bomb_idx + 1,
                        'used': False,
                        'release_pos': None,
                        'burst_pos': None,
                        'shielding_by_missile': {}
                    })
                continue
            
            # 计算起爆位置
            burst_pos = smoke_burst_position(t_release, t_burst_delay, drone_id, v_drone, alpha)
            
            # 检查起爆位置
            if burst_pos[2] < 0:
                if return_details:
                    bomb_details.append({
                        'drone_id': drone_id,
                        'bomb_idx': bomb_idx + 1,
                        'used': False,
                        'release_pos': None,
                        'burst_pos': None,
                        'shielding_by_missile': {}
                    })
                continue
            
            # 计算投放位置
            release_pos = smoke_release_position(t_release, drone_id, v_drone, alpha)
            
            # 评估对每枚导弹的遮蔽
            bomb_shielding = {}
            for missile_id in ['M1', 'M2', 'M3']:
                shielding_time, intervals = evaluate_single_bomb_shielding(
                    t_release, t_burst_delay, drone_id, v_drone, alpha, missile_id, dt
                )
                bomb_shielding[missile_id] = shielding_time
                missile_shielding_intervals[missile_id].append(intervals)
            
            if return_details:
                bomb_details.append({
                    'drone_id': drone_id,
                    'bomb_idx': bomb_idx + 1,
                    'used': True,
                    'params': (v_drone, alpha, t_release, t_burst_delay),
                    'release_pos': release_pos,
                    'burst_pos': burst_pos,
                    'shielding_by_missile': bomb_shielding
                })
    
    # 计算每枚导弹的总遮蔽时间（并集）
    total_shielding_by_missile = {}
    for missile_id in ['M1', 'M2', 'M3']:
        total_shielding_by_missile[missile_id] = merge_time_intervals(missile_shielding_intervals[missile_id])
    
    # 总目标函数：三枚导弹遮蔽时间之和
    total_objective = sum(total_shielding_by_missile.values())
    
    # 添加惩罚项：鼓励使用更少的弹药
    total_bombs_used = sum(1 for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5'] 
                          for use_bomb, _, _ in bomb_params[drone_id] if use_bomb)
    
    final_objective = total_objective - 0.01 * total_bombs_used  # 轻微惩罚弹药使用
    
    if return_details:
        return final_objective, {
            'drone_params': drone_params,
            'bomb_details': bomb_details,
            'missile_shielding': total_shielding_by_missile,
            'total_objective': total_objective,
            'bombs_used': total_bombs_used
        }
    else:
        return final_objective

print("✅ 核心计算函数定义完成")

# %% [markdown]
# ## 3. 优化求解

# %%
print("🔍 开始优化求解...")

# 定义优化边界
# 5架无人机 × 2参数 + 5架无人机 × 3枚弹 × 3参数 = 10 + 45 = 55个参数
bounds = []

# 无人机参数边界
for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
    bounds.extend([
        (v_drone_min, v_drone_max),  # 速度
        (0, 2*np.pi),                # 飞行方向角
    ])

# 烟幕弹参数边界
for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
    for bomb_idx in range(max_bombs_per_drone):
        bounds.extend([
            (0, 1),      # 是否使用该弹（0-1之间，后续二值化）
            (0, 40),     # 投放时间
            (0, 25)      # 起爆延时
        ])

print(f"📊 优化问题维度: {len(bounds)}维")
print(f"   - 5架无人机，每架2个参数（速度、方向）")
print(f"   - 5架无人机 × 3枚弹，每弹3个参数（使用、投放时间、起爆延时）")

# 目标函数（最大化遮蔽时间，所以取负值）
def objective_function(params):
    return -evaluate_comprehensive_strategy(params, dt=0.05)

# 使用差分进化算法进行全局优化
print("🎯 使用差分进化算法进行全局优化...")
print("   注意：由于问题复杂度极高，优化过程可能需要较长时间...")

result = differential_evolution(
    objective_function,
    bounds,
    seed=42,
    maxiter=500,
    popsize=40,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    disp=True
)

optimal_params = result.x
optimal_total_objective = -result.fun

print(f"✅ 优化完成！")
print(f"🎯 最优总目标值: {optimal_total_objective:.4f}")

# 用精确方法重新评估最优解
precise_objective, detailed_results = evaluate_comprehensive_strategy(
    optimal_params, dt=0.01, return_details=True
)

print(f"🔍 精确总目标值: {detailed_results['total_objective']:.4f}")
print(f"📊 各导弹遮蔽时间:")
for missile_id, shielding_time in detailed_results['missile_shielding'].items():
    print(f"   {missile_id}: {shielding_time:.4f} s")
print(f"💣 总投放弹数: {detailed_results['bombs_used']}")

# %% [markdown]
# ## 4. 详细结果分析

# %%
print("📈 分析最优策略详细结果...")

print(f"\n🚁 各架无人机配置:")
for drone_id, (v_drone, alpha) in detailed_results['drone_params'].items():
    print(f"   {drone_id}:")
    print(f"     🚁 速度: {v_drone:.2f} m/s, 方向: {np.degrees(alpha):.1f}°")

print(f"\n💣 烟幕弹投放详情:")
used_bombs = [detail for detail in detailed_results['bomb_details'] if detail['used']]
for detail in used_bombs:
    v, alpha, t_r, t_d = detail['params']
    print(f"   {detail['drone_id']}-弹{detail['bomb_idx']}:")
    print(f"     ⏰ 投放时间: {t_r:.2f} s, 起爆延时: {t_d:.2f} s")
    print(f"     📦 投放位置: ({detail['release_pos'][0]:.0f}, {detail['release_pos'][1]:.0f}, {detail['release_pos'][2]:.0f})")
    print(f"     💥 起爆位置: ({detail['burst_pos'][0]:.0f}, {detail['burst_pos'][1]:.0f}, {detail['burst_pos'][2]:.0f})")
    print(f"     🎯 遮蔽效果: M1={detail['shielding_by_missile']['M1']:.3f}s, M2={detail['shielding_by_missile']['M2']:.3f}s, M3={detail['shielding_by_missile']['M3']:.3f}s")

# %% [markdown]
# ## 5. 生成时间序列数据

# %%
print("📈 生成详细时间序列数据...")

# 计算最大时间范围
max_time = 0
for detail in used_bombs:
    v, alpha, t_r, t_d = detail['params']
    burst_time = t_r + t_d
    max_time = max(max_time, burst_time + cloud_duration)

t_max = max_time + 10
time_points = np.arange(0, t_max, 0.02)

# 存储轨迹数据
trajectory_data = []

for t in time_points:
    # 各导弹位置
    missile_data = {}
    for missile_id in ['M1', 'M2', 'M3']:
        missile_pos = missile_position(t, missile_id)
        missile_data[f'{missile_id}_x'] = missile_pos[0]
        missile_data[f'{missile_id}_y'] = missile_pos[1]
        missile_data[f'{missile_id}_z'] = missile_pos[2]
    
    # 各无人机位置
    drone_data = {}
    for drone_id, (v_drone, alpha) in detailed_results['drone_params'].items():
        drone_pos = drone_position(t, drone_id, v_drone, alpha)
        drone_data[f'{drone_id}_x'] = drone_pos[0]
        drone_data[f'{drone_id}_y'] = drone_pos[1]
        drone_data[f'{drone_id}_z'] = drone_pos[2]
    
    # 各云团状态
    cloud_data = {}
    missile_shielded = {'M1': False, 'M2': False, 'M3': False}
    
    for detail in used_bombs:
        v, alpha, t_r, t_d = detail['params']
        t_burst = t_r + t_d
        bomb_key = f"{detail['drone_id']}_B{detail['bomb_idx']}"
        
        if t >= t_burst and t <= t_burst + cloud_duration:
            cloud_pos = cloud_center_position(t, t_burst, detail['burst_pos'])
            if cloud_pos is not None and cloud_pos[2] >= 0:
                cloud_data[f'{bomb_key}_x'] = cloud_pos[0]
                cloud_data[f'{bomb_key}_y'] = cloud_pos[1]
                cloud_data[f'{bomb_key}_z'] = cloud_pos[2]
                
                # 检查对各导弹的遮蔽
                for missile_id in ['M1', 'M2', 'M3']:
                    missile_pos = missile_position(t, missile_id)
                    distance = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
                    if distance <= R_cloud:
                        missile_shielded[missile_id] = True
            else:
                cloud_data[f'{bomb_key}_x'] = np.nan
                cloud_data[f'{bomb_key}_y'] = np.nan
                cloud_data[f'{bomb_key}_z'] = np.nan
        else:
            cloud_data[f'{bomb_key}_x'] = np.nan
            cloud_data[f'{bomb_key}_y'] = np.nan
            cloud_data[f'{bomb_key}_z'] = np.nan
    
    # 合并数据
    row_data = {'time': t}
    row_data.update(missile_data)
    row_data.update(drone_data)
    row_data.update(cloud_data)
    row_data.update({f'{missile_id}_shielded': shielded for missile_id, shielded in missile_shielded.items()})
    
    trajectory_data.append(row_data)

trajectory_df = pd.DataFrame(trajectory_data)

print(f"✅ 生成了 {len(trajectory_df)} 个时间点的轨迹数据")

# %% [markdown]
# ## 6. 3D轨迹可视化

# %%
print("🎨 创建3D轨迹可视化...")

fig_3d = go.Figure()

# 导弹轨迹
missile_colors = ['red', 'orange', 'darkred']
for i, missile_id in enumerate(['M1', 'M2', 'M3']):
    fig_3d.add_trace(go.Scatter3d(
        x=trajectory_df[f'{missile_id}_x'],
        y=trajectory_df[f'{missile_id}_y'],
        z=trajectory_df[f'{missile_id}_z'],
        mode='lines+markers',
        line=dict(color=missile_colors[i], width=6),
        marker=dict(size=3, color=missile_colors[i]),
        name=f'导弹{missile_id}轨迹',
        hovertemplate=f'<b>导弹{missile_id}</b><br>' +
                      'X: %{x:.0f}m<br>' +
                      'Y: %{y:.0f}m<br>' +
                      'Z: %{z:.0f}m<br>' +
                      '<extra></extra>'
    ))

# 无人机轨迹
drone_colors = ['blue', 'green', 'purple', 'brown', 'pink']
for i, drone_id in enumerate(['FY1', 'FY2', 'FY3', 'FY4', 'FY5']):
    fig_3d.add_trace(go.Scatter3d(
        x=trajectory_df[f'{drone_id}_x'],
        y=trajectory_df[f'{drone_id}_y'],
        z=trajectory_df[f'{drone_id}_z'],
        mode='lines+markers',
        line=dict(color=drone_colors[i], width=4),
        marker=dict(size=2, color=drone_colors[i]),
        name=f'无人机{drone_id}轨迹',
        hovertemplate=f'<b>无人机{drone_id}</b><br>' +
                      'X: %{x:.0f}m<br>' +
                      'Y: %{y:.0f}m<br>' +
                      'Z: %{z:.0f}m<br>' +
                      '<extra></extra>'
    ))

# 云团轨迹
cloud_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
color_idx = 0
for detail in used_bombs:
    bomb_key = f"{detail['drone_id']}_B{detail['bomb_idx']}"
    cloud_mask = ~trajectory_df[f'{bomb_key}_x'].isna()
    
    if cloud_mask.any():
        fig_3d.add_trace(go.Scatter3d(
            x=trajectory_df.loc[cloud_mask, f'{bomb_key}_x'],
            y=trajectory_df.loc[cloud_mask, f'{bomb_key}_y'],
            z=trajectory_df.loc[cloud_mask, f'{bomb_key}_z'],
            mode='lines+markers',
            line=dict(color=cloud_colors[color_idx % len(cloud_colors)], width=6, dash='dash'),
            marker=dict(size=4, color=cloud_colors[color_idx % len(cloud_colors)], opacity=0.7),
            name=f'{bomb_key}云团',
            hovertemplate=f'<b>{bomb_key}云团</b><br>' +
                          'X: %{x:.0f}m<br>' +
                          'Y: %{y:.0f}m<br>' +
                          'Z: %{z:.0f}m<br>' +
                          '<extra></extra>'
        ))
        color_idx += 1

# 关键位置标记
# 导弹初始位置
missile_initial_x = [missile_positions[missile_id][0] for missile_id in ['M1', 'M2', 'M3']]
missile_initial_y = [missile_positions[missile_id][1] for missile_id in ['M1', 'M2', 'M3']]
missile_initial_z = [missile_positions[missile_id][2] for missile_id in ['M1', 'M2', 'M3']]
missile_labels = [f'导弹{missile_id}起点' for missile_id in ['M1', 'M2', 'M3']]

fig_3d.add_trace(go.Scatter3d(
    x=missile_initial_x,
    y=missile_initial_y,
    z=missile_initial_z,
    mode='markers',
    marker=dict(size=12, color=missile_colors, symbol='diamond'),
    name='导弹起点',
    text=missile_labels,
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 无人机初始位置
drone_initial_x = [drone_positions[drone_id][0] for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']]
drone_initial_y = [drone_positions[drone_id][1] for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']]
drone_initial_z = [drone_positions[drone_id][2] for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']]
drone_labels = [f'无人机{drone_id}起点' for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']]

fig_3d.add_trace(go.Scatter3d(
    x=drone_initial_x,
    y=drone_initial_y,
    z=drone_initial_z,
    mode='markers',
    marker=dict(size=10, color=drone_colors, symbol='diamond'),
    name='无人机起点',
    text=drone_labels,
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

# 设置布局
fig_3d.update_layout(
    title=dict(
        text='问题5：五机多弹综合防御策略 - 3D轨迹图',
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
        aspectratio=dict(x=3, y=2, z=1)
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)"
    ),
    width=1400,
    height=900
)

# 保存3D图
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_comprehensive.html")
fig_3d.write_image(f"{output_dir}/01_3d_trajectory_comprehensive.svg")
fig_3d.show()

print("✅ 3D轨迹图已保存")

# %% [markdown]
# ## 7. 综合遮蔽效果分析

# %%
print("📊 创建综合遮蔽效果分析图...")

# 创建子图
fig_analysis = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        '各导弹遮蔽状态时间序列',
        '各导弹累积遮蔽时间',
        '投放弹数统计',
        '遮蔽效果对比'
    ],
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"type": "bar"}],
           [{"type": "bar"}]]
)

# 各导弹遮蔽状态
for i, missile_id in enumerate(['M1', 'M2', 'M3']):
    shielding_status = trajectory_df[f'{missile_id}_shielded'].astype(int)
    
    fig_analysis.add_trace(
        go.Scatter(
            x=trajectory_df['time'],
            y=shielding_status + i * 0.1,  # 稍微错开显示
            mode='lines+markers',
            line=dict(color=missile_colors[i], width=3),
            marker=dict(size=3),
            name=f'{missile_id}遮蔽状态',
            hovertemplate=f'时间: %{{x:.2f}}s<br>{missile_id}遮蔽: %{{text}}<extra></extra>',
            text=['是' if x else '否' for x in trajectory_df[f'{missile_id}_shielded']]
        ),
        row=1, col=1
    )

# 累积遮蔽时间
for i, missile_id in enumerate(['M1', 'M2', 'M3']):
    shielding_status = trajectory_df[f'{missile_id}_shielded'].astype(int)
    cumulative_shielding = np.cumsum(shielding_status) * 0.02  # dt = 0.02
    
    fig_analysis.add_trace(
        go.Scatter(
            x=trajectory_df['time'],
            y=cumulative_shielding,
            mode='lines',
            line=dict(color=missile_colors[i], width=4),
            name=f'{missile_id}累积遮蔽',
            hovertemplate=f'时间: %{{x:.2f}}s<br>{missile_id}累积: %{{y:.3f}}s<extra></extra>'
        ),
        row=2, col=1
    )

# 各无人机投放弹数统计
drone_bomb_counts = {}
for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
    count = sum(1 for detail in detailed_results['bomb_details'] 
                if detail['drone_id'] == drone_id and detail['used'])
    drone_bomb_counts[drone_id] = count

fig_analysis.add_trace(
    go.Bar(
        x=list(drone_bomb_counts.keys()),
        y=list(drone_bomb_counts.values()),
        marker_color=drone_colors,
        name='投放弹数',
        text=list(drone_bomb_counts.values()),
        textposition='auto'
    ),
    row=3, col=1
)

# 各导弹遮蔽效果对比
missile_shielding_times = [detailed_results['missile_shielding'][missile_id] 
                          for missile_id in ['M1', 'M2', 'M3']]

fig_analysis.add_trace(
    go.Bar(
        x=['M1', 'M2', 'M3'],
        y=missile_shielding_times,
        marker_color=missile_colors,
        name='遮蔽时间',
        text=[f'{t:.3f}s' for t in missile_shielding_times],
        textposition='auto'
    ),
    row=4, col=1
)

# 更新布局
fig_analysis.update_layout(
    title=dict(
        text='问题5：五机多弹综合防御效果分析',
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
fig_analysis.update_xaxes(title_text="时间 (s)", row=2, col=1)
fig_analysis.update_yaxes(title_text="遮蔽状态", row=1, col=1)
fig_analysis.update_yaxes(title_text="累积时间 (s)", row=2, col=1)
fig_analysis.update_yaxes(title_text="投放弹数", row=3, col=1)
fig_analysis.update_yaxes(title_text="遮蔽时间 (s)", row=4, col=1)

# 保存分析图
fig_analysis.write_html(f"{output_dir}/02_comprehensive_analysis.html")
fig_analysis.write_image(f"{output_dir}/02_comprehensive_analysis.svg")
fig_analysis.show()

print("✅ 综合遮蔽效果分析图已保存")

# %% [markdown]
# ## 8. 生成result3.xlsx格式结果

# %%
print("📋 生成result3.xlsx格式结果...")

# 准备result3.xlsx格式的数据
result3_data = []

for detail in detailed_results['bomb_details']:
    if detail['used']:
        v, alpha, t_r, t_d = detail['params']
        
        # 转换角度为度数（0-360度，x轴正向逆时针为正）
        direction_deg = np.degrees(alpha)
        if direction_deg < 0:
            direction_deg += 360
        
        # 找到主要干扰的导弹（遮蔽时间最长的）
        max_shielding = 0
        primary_missile = 'M1'
        for missile_id, shielding_time in detail['shielding_by_missile'].items():
            if shielding_time > max_shielding:
                max_shielding = shielding_time
                primary_missile = missile_id
        
        row = {
            '无人机编号': detail['drone_id'],
            '无人机运动方向': direction_deg,
            '无人机运动速度 (m/s)': v,
            '烟幕干扰弹编号': detail['bomb_idx'],
            '烟幕干扰弹投放点的x坐标 (m)': detail['release_pos'][0],
            '烟幕干扰弹投放点的y坐标 (m)': detail['release_pos'][1],
            '烟幕干扰弹投放点的z坐标 (m)': detail['release_pos'][2],
            '烟幕干扰弹起爆点的x坐标 (m)': detail['burst_pos'][0],
            '烟幕干扰弹起爆点的y坐标 (m)': detail['burst_pos'][1],
            '烟幕干扰弹起爆点的z坐标 (m)': detail['burst_pos'][2],
            '有效干扰时长 (s)': max_shielding,
            '干扰的导弹编号': primary_missile
        }
        result3_data.append(row)

# 创建DataFrame
result3_df = pd.DataFrame(result3_data)

# 保存为Excel文件
result3_df.to_excel(f"{output_dir}/03_result3.xlsx", index=False)

print("✅ result3.xlsx格式文件已生成")
print(f"\n📊 result3.xlsx内容预览 (共{len(result3_df)}枚弹):")
if len(result3_df) > 0:
    print(result3_df.to_string(index=False))
else:
    print("   无有效投放的烟幕弹")

# %% [markdown]
# ## 9. 策略效果评估

# %%
print("🔬 进行策略效果评估...")

# 计算总体效果指标
total_shielding_time = sum(detailed_results['missile_shielding'].values())
average_shielding_per_missile = total_shielding_time / 3
bombs_efficiency = total_shielding_time / detailed_results['bombs_used'] if detailed_results['bombs_used'] > 0 else 0

print(f"\n📈 总体效果评估:")
print(f"   ⏱️  总遮蔽时间: {total_shielding_time:.4f} s")
print(f"   📊 平均每导弹遮蔽: {average_shielding_per_missile:.4f} s")
print(f"   💣 总投放弹数: {detailed_results['bombs_used']}")
print(f"   📈 弹药效率: {bombs_efficiency:.4f} s/弹")

# 各无人机贡献分析
print(f"\n🚁 各无人机贡献分析:")
drone_contributions = {}
for drone_id in ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']:
    drone_bombs = [detail for detail in detailed_results['bomb_details'] 
                   if detail['drone_id'] == drone_id and detail['used']]
    
    total_contribution = 0
    for detail in drone_bombs:
        total_contribution += sum(detail['shielding_by_missile'].values())
    
    drone_contributions[drone_id] = {
        'bombs_used': len(drone_bombs),
        'total_contribution': total_contribution,
        'efficiency': total_contribution / len(drone_bombs) if len(drone_bombs) > 0 else 0
    }
    
    print(f"   {drone_id}: {len(drone_bombs)}弹, 贡献{total_contribution:.3f}s, 效率{drone_contributions[drone_id]['efficiency']:.3f}s/弹")

# 创建策略效果评估图
fig_evaluation = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '各导弹遮蔽时间分布',
        '各无人机投放弹数',
        '各无人机贡献效率',
        '总体效果指标'
    ],
    specs=[[{"type": "pie"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "indicator"}]]
)

# 各导弹遮蔽时间饼图
fig_evaluation.add_trace(
    go.Pie(
        labels=['M1', 'M2', 'M3'],
        values=[detailed_results['missile_shielding'][missile_id] for missile_id in ['M1', 'M2', 'M3']],
        marker_colors=missile_colors,
        name='导弹遮蔽分布'
    ),
    row=1, col=1
)

# 各无人机投放弹数
drone_ids = list(drone_contributions.keys())
bombs_counts = [drone_contributions[drone_id]['bombs_used'] for drone_id in drone_ids]

fig_evaluation.add_trace(
    go.Bar(
        x=drone_ids,
        y=bombs_counts,
        marker_color=drone_colors,
        name='投放弹数',
        text=bombs_counts,
        textposition='auto'
    ),
    row=1, col=2
)

# 各无人机贡献效率
efficiencies = [drone_contributions[drone_id]['efficiency'] for drone_id in drone_ids]

fig_evaluation.add_trace(
    go.Bar(
        x=drone_ids,
        y=efficiencies,
        marker_color=drone_colors,
        name='贡献效率',
        text=[f'{e:.3f}' for e in efficiencies],
        textposition='auto'
    ),
    row=2, col=1
)

# 总体效果指示器
fig_evaluation.add_trace(
    go.Indicator(
        mode="gauge+number+delta",
        value=total_shielding_time,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "总遮蔽时间 (s)"},
        delta={'reference': 20},  # 参考值
        gauge={
            'axis': {'range': [None, 30]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgray"},
                {'range': [10, 20], 'color': "gray"},
                {'range': [20, 30], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ),
    row=2, col=2
)

fig_evaluation.update_layout(
    title='问题5：五机多弹策略效果评估',
    height=800,
    showlegend=False
)

fig_evaluation.write_html(f"{output_dir}/04_strategy_evaluation.html")
fig_evaluation.write_image(f"{output_dir}/04_strategy_evaluation.svg")
fig_evaluation.show()

print("✅ 策略效果评估完成")

# %% [markdown]
# ## 10. 结果汇总与保存

# %%
print("💾 保存完整结果数据...")

# 汇总结果
results_summary = {
    'problem': '问题5：五机多弹综合防御策略',
    'optimization_method': '差分进化算法',
    'drones_configuration': {},
    'bombs_details': [],
    'performance': {
        'total_shielding_time_s': float(total_shielding_time),
        'missile_shielding_times': {k: float(v) for k, v in detailed_results['missile_shielding'].items()},
        'total_bombs_used': int(detailed_results['bombs_used']),
        'bombs_efficiency_s_per_bomb': float(bombs_efficiency),
        'average_shielding_per_missile_s': float(average_shielding_per_missile)
    },
    'constraints': {
        'drone_speed_range_ms': [v_drone_min, v_drone_max],
        'max_bombs_per_drone': max_bombs_per_drone,
        'cloud_radius_m': R_cloud,
        'cloud_duration_s': cloud_duration,
        'min_release_interval_s': 1.0
    }
}

# 添加无人机配置信息
for drone_id, (v_drone, alpha) in detailed_results['drone_params'].items():
    results_summary['drones_configuration'][drone_id] = {
        'speed_ms': float(v_drone),
        'direction_rad': float(alpha),
        'direction_deg': float(np.degrees(alpha))
    }

# 添加烟幕弹详细信息
for detail in detailed_results['bomb_details']:
    if detail['used']:
        v, alpha, t_r, t_d = detail['params']
        bomb_info = {
            'drone_id': detail['drone_id'],
            'bomb_index': detail['bomb_idx'],
            'release_time_s': float(t_r),
            'burst_delay_s': float(t_d),
            'release_position': detail['release_pos'].tolist(),
            'burst_position': detail['burst_pos'].tolist(),
            'shielding_by_missile': {k: float(v) for k, v in detail['shielding_by_missile'].items()}
        }
        results_summary['bombs_details'].append(bomb_info)

# 保存JSON结果
with open(f"{output_dir}/05_results_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存详细轨迹数据
trajectory_df.to_csv(f"{output_dir}/06_detailed_trajectory.csv", index=False)

# 创建完整的Excel报告
with pd.ExcelWriter(f"{output_dir}/07_complete_results.xlsx", engine='openpyxl') as writer:
    # result3格式表
    result3_df.to_excel(writer, sheet_name='result3', index=False)
    
    # 汇总表
    summary_data = []
    for missile_id in ['M1', 'M2', 'M3']:
        summary_data.append([
            missile_id,
            f"{detailed_results['missile_shielding'][missile_id]:.6f}",
            f"{detailed_results['missile_shielding'][missile_id] / total_shielding_time * 100:.1f}%" if total_shielding_time > 0 else "0%"
        ])
    
    summary_data.append(['总计', f"{total_shielding_time:.6f}", "100%"])
    summary_data.append(['投放弹数', str(detailed_results['bombs_used']), ""])
    summary_data.append(['弹药效率', f"{bombs_efficiency:.4f} s/弹", ""])
    
    summary_df = pd.DataFrame(summary_data, columns=['项目', '数值', '占比'])
    summary_df.to_excel(writer, sheet_name='summary', index=False)
    
    # 无人机配置表
    config_data = []
    for drone_id, (v_drone, alpha) in detailed_results['drone_params'].items():
        config_data.append([
            drone_id,
            f"{v_drone:.2f}",
            f"{np.degrees(alpha):.2f}",
            drone_contributions[drone_id]['bombs_used'],
            f"{drone_contributions[drone_id]['total_contribution']:.4f}",
            f"{drone_contributions[drone_id]['efficiency']:.4f}"
        ])
    
    config_df = pd.DataFrame(config_data, columns=[
        '无人机编号', '速度 (m/s)', '方向 (度)', '投放弹数', '总贡献 (s)', '效率 (s/弹)'
    ])
    config_df.to_excel(writer, sheet_name='无人机配置', index=False)
    
    # 遮蔽区间表
    intervals_data = []
    for detail in detailed_results['bomb_details']:
        if detail['used']:
            v, alpha, t_r, t_d = detail['params']
            t_burst = t_r + t_d
            for missile_id, shielding_time in detail['shielding_by_missile'].items():
                if shielding_time > 0:
                    intervals_data.append([
                        f"{detail['drone_id']}-弹{detail['bomb_idx']}",
                        missile_id,
                        f"{t_burst:.2f}",
                        f"{t_burst + cloud_duration:.2f}",
                        f"{shielding_time:.4f}"
                    ])
    
    if intervals_data:
        intervals_df = pd.DataFrame(intervals_data, columns=[
            '烟幕弹', '导弹', '起爆时间 (s)', '结束时间 (s)', '遮蔽时长 (s)'
        ])
        intervals_df.to_excel(writer, sheet_name='遮蔽区间', index=False)

print("✅ 所有结果已保存到 ImageOutput/05/ 目录")

# %% [markdown]
# ## 11. 结果总结

# %%
print("\n" + "="*60)
print("🎯 问题5：五机多弹综合防御策略 - 结果总结")
print("="*60)

print(f"\n🚁 五架无人机最优配置:")
for drone_id, (v_drone, alpha) in detailed_results['drone_params'].items():
    bombs_count = drone_contributions[drone_id]['bombs_used']
    contribution = drone_contributions[drone_id]['total_contribution']
    print(f"   {drone_id}: 速度{v_drone:.1f}m/s, 方向{np.degrees(alpha):.0f}°, 投放{bombs_count}弹, 贡献{contribution:.3f}s")

print(f"\n💣 烟幕弹投放详情:")
for detail in used_bombs:
    v, alpha, t_r, t_d = detail['params']
    max_shielding = max(detail['shielding_by_missile'].values())
    primary_missile = max(detail['shielding_by_missile'], key=detail['shielding_by_missile'].get)
    print(f"   {detail['drone_id']}-弹{detail['bomb_idx']}: 投放{t_r:.1f}s, 延时{t_d:.1f}s, 主干扰{primary_missile}({max_shielding:.3f}s)")

print(f"\n🎯 防御效果分析:")
print(f"   🚀 M1遮蔽时间: {detailed_results['missile_shielding']['M1']:.4f} s")
print(f"   🚀 M2遮蔽时间: {detailed_results['missile_shielding']['M2']:.4f} s")
print(f"   🚀 M3遮蔽时间: {detailed_results['missile_shielding']['M3']:.4f} s")
print(f"   ⏱️  总遮蔽时间: {total_shielding_time:.4f} s")
print(f"   💣 总投放弹数: {detailed_results['bombs_used']}")
print(f"   📈 弹药效率: {bombs_efficiency:.4f} s/弹")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_comprehensive.html - 3D轨迹交互图")
print(f"   📊 02_comprehensive_analysis.html - 综合遮蔽分析")
print(f"   📋 03_result3.xlsx - 标准格式结果表")
print(f"   🔬 04_strategy_evaluation.html - 策略效果评估")
print(f"   📋 05_results_summary.json - 完整结果汇总")
print(f"   📊 06_detailed_trajectory.csv - 详细轨迹数据")
print(f"   📑 07_complete_results.xlsx - 完整Excel报告")

print(f"\n✅ 问题5求解完成！所有结果已保存到 ImageOutput/05/ 目录")
print("="*60)
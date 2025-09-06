# %% [markdown]
# # 问题2：单弹最优投放策略优化
# 
# ## 问题描述
# - 无人机：FY1
# - 需优化：飞行方向、速度、投放点、起爆点
# - 目标：最大化对M1的遮蔽时间
# - 约束：速度70-140m/s，等高度飞行

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
output_dir = "../../ImageOutput/02"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题2：单弹最优投放策略优化")
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
FY1_initial = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置
v_drone_min = 70.0  # 最小速度 (m/s)
v_drone_max = 140.0  # 最大速度 (m/s)

# 计算导弹单位方向向量（指向假目标原点）
missile_direction = -M1_initial / np.linalg.norm(M1_initial)

print(f"📍 导弹M1初始位置: {M1_initial}")
print(f"📍 无人机FY1初始位置: {FY1_initial}")
print(f"📍 真目标位置: {target_pos}")
print(f"🎯 导弹飞行方向: {missile_direction}")

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

def evaluate_shielding_time(params, dt=0.02, smooth=False, kappa=50):
    """
    评估遮蔽时间
    params: [v_drone, alpha, t_release, t_burst_delay]
    """
    v_drone, alpha, t_release, t_burst_delay = params
    
    # 约束检查
    if v_drone < v_drone_min or v_drone > v_drone_max:
        return -1000
    if t_release < 0 or t_burst_delay < 0:
        return -1000
    
    # 计算起爆位置和时间
    burst_pos = smoke_burst_position(t_release, t_burst_delay, v_drone, alpha)
    t_burst = t_release + t_burst_delay
    
    # 检查起爆位置是否合理（不能在地面以下）
    if burst_pos[2] < 0:
        return -1000
    
    # 计算遮蔽时间
    total_shielding = 0.0
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
        
        if smooth:
            # 平滑目标函数（用于优化）
            shielding_factor = 1.0 / (1.0 + np.exp(kappa * (distance - R_cloud)))
            total_shielding += shielding_factor * dt
        else:
            # 硬阈值（用于精确评估）
            if distance <= R_cloud:
                total_shielding += dt
    
    return total_shielding

print("✅ 核心计算函数定义完成")

# %% [markdown]
# ## 3. 优化求解

# %%
print("🔍 开始优化求解...")

# 定义优化边界
# [v_drone, alpha, t_release, t_burst_delay]
bounds = [
    (v_drone_min, v_drone_max),  # 无人机速度
    (0, 2*np.pi),                # 飞行方向角
    (0, 30),                     # 投放时间
    (0, 20)                      # 起爆延时
]

# 目标函数（最大化遮蔽时间，所以取负值）
def objective_function(params):
    return -evaluate_shielding_time(params, dt=0.05, smooth=True, kappa=30)

# 使用差分进化算法进行全局优化
print("🎯 使用差分进化算法进行全局优化...")
result = differential_evolution(
    objective_function,
    bounds,
    seed=42,
    maxiter=200,
    popsize=20,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    disp=True
)

optimal_params = result.x
optimal_shielding_time = -result.fun

print(f"✅ 优化完成！")
print(f"📊 最优参数:")
print(f"   - 无人机速度: {optimal_params[0]:.2f} m/s")
print(f"   - 飞行方向角: {optimal_params[1]:.4f} rad ({np.degrees(optimal_params[1]):.2f}°)")
print(f"   - 投放时间: {optimal_params[2]:.2f} s")
print(f"   - 起爆延时: {optimal_params[3]:.2f} s")
print(f"🎯 最大遮蔽时间: {optimal_shielding_time:.4f} s")

# 用精确方法重新评估最优解
precise_shielding_time = evaluate_shielding_time(optimal_params, dt=0.01, smooth=False)
print(f"🔍 精确遮蔽时间: {precise_shielding_time:.4f} s")

# %% [markdown]
# ## 4. 详细轨迹分析

# %%
print("📈 生成详细轨迹数据...")

v_opt, alpha_opt, t_release_opt, t_burst_delay_opt = optimal_params
t_burst_opt = t_release_opt + t_burst_delay_opt

# 计算关键位置
release_pos = smoke_release_position(t_release_opt, v_opt, alpha_opt)
burst_pos = smoke_burst_position(t_release_opt, t_burst_delay_opt, v_opt, alpha_opt)

print(f"📍 投放位置: ({release_pos[0]:.1f}, {release_pos[1]:.1f}, {release_pos[2]:.1f})")
print(f"💥 起爆位置: ({burst_pos[0]:.1f}, {burst_pos[1]:.1f}, {burst_pos[2]:.1f})")
print(f"⏰ 起爆时间: {t_burst_opt:.2f} s")

# 生成时间序列数据
t_max = t_burst_opt + cloud_duration + 5
time_points = np.arange(0, t_max, 0.02)

# 存储轨迹数据
trajectory_data = []
shielding_data = []

for t in time_points:
    # 导弹位置
    missile_pos = missile_position(t)
    
    # 无人机位置
    drone_pos = drone_position(t, v_opt, alpha_opt)
    
    # 云团位置（如果已起爆）
    cloud_pos = None
    distance_to_line = np.inf
    is_shielded = False
    
    if t >= t_burst_opt and t <= t_burst_opt + cloud_duration:
        cloud_pos = cloud_center_position(t, t_burst_opt, burst_pos)
        if cloud_pos is not None and cloud_pos[2] >= 0:
            distance_to_line = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
            is_shielded = distance_to_line <= R_cloud
    
    trajectory_data.append({
        'time': t,
        'missile_x': missile_pos[0],
        'missile_y': missile_pos[1],
        'missile_z': missile_pos[2],
        'drone_x': drone_pos[0],
        'drone_y': drone_pos[1],
        'drone_z': drone_pos[2],
        'cloud_x': cloud_pos[0] if cloud_pos is not None else np.nan,
        'cloud_y': cloud_pos[1] if cloud_pos is not None else np.nan,
        'cloud_z': cloud_pos[2] if cloud_pos is not None else np.nan,
        'distance_to_line': distance_to_line if distance_to_line != np.inf else np.nan,
        'is_shielded': is_shielded
    })

trajectory_df = pd.DataFrame(trajectory_data)

print(f"✅ 生成了 {len(trajectory_df)} 个时间点的轨迹数据")

# %% [markdown]
# ## 5. 3D轨迹可视化

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

# 云团轨迹（仅显示有效时间内）
cloud_mask = ~trajectory_df['cloud_x'].isna()
if cloud_mask.any():
    fig_3d.add_trace(go.Scatter3d(
        x=trajectory_df.loc[cloud_mask, 'cloud_x'],
        y=trajectory_df.loc[cloud_mask, 'cloud_y'],
        z=trajectory_df.loc[cloud_mask, 'cloud_z'],
        mode='lines+markers',
        line=dict(color='gray', width=8, dash='dash'),
        marker=dict(size=4, color='gray', opacity=0.7),
        name='云团中心轨迹',
        hovertemplate='<b>云团中心</b><br>' +
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
fig_3d.add_trace(go.Scatter3d(
    x=[release_pos[0], burst_pos[0]],
    y=[release_pos[1], burst_pos[1]],
    z=[release_pos[2], burst_pos[2]],
    mode='markers',
    marker=dict(size=12, color=['green', 'orange'], symbol='x'),
    name='关键事件',
    text=['烟幕弹投放', '烟幕弹起爆'],
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 设置布局
fig_3d.update_layout(
    title=dict(
        text='问题2：最优投放策略 - 3D轨迹图',
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
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_optimal.html")
fig_3d.write_image(f"{output_dir}/01_3d_trajectory_optimal.svg")
fig_3d.show()

print("✅ 3D轨迹图已保存")

# %% [markdown]
# ## 6. 遮蔽效果分析

# %%
print("📊 创建遮蔽效果分析图...")

# 创建子图
fig_analysis = make_subplots(
    rows=3, cols=1,
    subplot_titles=[
        '云团到导弹-目标连线的距离',
        '遮蔽状态时间序列',
        '累积遮蔽时间'
    ],
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]]
)

# 过滤有效数据
valid_mask = ~trajectory_df['distance_to_line'].isna()
valid_data = trajectory_df[valid_mask].copy()

if len(valid_data) > 0:
    # 距离曲线
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=valid_data['distance_to_line'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='距离',
            hovertemplate='时间: %{x:.2f}s<br>距离: %{y:.2f}m<extra></extra>'
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
    
    # 遮蔽状态
    shielding_status = valid_data['is_shielded'].astype(int)
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=shielding_status,
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=4),
            name='遮蔽状态',
            hovertemplate='时间: %{x:.2f}s<br>遮蔽: %{text}<extra></extra>',
            text=['是' if x else '否' for x in valid_data['is_shielded']]
        ),
        row=2, col=1
    )
    
    # 累积遮蔽时间
    cumulative_shielding = np.cumsum(shielding_status) * 0.02  # dt = 0.02
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=cumulative_shielding,
            mode='lines',
            line=dict(color='purple', width=3),
            name='累积遮蔽时间',
            hovertemplate='时间: %{x:.2f}s<br>累积遮蔽: %{y:.3f}s<extra></extra>'
        ),
        row=3, col=1
    )

# 更新布局
fig_analysis.update_layout(
    title=dict(
        text='问题2：最优策略遮蔽效果分析',
        x=0.5,
        font=dict(size=18, color='darkblue')
    ),
    height=900,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# 更新坐标轴
fig_analysis.update_xaxes(title_text="时间 (s)", row=3, col=1)
fig_analysis.update_yaxes(title_text="距离 (m)", row=1, col=1)
fig_analysis.update_yaxes(title_text="遮蔽状态", row=2, col=1)
fig_analysis.update_yaxes(title_text="累积时间 (s)", row=3, col=1)

# 保存分析图
fig_analysis.write_html(f"{output_dir}/02_shielding_analysis.html")
fig_analysis.write_image(f"{output_dir}/02_shielding_analysis.svg")
fig_analysis.show()

print("✅ 遮蔽效果分析图已保存")

# %% [markdown]
# ## 7. 参数敏感性分析

# %%
print("🔬 进行参数敏感性分析...")

# 定义参数变化范围
param_names = ['速度 (m/s)', '方向角 (rad)', '投放时间 (s)', '起爆延时 (s)']
param_ranges = [
    np.linspace(70, 140, 15),
    np.linspace(0, 2*np.pi, 20),
    np.linspace(0, 20, 15),
    np.linspace(0, 15, 15)
]

sensitivity_results = []

for i, (param_name, param_range) in enumerate(zip(param_names, param_ranges)):
    print(f"  分析参数: {param_name}")
    
    param_shielding = []
    for param_value in param_range:
        # 创建测试参数组合
        test_params = optimal_params.copy()
        test_params[i] = param_value
        
        # 评估遮蔽时间
        shielding_time = evaluate_shielding_time(test_params, dt=0.05, smooth=False)
        param_shielding.append(max(0, shielding_time))  # 确保非负
    
    sensitivity_results.append({
        'param_name': param_name,
        'param_values': param_range,
        'shielding_times': param_shielding,
        'optimal_value': optimal_params[i],
        'optimal_shielding': precise_shielding_time
    })

# 创建敏感性分析图
fig_sensitivity = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{result["param_name"]}敏感性' for result in sensitivity_results],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

positions = [(1,1), (1,2), (2,1), (2,2)]

for idx, (result, pos) in enumerate(zip(sensitivity_results, positions)):
    fig_sensitivity.add_trace(
        go.Scatter(
            x=result['param_values'],
            y=result['shielding_times'],
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=6),
            name=result['param_name'],
            hovertemplate=f'{result["param_name"]}: %{{x:.3f}}<br>遮蔽时间: %{{y:.4f}}s<extra></extra>'
        ),
        row=pos[0], col=pos[1]
    )
    
    # 标记最优值
    fig_sensitivity.add_vline(
        x=result['optimal_value'],
        line_dash="dash",
        line_color="red",
        annotation_text="最优值",
        row=pos[0], col=pos[1]
    )

fig_sensitivity.update_layout(
    title=dict(
        text='问题2：参数敏感性分析',
        x=0.5,
        font=dict(size=18, color='darkblue')
    ),
    height=800,
    showlegend=False
)

# 更新坐标轴标签
fig_sensitivity.update_yaxes(title_text="遮蔽时间 (s)")

# 保存敏感性分析图
fig_sensitivity.write_html(f"{output_dir}/03_sensitivity_analysis.html")
fig_sensitivity.write_image(f"{output_dir}/03_sensitivity_analysis.svg")
fig_sensitivity.show()

print("✅ 参数敏感性分析完成")

# %% [markdown]
# ## 8. 结果汇总与保存

# %%
print("💾 保存结果数据...")

# 汇总结果
results_summary = {
    'problem': '问题2：单弹最优投放策略',
    'optimization_method': '差分进化算法',
    'optimal_parameters': {
        'drone_speed_ms': float(optimal_params[0]),
        'flight_direction_rad': float(optimal_params[1]),
        'flight_direction_deg': float(np.degrees(optimal_params[1])),
        'release_time_s': float(optimal_params[2]),
        'burst_delay_s': float(optimal_params[3])
    },
    'key_positions': {
        'release_position': release_pos.tolist(),
        'burst_position': burst_pos.tolist(),
        'burst_time_s': float(t_burst_opt)
    },
    'performance': {
        'max_shielding_time_s': float(precise_shielding_time),
        'optimization_shielding_time_s': float(optimal_shielding_time)
    },
    'constraints': {
        'drone_speed_range_ms': [v_drone_min, v_drone_max],
        'cloud_radius_m': R_cloud,
        'cloud_duration_s': cloud_duration
    }
}

# 保存JSON结果
with open(f"{output_dir}/04_results_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存详细轨迹数据
trajectory_df.to_csv(f"{output_dir}/05_detailed_trajectory.csv", index=False)

# 创建Excel结果表
excel_data = {
    '最优参数': [
        '无人机速度 (m/s)',
        '飞行方向角 (rad)',
        '飞行方向角 (度)',
        '投放时间 (s)',
        '起爆延时 (s)',
        '起爆时间 (s)'
    ],
    '数值': [
        f"{optimal_params[0]:.3f}",
        f"{optimal_params[1]:.6f}",
        f"{np.degrees(optimal_params[1]):.2f}",
        f"{optimal_params[2]:.3f}",
        f"{optimal_params[3]:.3f}",
        f"{t_burst_opt:.3f}"
    ]
}

excel_df = pd.DataFrame(excel_data)

# 添加关键位置信息
position_data = pd.DataFrame({
    '关键位置': ['投放位置X (m)', '投放位置Y (m)', '投放位置Z (m)',
                '起爆位置X (m)', '起爆位置Y (m)', '起爆位置Z (m)'],
    '坐标值': [f"{release_pos[0]:.1f}", f"{release_pos[1]:.1f}", f"{release_pos[2]:.1f}",
              f"{burst_pos[0]:.1f}", f"{burst_pos[1]:.1f}", f"{burst_pos[2]:.1f}"]
})

# 添加性能指标
performance_data = pd.DataFrame({
    '性能指标': ['最大遮蔽时间 (s)', '优化目标值', '计算精度'],
    '结果': [f"{precise_shielding_time:.6f}", f"{optimal_shielding_time:.6f}", "0.01s"]
})

# 保存到Excel
with pd.ExcelWriter(f"{output_dir}/06_optimization_results.xlsx", engine='openpyxl') as writer:
    excel_df.to_excel(writer, sheet_name='最优参数', index=False)
    position_data.to_excel(writer, sheet_name='关键位置', index=False)
    performance_data.to_excel(writer, sheet_name='性能指标', index=False)
    
    # 保存敏感性分析结果
    for i, result in enumerate(sensitivity_results):
        sensitivity_df = pd.DataFrame({
            result['param_name']: result['param_values'],
            '遮蔽时间 (s)': result['shielding_times']
        })
        sensitivity_df.to_excel(writer, sheet_name=f'敏感性_{i+1}', index=False)

print("✅ 所有结果已保存到 ImageOutput/02/ 目录")

# %% [markdown]
# ## 9. 结果总结

# %%
print("\n" + "="*60)
print("🎯 问题2：单弹最优投放策略 - 结果总结")
print("="*60)

print(f"\n📊 最优策略参数:")
print(f"   🚁 无人机速度: {optimal_params[0]:.2f} m/s")
print(f"   🧭 飞行方向: {np.degrees(optimal_params[1]):.1f}° ({optimal_params[1]:.4f} rad)")
print(f"   ⏰ 投放时间: {optimal_params[2]:.2f} s")
print(f"   💥 起爆延时: {optimal_params[3]:.2f} s")
print(f"   🎆 起爆时间: {t_burst_opt:.2f} s")

print(f"\n📍 关键位置:")
print(f"   📦 投放位置: ({release_pos[0]:.0f}, {release_pos[1]:.0f}, {release_pos[2]:.0f}) m")
print(f"   💥 起爆位置: ({burst_pos[0]:.0f}, {burst_pos[1]:.0f}, {burst_pos[2]:.0f}) m")

print(f"\n🎯 性能指标:")
print(f"   ⏱️  最大遮蔽时间: {precise_shielding_time:.4f} s")
print(f"   🔍 优化精度: ±0.01 s")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_optimal.html - 3D轨迹交互图")
print(f"   📊 02_shielding_analysis.html - 遮蔽效果分析")
print(f"   🔬 03_sensitivity_analysis.html - 参数敏感性分析")
print(f"   📋 04_results_summary.json - 完整结果汇总")
print(f"   📊 05_detailed_trajectory.csv - 详细轨迹数据")
print(f"   📑 06_optimization_results.xlsx - Excel结果表格")

print(f"\n✅ 问题2求解完成！所有结果已保存到 ImageOutput/02/ 目录")
print("="*60)# # 问题2：单弹最优投放策略优化
# 
# ## 问题描述
# - 无人机：FY1
# - 需优化：飞行方向、速度、投放点、起爆点
# - 目标：最大化对M1的遮蔽时间
# - 约束：速度70-140m/s，等高度飞行

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
output_dir = "../../ImageOutput/02"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题2：单弹最优投放策略优化")
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
FY1_initial = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置
v_drone_min = 70.0  # 最小速度 (m/s)
v_drone_max = 140.0  # 最大速度 (m/s)

# 计算导弹单位方向向量（指向假目标原点）
missile_direction = -M1_initial / np.linalg.norm(M1_initial)

print(f"📍 导弹M1初始位置: {M1_initial}")
print(f"📍 无人机FY1初始位置: {FY1_initial}")
print(f"📍 真目标位置: {target_pos}")
print(f"🎯 导弹飞行方向: {missile_direction}")

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

def evaluate_shielding_time(params, dt=0.02, smooth=False, kappa=50):
    """
    评估遮蔽时间
    params: [v_drone, alpha, t_release, t_burst_delay]
    """
    v_drone, alpha, t_release, t_burst_delay = params
    
    # 约束检查
    if v_drone < v_drone_min or v_drone > v_drone_max:
        return -1000
    if t_release < 0 or t_burst_delay < 0:
        return -1000
    
    # 计算起爆位置和时间
    burst_pos = smoke_burst_position(t_release, t_burst_delay, v_drone, alpha)
    t_burst = t_release + t_burst_delay
    
    # 检查起爆位置是否合理（不能在地面以下）
    if burst_pos[2] < 0:
        return -1000
    
    # 计算遮蔽时间
    total_shielding = 0.0
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
        
        if smooth:
            # 平滑目标函数（用于优化）
            shielding_factor = 1.0 / (1.0 + np.exp(kappa * (distance - R_cloud)))
            total_shielding += shielding_factor * dt
        else:
            # 硬阈值（用于精确评估）
            if distance <= R_cloud:
                total_shielding += dt
    
    return total_shielding

print("✅ 核心计算函数定义完成")

# %% [markdown]
# ## 3. 优化求解

# %%
print("🔍 开始优化求解...")

# 定义优化边界
# [v_drone, alpha, t_release, t_burst_delay]
bounds = [
    (v_drone_min, v_drone_max),  # 无人机速度
    (0, 2*np.pi),                # 飞行方向角
    (0, 30),                     # 投放时间
    (0, 20)                      # 起爆延时
]

# 目标函数（最大化遮蔽时间，所以取负值）
def objective_function(params):
    return -evaluate_shielding_time(params, dt=0.05, smooth=True, kappa=30)

# 使用差分进化算法进行全局优化
print("🎯 使用差分进化算法进行全局优化...")
result = differential_evolution(
    objective_function,
    bounds,
    seed=42,
    maxiter=200,
    popsize=20,
    atol=1e-6,
    tol=1e-6,
    workers=1,
    disp=True
)

optimal_params = result.x
optimal_shielding_time = -result.fun

print(f"✅ 优化完成！")
print(f"📊 最优参数:")
print(f"   - 无人机速度: {optimal_params[0]:.2f} m/s")
print(f"   - 飞行方向角: {optimal_params[1]:.4f} rad ({np.degrees(optimal_params[1]):.2f}°)")
print(f"   - 投放时间: {optimal_params[2]:.2f} s")
print(f"   - 起爆延时: {optimal_params[3]:.2f} s")
print(f"🎯 最大遮蔽时间: {optimal_shielding_time:.4f} s")

# 用精确方法重新评估最优解
precise_shielding_time = evaluate_shielding_time(optimal_params, dt=0.01, smooth=False)
print(f"🔍 精确遮蔽时间: {precise_shielding_time:.4f} s")

# %% [markdown]
# ## 4. 详细轨迹分析

# %%
print("📈 生成详细轨迹数据...")

v_opt, alpha_opt, t_release_opt, t_burst_delay_opt = optimal_params
t_burst_opt = t_release_opt + t_burst_delay_opt

# 计算关键位置
release_pos = smoke_release_position(t_release_opt, v_opt, alpha_opt)
burst_pos = smoke_burst_position(t_release_opt, t_burst_delay_opt, v_opt, alpha_opt)

print(f"📍 投放位置: ({release_pos[0]:.1f}, {release_pos[1]:.1f}, {release_pos[2]:.1f})")
print(f"💥 起爆位置: ({burst_pos[0]:.1f}, {burst_pos[1]:.1f}, {burst_pos[2]:.1f})")
print(f"⏰ 起爆时间: {t_burst_opt:.2f} s")

# 生成时间序列数据
t_max = t_burst_opt + cloud_duration + 5
time_points = np.arange(0, t_max, 0.02)

# 存储轨迹数据
trajectory_data = []
shielding_data = []

for t in time_points:
    # 导弹位置
    missile_pos = missile_position(t)
    
    # 无人机位置
    drone_pos = drone_position(t, v_opt, alpha_opt)
    
    # 云团位置（如果已起爆）
    cloud_pos = None
    distance_to_line = np.inf
    is_shielded = False
    
    if t >= t_burst_opt and t <= t_burst_opt + cloud_duration:
        cloud_pos = cloud_center_position(t, t_burst_opt, burst_pos)
        if cloud_pos is not None and cloud_pos[2] >= 0:
            distance_to_line = distance_to_missile_target_line(cloud_pos, missile_pos, target_pos)
            is_shielded = distance_to_line <= R_cloud
    
    trajectory_data.append({
        'time': t,
        'missile_x': missile_pos[0],
        'missile_y': missile_pos[1],
        'missile_z': missile_pos[2],
        'drone_x': drone_pos[0],
        'drone_y': drone_pos[1],
        'drone_z': drone_pos[2],
        'cloud_x': cloud_pos[0] if cloud_pos is not None else np.nan,
        'cloud_y': cloud_pos[1] if cloud_pos is not None else np.nan,
        'cloud_z': cloud_pos[2] if cloud_pos is not None else np.nan,
        'distance_to_line': distance_to_line if distance_to_line != np.inf else np.nan,
        'is_shielded': is_shielded
    })

trajectory_df = pd.DataFrame(trajectory_data)

print(f"✅ 生成了 {len(trajectory_df)} 个时间点的轨迹数据")

# %% [markdown]
# ## 5. 3D轨迹可视化

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

# 云团轨迹（仅显示有效时间内）
cloud_mask = ~trajectory_df['cloud_x'].isna()
if cloud_mask.any():
    fig_3d.add_trace(go.Scatter3d(
        x=trajectory_df.loc[cloud_mask, 'cloud_x'],
        y=trajectory_df.loc[cloud_mask, 'cloud_y'],
        z=trajectory_df.loc[cloud_mask, 'cloud_z'],
        mode='lines+markers',
        line=dict(color='gray', width=8, dash='dash'),
        marker=dict(size=4, color='gray', opacity=0.7),
        name='云团中心轨迹',
        hovertemplate='<b>云团中心</b><br>' +
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
fig_3d.add_trace(go.Scatter3d(
    x=[release_pos[0], burst_pos[0]],
    y=[release_pos[1], burst_pos[1]],
    z=[release_pos[2], burst_pos[2]],
    mode='markers',
    marker=dict(size=12, color=['green', 'orange'], symbol='x'),
    name='关键事件',
    text=['烟幕弹投放', '烟幕弹起爆'],
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x:.0f}m<br>' +
                  'Y: %{y:.0f}m<br>' +
                  'Z: %{z:.0f}m<br>' +
                  '<extra></extra>'
))

# 设置布局
fig_3d.update_layout(
    title=dict(
        text='问题2：最优投放策略 - 3D轨迹图',
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
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_optimal.html")
fig_3d.write_image(f"{output_dir}/01_3d_trajectory_optimal.svg")
fig_3d.show()

print("✅ 3D轨迹图已保存")

# %% [markdown]
# ## 6. 遮蔽效果分析

# %%
print("📊 创建遮蔽效果分析图...")

# 创建子图
fig_analysis = make_subplots(
    rows=3, cols=1,
    subplot_titles=[
        '云团到导弹-目标连线的距离',
        '遮蔽状态时间序列',
        '累积遮蔽时间'
    ],
    vertical_spacing=0.08,
    specs=[[{"secondary_y": False}],
           [{"secondary_y": False}],
           [{"secondary_y": False}]]
)

# 过滤有效数据
valid_mask = ~trajectory_df['distance_to_line'].isna()
valid_data = trajectory_df[valid_mask].copy()

if len(valid_data) > 0:
    # 距离曲线
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=valid_data['distance_to_line'],
            mode='lines',
            line=dict(color='blue', width=3),
            name='距离',
            hovertemplate='时间: %{x:.2f}s<br>距离: %{y:.2f}m<extra></extra>'
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
    
    # 遮蔽状态
    shielding_status = valid_data['is_shielded'].astype(int)
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=shielding_status,
            mode='lines+markers',
            line=dict(color='green', width=4),
            marker=dict(size=4),
            name='遮蔽状态',
            hovertemplate='时间: %{x:.2f}s<br>遮蔽: %{text}<extra></extra>',
            text=['是' if x else '否' for x in valid_data['is_shielded']]
        ),
        row=2, col=1
    )
    
    # 累积遮蔽时间
    cumulative_shielding = np.cumsum(shielding_status) * 0.02  # dt = 0.02
    fig_analysis.add_trace(
        go.Scatter(
            x=valid_data['time'],
            y=cumulative_shielding,
            mode='lines',
            line=dict(color='purple', width=3),
            name='累积遮蔽时间',
            hovertemplate='时间: %{x:.2f}s<br>累积遮蔽: %{y:.3f}s<extra></extra>'
        ),
        row=3, col=1
    )

# 更新布局
fig_analysis.update_layout(
    title=dict(
        text='问题2：最优策略遮蔽效果分析',
        x=0.5,
        font=dict(size=18, color='darkblue')
    ),
    height=900,
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

# 更新坐标轴
fig_analysis.update_xaxes(title_text="时间 (s)", row=3, col=1)
fig_analysis.update_yaxes(title_text="距离 (m)", row=1, col=1)
fig_analysis.update_yaxes(title_text="遮蔽状态", row=2, col=1)
fig_analysis.update_yaxes(title_text="累积时间 (s)", row=3, col=1)

# 保存分析图
fig_analysis.write_html(f"{output_dir}/02_shielding_analysis.html")
fig_analysis.write_image(f"{output_dir}/02_shielding_analysis.svg")
fig_analysis.show()

print("✅ 遮蔽效果分析图已保存")

# %% [markdown]
# ## 7. 参数敏感性分析

# %%
print("🔬 进行参数敏感性分析...")

# 定义参数变化范围
param_names = ['速度 (m/s)', '方向角 (rad)', '投放时间 (s)', '起爆延时 (s)']
param_ranges = [
    np.linspace(70, 140, 15),
    np.linspace(0, 2*np.pi, 20),
    np.linspace(0, 20, 15),
    np.linspace(0, 15, 15)
]

sensitivity_results = []

for i, (param_name, param_range) in enumerate(zip(param_names, param_ranges)):
    print(f"  分析参数: {param_name}")
    
    param_shielding = []
    for param_value in param_range:
        # 创建测试参数组合
        test_params = optimal_params.copy()
        test_params[i] = param_value
        
        # 评估遮蔽时间
        shielding_time = evaluate_shielding_time(test_params, dt=0.05, smooth=False)
        param_shielding.append(max(0, shielding_time))  # 确保非负
    
    sensitivity_results.append({
        'param_name': param_name,
        'param_values': param_range,
        'shielding_times': param_shielding,
        'optimal_value': optimal_params[i],
        'optimal_shielding': precise_shielding_time
    })

# 创建敏感性分析图
fig_sensitivity = make_subplots(
    rows=2, cols=2,
    subplot_titles=[f'{result["param_name"]}敏感性' for result in sensitivity_results],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

positions = [(1,1), (1,2), (2,1), (2,2)]

for idx, (result, pos) in enumerate(zip(sensitivity_results, positions)):
    fig_sensitivity.add_trace(
        go.Scatter(
            x=result['param_values'],
            y=result['shielding_times'],
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=6),
            name=result['param_name'],
            hovertemplate=f'{result["param_name"]}: %{{x:.3f}}<br>遮蔽时间: %{{y:.4f}}s<extra></extra>'
        ),
        row=pos[0], col=pos[1]
    )
    
    # 标记最优值
    fig_sensitivity.add_vline(
        x=result['optimal_value'],
        line_dash="dash",
        line_color="red",
        annotation_text="最优值",
        row=pos[0], col=pos[1]
    )

fig_sensitivity.update_layout(
    title=dict(
        text='问题2：参数敏感性分析',
        x=0.5,
        font=dict(size=18, color='darkblue')
    ),
    height=800,
    showlegend=False
)

# 更新坐标轴标签
fig_sensitivity.update_yaxes(title_text="遮蔽时间 (s)")

# 保存敏感性分析图
fig_sensitivity.write_html(f"{output_dir}/03_sensitivity_analysis.html")
fig_sensitivity.write_image(f"{output_dir}/03_sensitivity_analysis.svg")
fig_sensitivity.show()

print("✅ 参数敏感性分析完成")

# %% [markdown]
# ## 8. 结果汇总与保存

# %%
print("💾 保存结果数据...")

# 汇总结果
results_summary = {
    'problem': '问题2：单弹最优投放策略',
    'optimization_method': '差分进化算法',
    'optimal_parameters': {
        'drone_speed_ms': float(optimal_params[0]),
        'flight_direction_rad': float(optimal_params[1]),
        'flight_direction_deg': float(np.degrees(optimal_params[1])),
        'release_time_s': float(optimal_params[2]),
        'burst_delay_s': float(optimal_params[3])
    },
    'key_positions': {
        'release_position': release_pos.tolist(),
        'burst_position': burst_pos.tolist(),
        'burst_time_s': float(t_burst_opt)
    },
    'performance': {
        'max_shielding_time_s': float(precise_shielding_time),
        'optimization_shielding_time_s': float(optimal_shielding_time)
    },
    'constraints': {
        'drone_speed_range_ms': [v_drone_min, v_drone_max],
        'cloud_radius_m': R_cloud,
        'cloud_duration_s': cloud_duration
    }
}

# 保存JSON结果
with open(f"{output_dir}/04_results_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存详细轨迹数据
trajectory_df.to_csv(f"{output_dir}/05_detailed_trajectory.csv", index=False)

# 创建Excel结果表
excel_data = {
    '最优参数': [
        '无人机速度 (m/s)',
        '飞行方向角 (rad)',
        '飞行方向角 (度)',
        '投放时间 (s)',
        '起爆延时 (s)',
        '起爆时间 (s)'
    ],
    '数值': [
        f"{optimal_params[0]:.3f}",
        f"{optimal_params[1]:.6f}",
        f"{np.degrees(optimal_params[1]):.2f}",
        f"{optimal_params[2]:.3f}",
        f"{optimal_params[3]:.3f}",
        f"{t_burst_opt:.3f}"
    ]
}

excel_df = pd.DataFrame(excel_data)

# 添加关键位置信息
position_data = pd.DataFrame({
    '关键位置': ['投放位置X (m)', '投放位置Y (m)', '投放位置Z (m)',
                '起爆位置X (m)', '起爆位置Y (m)', '起爆位置Z (m)'],
    '坐标值': [f"{release_pos[0]:.1f}", f"{release_pos[1]:.1f}", f"{release_pos[2]:.1f}",
              f"{burst_pos[0]:.1f}", f"{burst_pos[1]:.1f}", f"{burst_pos[2]:.1f}"]
})

# 添加性能指标
performance_data = pd.DataFrame({
    '性能指标': ['最大遮蔽时间 (s)', '优化目标值', '计算精度'],
    '结果': [f"{precise_shielding_time:.6f}", f"{optimal_shielding_time:.6f}", "0.01s"]
})

# 保存到Excel
with pd.ExcelWriter(f"{output_dir}/06_optimization_results.xlsx", engine='openpyxl') as writer:
    excel_df.to_excel(writer, sheet_name='最优参数', index=False)
    position_data.to_excel(writer, sheet_name='关键位置', index=False)
    performance_data.to_excel(writer, sheet_name='性能指标', index=False)
    
    # 保存敏感性分析结果
    for i, result in enumerate(sensitivity_results):
        sensitivity_df = pd.DataFrame({
            result['param_name']: result['param_values'],
            '遮蔽时间 (s)': result['shielding_times']
        })
        sensitivity_df.to_excel(writer, sheet_name=f'敏感性_{i+1}', index=False)

print("✅ 所有结果已保存到 ImageOutput/02/ 目录")

# %% [markdown]
# ## 9. 结果总结

# %%
print("\n" + "="*60)
print("🎯 问题2：单弹最优投放策略 - 结果总结")
print("="*60)

print(f"\n📊 最优策略参数:")
print(f"   🚁 无人机速度: {optimal_params[0]:.2f} m/s")
print(f"   🧭 飞行方向: {np.degrees(optimal_params[1]):.1f}° ({optimal_params[1]:.4f} rad)")
print(f"   ⏰ 投放时间: {optimal_params[2]:.2f} s")
print(f"   💥 起爆延时: {optimal_params[3]:.2f} s")
print(f"   🎆 起爆时间: {t_burst_opt:.2f} s")

print(f"\n📍 关键位置:")
print(f"   📦 投放位置: ({release_pos[0]:.0f}, {release_pos[1]:.0f}, {release_pos[2]:.0f}) m")
print(f"   💥 起爆位置: ({burst_pos[0]:.0f}, {burst_pos[1]:.0f}, {burst_pos[2]:.0f}) m")

print(f"\n🎯 性能指标:")
print(f"   ⏱️  最大遮蔽时间: {precise_shielding_time:.4f} s")
print(f"   🔍 优化精度: ±0.01 s")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_optimal.html - 3D轨迹交互图")
print(f"   📊 02_shielding_analysis.html - 遮蔽效果分析")
print(f"   🔬 03_sensitivity_analysis.html - 参数敏感性分析")
print(f"   📋 04_results_summary.json - 完整结果汇总")
print(f"   📊 05_detailed_trajectory.csv - 详细轨迹数据")
print(f"   📑 06_optimization_results.xlsx - Excel结果表格")

print(f"\n✅ 问题2求解完成！所有结果已保存到 ImageOutput/02/ 目录")
print("="*60)
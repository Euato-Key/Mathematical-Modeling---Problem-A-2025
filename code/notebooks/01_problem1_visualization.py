"""
问题1：烟幕干扰弹对M1的有效遮蔽时长计算 - 可视化版本
基于01_problem1_fixed_parameters.py，增加Plotly交互式可视化

功能：
1. 3D轨迹可视化（导弹、无人机、烟幕弹、云团）
2. 遮蔽效果动态分析
3. 时间序列图表
4. 交互式参数面板
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import os
import json

# 设置Plotly在Jupyter中显示
pyo.init_notebook_mode(connected=True)

# ============================================================================
# 第一步：定义基本参数和常量
# ============================================================================

print("=== 第一步：定义基本参数 ===")

# 物理常量
g = 9.8  # 重力加速度 m/s²
smoke_sink_speed = 3.0  # 烟幕云团下沉速度 m/s
effective_radius = 10.0  # 有效遮蔽半径 m
effective_duration = 20.0  # 有效遮蔽持续时间 s

# 导弹参数
missile_speed = 300.0  # 导弹速度 m/s
M1_initial = np.array([20000.0, 0.0, 2000.0])  # 导弹M1初始位置

# 无人机参数
FY1_initial = np.array([17800.0, 0.0, 1800.0])  # 无人机FY1初始位置
drone_speed = 120.0  # 无人机速度 m/s

# 目标位置
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
real_target = np.array([0.0, 200.0, 0.0])  # 真目标位置

# 时间参数
t_deploy = 1.5  # 投放时间 s
t_explode_delay = 3.6  # 起爆延迟 s
t_explode = t_deploy + t_explode_delay  # 起爆时间 s

print(f"导弹M1初始位置: {M1_initial}")
print(f"无人机FY1初始位置: {FY1_initial}")
print(f"投放时间: {t_deploy}s")
print(f"起爆时间: {t_explode}s")

# ============================================================================
# 第二步：运动模型函数定义
# ============================================================================

def calculate_missile_velocity(initial_pos: np.ndarray, target_pos: np.ndarray, speed: float) -> np.ndarray:
    """计算导弹速度向量"""
    direction = target_pos - initial_pos
    unit_direction = direction / np.linalg.norm(direction)
    return speed * unit_direction

def missile_position(t: float, initial_pos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """计算导弹在时刻t的位置"""
    return initial_pos + velocity * t

def calculate_drone_velocity_horizontal(initial_pos: np.ndarray, target_pos: np.ndarray, speed: float) -> np.ndarray:
    """计算无人机水平方向速度向量（等高度飞行）"""
    direction_2d = target_pos[:2] - initial_pos[:2]
    unit_direction_2d = direction_2d / np.linalg.norm(direction_2d)
    velocity_3d = np.array([unit_direction_2d[0], unit_direction_2d[1], 0.0]) * speed
    return velocity_3d

def drone_position(t: float, initial_pos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """计算无人机在时刻t的位置"""
    return initial_pos + velocity * t

def smoke_bomb_position(t: float, deploy_time: float, deploy_pos: np.ndarray,
                       initial_velocity: np.ndarray) -> np.ndarray:
    """计算烟幕弹在时刻t的位置（考虑重力）"""
    if t < deploy_time:
        return deploy_pos

    dt = t - deploy_time
    horizontal_displacement = initial_velocity[:2] * dt
    vertical_displacement = initial_velocity[2] * dt - 0.5 * g * dt**2

    position = deploy_pos.copy()
    position[:2] += horizontal_displacement
    position[2] += vertical_displacement

    return position

def smoke_cloud_position(t: float, explode_time: float, explode_pos: np.ndarray) -> Optional[np.ndarray]:
    """计算烟幕云团在时刻t的位置"""
    if t < explode_time:
        return None

    dt = t - explode_time
    if dt > effective_duration:
        return None

    position = explode_pos.copy()
    position[2] -= smoke_sink_speed * dt

    return position

def point_to_line_segment_distance(point: np.ndarray, line_start: np.ndarray,
                                 line_end: np.ndarray) -> Tuple[float, float]:
    """计算点到线段的最短距离"""
    AB = line_end - line_start
    AP = point - line_start

    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return float(np.linalg.norm(AP)), 0.0

    u = float(np.dot(AP, AB) / AB_squared)

    if u < 0:
        distance = float(np.linalg.norm(AP))
    elif u > 1:
        BP = point - line_end
        distance = float(np.linalg.norm(BP))
    else:
        cross_product = np.cross(AP, AB)
        if AB.ndim == 1 and len(AB) == 3:
            distance = float(np.linalg.norm(cross_product) / np.linalg.norm(AB))
        else:
            distance = float(abs(cross_product) / np.linalg.norm(AB))

    return distance, u

def is_shielded(t: float, explode_time: float, explode_pos: np.ndarray,
               missile_initial: np.ndarray, missile_vel: np.ndarray,
               target_pos: np.ndarray, radius: float) -> Tuple[bool, dict]:
    """判断在时刻t是否被遮蔽"""
    if t < explode_time or t > explode_time + effective_duration:
        return False, {"reason": "云团无效"}

    missile_pos = missile_position(t, missile_initial, missile_vel)
    cloud_pos = smoke_cloud_position(t, explode_time, explode_pos)

    if cloud_pos is None:
        return False, {"reason": "云团位置无效"}

    distance, u = point_to_line_segment_distance(cloud_pos, missile_pos, target_pos)
    is_blocked = distance <= radius and 0 <= u <= 1

    info = {
        "time": t,
        "missile_pos": missile_pos,
        "cloud_pos": cloud_pos,
        "target_pos": target_pos,
        "distance": distance,
        "projection_u": u,
        "is_blocked": is_blocked
    }

    return is_blocked, info

# ============================================================================
# 第三步：计算轨迹数据
# ============================================================================

print("\n=== 第三步：计算轨迹数据 ===")

# 计算速度向量
missile_velocity = calculate_missile_velocity(M1_initial, fake_target, missile_speed)
drone_velocity = calculate_drone_velocity_horizontal(FY1_initial, fake_target, drone_speed)

# 计算关键位置
deploy_position = drone_position(t_deploy, FY1_initial, drone_velocity)
explode_position = smoke_bomb_position(t_explode, t_deploy, deploy_position, drone_velocity)

print(f"导弹速度向量: {missile_velocity}")
print(f"无人机速度向量: {drone_velocity}")
print(f"投放位置: {deploy_position}")
print(f"起爆位置: {explode_position}")

# 生成时间序列数据
t_max = 30.0  # 总仿真时间
dt = 0.01  # 时间步长（与原始计算保持一致）
time_points = np.arange(0, t_max + dt, dt)

# 计算所有轨迹点
trajectory_data = []

for t in time_points:
    # 导弹位置
    missile_pos = missile_position(t, M1_initial, missile_velocity)

    # 无人机位置
    drone_pos = drone_position(t, FY1_initial, drone_velocity)

    # 烟幕弹位置
    if t >= t_deploy:
        bomb_pos = smoke_bomb_position(t, t_deploy, deploy_position, drone_velocity)
    else:
        bomb_pos = drone_pos.copy()

    # 烟幕云团位置
    cloud_pos = smoke_cloud_position(t, t_explode, explode_position)

    # 遮蔽状态
    blocked, shield_info = is_shielded(t, t_explode, explode_position,
                                     M1_initial, missile_velocity, real_target, effective_radius)

    trajectory_data.append({
        'time': t,
        'missile_x': missile_pos[0], 'missile_y': missile_pos[1], 'missile_z': missile_pos[2],
        'drone_x': drone_pos[0], 'drone_y': drone_pos[1], 'drone_z': drone_pos[2],
        'bomb_x': bomb_pos[0], 'bomb_y': bomb_pos[1], 'bomb_z': bomb_pos[2],
        'cloud_x': cloud_pos[0] if cloud_pos is not None else None,
        'cloud_y': cloud_pos[1] if cloud_pos is not None else None,
        'cloud_z': cloud_pos[2] if cloud_pos is not None else None,
        'is_shielded': blocked,
        'shield_distance': shield_info.get('distance', np.nan),
        'projection_u': shield_info.get('projection_u', np.nan)
    })

df = pd.DataFrame(trajectory_data)

# 计算有效遮蔽时长
shielded_df = df[df['is_shielded'] == True]
if len(shielded_df) > 0:
    shielding_duration = len(shielded_df) * dt
    print(f"\n有效遮蔽时长: {shielding_duration:.3f} 秒")
    print(f"遮蔽开始时间: {shielded_df['time'].min():.3f}s")
    print(f"遮蔽结束时间: {shielded_df['time'].max():.3f}s")
else:
    shielding_duration = 0.0
    print(f"\n有效遮蔽时长: {shielding_duration:.3f} 秒")

# ============================================================================
# 第四步：创建3D轨迹可视化
# ============================================================================

print("\n=== 第四步：创建3D轨迹可视化 ===")

def create_3d_trajectory_plot():
    """创建3D轨迹图"""
    fig = go.Figure()

    # 导弹轨迹
    fig.add_trace(go.Scatter3d(
        x=df['missile_x'], y=df['missile_y'], z=df['missile_z'],
        mode='lines+markers',
        name='导弹M1轨迹',
        line=dict(color='red', width=4),
        marker=dict(size=3, color='red')
    ))

    # 无人机轨迹
    fig.add_trace(go.Scatter3d(
        x=df['drone_x'], y=df['drone_y'], z=df['drone_z'],
        mode='lines+markers',
        name='无人机FY1轨迹',
        line=dict(color='blue', width=4),
        marker=dict(size=3, color='blue')
    ))

    # 烟幕弹轨迹
    bomb_df = df[df['time'] >= t_deploy]
    fig.add_trace(go.Scatter3d(
        x=bomb_df['bomb_x'], y=bomb_df['bomb_y'], z=bomb_df['bomb_z'],
        mode='lines+markers',
        name='烟幕弹轨迹',
        line=dict(color='orange', width=3),
        marker=dict(size=2, color='orange')
    ))

    # 烟幕云团轨迹
    cloud_df = df.dropna(subset=['cloud_x'])
    if len(cloud_df) > 0:
        fig.add_trace(go.Scatter3d(
            x=cloud_df['cloud_x'], y=cloud_df['cloud_y'], z=cloud_df['cloud_z'],
            mode='lines+markers',
            name='烟幕云团轨迹',
            line=dict(color='gray', width=5),
            marker=dict(size=4, color='gray', opacity=0.7)
        ))

    # 关键点标记
    # 假目标
    fig.add_trace(go.Scatter3d(
        x=[fake_target[0]], y=[fake_target[1]], z=[fake_target[2]],
        mode='markers',
        name='假目标',
        marker=dict(size=10, color='black', symbol='diamond')
    ))

    # 真目标
    fig.add_trace(go.Scatter3d(
        x=[real_target[0]], y=[real_target[1]], z=[real_target[2]],
        mode='markers',
        name='真目标',
        marker=dict(size=10, color='green', symbol='square')
    ))

    # 投放点
    fig.add_trace(go.Scatter3d(
        x=[deploy_position[0]], y=[deploy_position[1]], z=[deploy_position[2]],
        mode='markers',
        name='投放点',
        marker=dict(size=8, color='purple', symbol='cross')
    ))

    # 起爆点
    fig.add_trace(go.Scatter3d(
        x=[explode_position[0]], y=[explode_position[1]], z=[explode_position[2]],
        mode='markers',
        name='起爆点',
        marker=dict(size=8, color='yellow', symbol='diamond')
    ))

    # 遮蔽区域可视化（选择几个时间点）
    shield_times = np.linspace(t_explode, t_explode + effective_duration, 5)
    for i, t in enumerate(shield_times):
        cloud_pos = smoke_cloud_position(t, t_explode, explode_position)
        if cloud_pos is not None:
            # 创建球体表面点
            phi = np.linspace(0, 2*np.pi, 20)
            theta = np.linspace(0, np.pi, 10)
            phi, theta = np.meshgrid(phi, theta)

            x_sphere = cloud_pos[0] + effective_radius * np.sin(theta) * np.cos(phi)
            y_sphere = cloud_pos[1] + effective_radius * np.sin(theta) * np.sin(phi)
            z_sphere = cloud_pos[2] + effective_radius * np.cos(theta)

            fig.add_trace(go.Surface(
                x=x_sphere, y=y_sphere, z=z_sphere,
                opacity=0.1,
                colorscale='Greys',
                showscale=False,
                name=f'遮蔽球t={t:.1f}s'
            ))

    fig.update_layout(
        title='烟幕干扰弹3D轨迹可视化',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            yaxis=dict(
                dtick=50,  # Y轴刻度间隔50m
                tickmode='linear'
            ),
            xaxis=dict(
                dtick=2000,  # X轴刻度间隔2000m
                tickmode='linear'
            ),
            zaxis=dict(
                dtick=200,  # Z轴刻度间隔200m
                tickmode='linear'
            )
        ),
        width=1000,
        height=800
    )

    return fig

# 创建并显示3D图
fig_3d = create_3d_trajectory_plot()
fig_3d.show()

# ============================================================================
# 第五步：创建时间序列分析图
# ============================================================================

print("\n=== 第五步：创建时间序列分析图 ===")

def create_time_series_analysis():
    """创建时间序列分析图"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('遮蔽状态时间序列', '遮蔽距离变化',
                       '高度变化', '水平位置变化',
                       '投影参数u', '速度分析'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. 遮蔽状态时间序列
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['is_shielded'].astype(int),
                  mode='lines+markers', name='遮蔽状态',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )

    # 添加关键时间点标记
    for time_point, label in [(t_deploy, '投放'), (t_explode, '起爆'),
                             (t_explode + effective_duration, '云团消失')]:
        fig.add_vline(x=time_point, line_dash="dash", line_color="gray",
                     annotation_text=label, row=1, col=1)

    # 2. 遮蔽距离变化
    valid_distances = df.dropna(subset=['shield_distance'])
    if len(valid_distances) > 0:
        fig.add_trace(
            go.Scatter(x=valid_distances['time'], y=valid_distances['shield_distance'],
                      mode='lines', name='遮蔽距离',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        fig.add_hline(y=effective_radius, line_dash="dash", line_color="red",
                     annotation_text="有效半径", row=1, col=2)

    # 3. 高度变化
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['missile_z'],
                  mode='lines', name='导弹高度',
                  line=dict(color='red', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['drone_z'],
                  mode='lines', name='无人机高度',
                  line=dict(color='blue', width=2)),
        row=2, col=1
    )
    cloud_df = df.dropna(subset=['cloud_z'])
    if len(cloud_df) > 0:
        fig.add_trace(
            go.Scatter(x=cloud_df['time'], y=cloud_df['cloud_z'],
                      mode='lines', name='云团高度',
                      line=dict(color='gray', width=2)),
            row=2, col=1
        )

    # 4. 水平位置变化
    fig.add_trace(
        go.Scatter(x=df['missile_x'], y=df['missile_y'],
                  mode='lines', name='导弹水平轨迹',
                  line=dict(color='red', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['drone_x'], y=df['drone_y'],
                  mode='lines', name='无人机水平轨迹',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )

    # 5. 投影参数u
    valid_u = df.dropna(subset=['projection_u'])
    if len(valid_u) > 0:
        fig.add_trace(
            go.Scatter(x=valid_u['time'], y=valid_u['projection_u'],
                      mode='lines', name='投影参数u',
                      line=dict(color='green', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=3, col=1)

    # 6. 速度分析
    missile_speed_calc = np.sqrt(missile_velocity[0]**2 + missile_velocity[1]**2 + missile_velocity[2]**2)
    drone_speed_calc = np.sqrt(drone_velocity[0]**2 + drone_velocity[1]**2 + drone_velocity[2]**2)

    fig.add_trace(
        go.Scatter(x=[0, t_max], y=[missile_speed_calc, missile_speed_calc],
                  mode='lines', name=f'导弹速度 {missile_speed_calc:.1f} m/s',
                  line=dict(color='red', width=2, dash='dash')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0, t_max], y=[drone_speed_calc, drone_speed_calc],
                  mode='lines', name=f'无人机速度 {drone_speed_calc:.1f} m/s',
                  line=dict(color='blue', width=2, dash='dash')),
        row=3, col=2
    )

    fig.update_layout(
        title='烟幕干扰效果时间序列分析',
        height=1200,
        showlegend=True
    )

    return fig

# 创建并显示时间序列图
fig_time = create_time_series_analysis()
fig_time.show()

# ============================================================================
# 第六步：创建遮蔽效果分析仪表板
# ============================================================================

print("\n=== 第六步：创建遮蔽效果分析仪表板 ===")

def create_shielding_dashboard():
    """创建遮蔽效果分析仪表板"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('遮蔽时长统计', '距离分布', '遮蔽效率',
                       '关键参数', '轨迹对比', '3D视角'),
        specs=[[{"type": "indicator"}, {"type": "histogram"}, {"type": "bar"}],
               [{"type": "table"}, {"type": "scatter"}, {"type": "scatter3d"}]]
    )

    # 1. 遮蔽时长指示器
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=shielding_duration,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "有效遮蔽时长 (秒)"},
            delta={'reference': 10},
            gauge={'axis': {'range': [None, 20]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 15], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 15}}
        ),
        row=1, col=1
    )

    # 2. 距离分布直方图
    if len(shielded_df) > 0:
        fig.add_trace(
            go.Histogram(x=shielded_df['shield_distance'],
                        nbinsx=20, name='遮蔽距离分布'),
            row=1, col=2
        )

    # 3. 遮蔽效率柱状图
    total_cloud_time = effective_duration
    efficiency = (shielding_duration / total_cloud_time * 100) if total_cloud_time > 0 else 0

    fig.add_trace(
        go.Bar(x=['遮蔽效率', '未遮蔽'],
               y=[efficiency, 100-efficiency],
               marker_color=['green', 'red']),
        row=1, col=3
    )

    # 4. 关键参数表格
    key_params = [
        ['参数', '数值', '单位'],
        ['有效遮蔽时长', f'{shielding_duration:.3f}', '秒'],
        ['投放时间', f'{t_deploy}', '秒'],
        ['起爆时间', f'{t_explode}', '秒'],
        ['遮蔽效率', f'{efficiency:.1f}', '%'],
        ['有效半径', f'{effective_radius}', '米'],
        ['云团持续时间', f'{effective_duration}', '秒'],
        ['导弹速度', f'{missile_speed}', 'm/s'],
        ['无人机速度', f'{drone_speed}', 'm/s']
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=key_params[0]),
            cells=dict(values=list(zip(*key_params[1:])))
        ),
        row=2, col=1
    )

    # 5. 轨迹对比（俯视图）
    fig.add_trace(
        go.Scatter(x=df['missile_x'], y=df['missile_y'],
                  mode='lines', name='导弹轨迹',
                  line=dict(color='red', width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['drone_x'], y=df['drone_y'],
                  mode='lines', name='无人机轨迹',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )

    # 添加目标点
    fig.add_trace(
        go.Scatter(x=[fake_target[0], real_target[0]],
                  y=[fake_target[1], real_target[1]],
                  mode='markers', name='目标',
                  marker=dict(size=10, color=['black', 'green'])),
        row=2, col=2
    )

    # 6. 简化3D视角
    fig.add_trace(
        go.Scatter3d(x=df['missile_x'][::10], y=df['missile_y'][::10], z=df['missile_z'][::10],
                    mode='lines', name='导弹3D',
                    line=dict(color='red', width=4)),
        row=2, col=3
    )

    cloud_df_sample = df.dropna(subset=['cloud_x'])[::5]
    if len(cloud_df_sample) > 0:
        fig.add_trace(
            go.Scatter3d(x=cloud_df_sample['cloud_x'], y=cloud_df_sample['cloud_y'], z=cloud_df_sample['cloud_z'],
                        mode='markers', name='云团3D',
                        marker=dict(size=5, color='gray', opacity=0.7)),
            row=2, col=3
        )

    fig.update_layout(
        title='烟幕干扰效果综合分析仪表板',
        height=1000,
        showlegend=True
    )

    return fig

# 创建并显示仪表板
fig_dashboard = create_shielding_dashboard()
fig_dashboard.show()

# ============================================================================
# 第七步：创建动画效果
# ============================================================================

print("\n=== 第七步：创建动画效果 ===")

def create_animation():
    """创建动态动画"""
    # 选择关键时间点进行动画
    animation_times = np.arange(0, 25, 0.5)

    frames = []
    for t in animation_times:
        # 计算当前时刻各对象位置
        missile_pos = missile_position(t, M1_initial, missile_velocity)
        drone_pos = drone_position(t, FY1_initial, drone_velocity)

        if t >= t_deploy:
            bomb_pos = smoke_bomb_position(t, t_deploy, deploy_position, drone_velocity)
        else:
            bomb_pos = drone_pos.copy()

        cloud_pos = smoke_cloud_position(t, t_explode, explode_position)

        # 创建帧数据
        frame_data = []

        # 导弹轨迹（到当前时间）
        t_indices = df['time'] <= t
        frame_data.append(
            go.Scatter3d(
                x=df.loc[t_indices, 'missile_x'],
                y=df.loc[t_indices, 'missile_y'],
                z=df.loc[t_indices, 'missile_z'],
                mode='lines+markers',
                name='导弹轨迹',
                line=dict(color='red', width=4),
                marker=dict(size=2)
            )
        )

        # 当前导弹位置
        frame_data.append(
            go.Scatter3d(
                x=[missile_pos[0]], y=[missile_pos[1]], z=[missile_pos[2]],
                mode='markers',
                name='导弹当前位置',
                marker=dict(size=8, color='red', symbol='diamond')
            )
        )

        # 无人机轨迹
        frame_data.append(
            go.Scatter3d(
                x=df.loc[t_indices, 'drone_x'],
                y=df.loc[t_indices, 'drone_y'],
                z=df.loc[t_indices, 'drone_z'],
                mode='lines+markers',
                name='无人机轨迹',
                line=dict(color='blue', width=3),
                marker=dict(size=2)
            )
        )

        # 当前无人机位置
        frame_data.append(
            go.Scatter3d(
                x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],
                mode='markers',
                name='无人机当前位置',
                marker=dict(size=6, color='blue')
            )
        )

        # 烟幕弹位置
        if t >= t_deploy:
            frame_data.append(
                go.Scatter3d(
                    x=[bomb_pos[0]], y=[bomb_pos[1]], z=[bomb_pos[2]],
                    mode='markers',
                    name='烟幕弹',
                    marker=dict(size=5, color='orange')
                )
            )

        # 烟幕云团
        if cloud_pos is not None:
            frame_data.append(
                go.Scatter3d(
                    x=[cloud_pos[0]], y=[cloud_pos[1]], z=[cloud_pos[2]],
                    mode='markers',
                    name='烟幕云团',
                    marker=dict(size=15, color='gray', opacity=0.7)
                )
            )

        # 目标点
        frame_data.extend([
            go.Scatter3d(
                x=[fake_target[0]], y=[fake_target[1]], z=[fake_target[2]],
                mode='markers',
                name='假目标',
                marker=dict(size=8, color='black', symbol='diamond')
            ),
            go.Scatter3d(
                x=[real_target[0]], y=[real_target[1]], z=[real_target[2]],
                mode='markers',
                name='真目标',
                marker=dict(size=8, color='green', symbol='square')
            )
        ])

        frames.append(go.Frame(data=frame_data, name=f't={t:.1f}s'))

    # 创建初始图形
    fig_anim = go.Figure(
        data=frames[0].data,
        frames=frames
    )

    # 添加播放控件
    fig_anim.update_layout(
        title='烟幕干扰弹动态仿真',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            yaxis=dict(
                dtick=50,  # Y轴刻度间隔50m
                tickmode='linear'
            ),
            xaxis=dict(
                dtick=2000,  # X轴刻度间隔2000m
                tickmode='linear'
            ),
            zaxis=dict(
                dtick=200,  # Z轴刻度间隔200m
                tickmode='linear'
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '播放',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 200, 'redraw': True},
                                   'fromcurrent': True}]
                },
                {
                    'label': '暂停',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}}]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[f't={t:.1f}s'], {'frame': {'duration': 0, 'redraw': True},
                                              'mode': 'immediate'}],
                    'label': f't={t:.1f}s',
                    'method': 'animate'
                }
                for t in animation_times
            ],
            'active': 0,
            'currentvalue': {'prefix': '时间: '},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }],
        width=1000,
        height=800
    )

    return fig_anim

# 创建并显示动画
fig_animation = create_animation()
fig_animation.show()

# ============================================================================
# 第八步：输出最终结果和总结
# ============================================================================

print("\n" + "="*60)
print("烟幕干扰弹对M1的有效遮蔽时长计算 - 最终结果")
print("="*60)

print(f"\n📊 核心计算结果:")
print(f"   • 有效遮蔽时长: {shielding_duration:.3f} 秒")
print(f"   • 遮蔽效率: {(shielding_duration/effective_duration*100):.1f}%")

if len(shielded_df) > 0:
    print(f"\n⏰ 时间节点:")
    print(f"   • 投放时间: {t_deploy:.1f}s")
    print(f"   • 起爆时间: {t_explode:.1f}s")
    print(f"   • 遮蔽开始: {shielded_df['time'].min():.3f}s")
    print(f"   • 遮蔽结束: {shielded_df['time'].max():.3f}s")
    print(f"   • 云团消失: {t_explode + effective_duration:.1f}s")

    print(f"\n📏 几何参数:")
    print(f"   • 最小遮蔽距离: {shielded_df['shield_distance'].min():.3f}m")
    print(f"   • 最大遮蔽距离: {shielded_df['shield_distance'].max():.3f}m")
    print(f"   • 平均遮蔽距离: {shielded_df['shield_distance'].mean():.3f}m")

print(f"\n📍 关键位置:")
print(f"   • 投放位置: ({deploy_position[0]:.1f}, {deploy_position[1]:.1f}, {deploy_position[2]:.1f})")
print(f"   • 起爆位置: ({explode_position[0]:.1f}, {explode_position[1]:.1f}, {explode_position[2]:.1f})")

print(f"\n🎯 可视化图表已生成:")
print(f"   • 3D轨迹可视化")
print(f"   • 时间序列分析图")
print(f"   • 遮蔽效果仪表板")
print(f"   • 动态仿真动画")

print(f"\n✅ 计算完成！所有图表已在Jupyter Notebook中显示。")
print("="*60)

# ============================================================================
# 第九步：保存图表和数据文件
# ============================================================================

print("\n=== 第九步：保存图表和数据文件 ===")

# 创建输出目录
output_dir = "../ImageOutput/01"
os.makedirs(output_dir, exist_ok=True)

# 保存3D轨迹图
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_visualization.html")
print(f"📈 3D轨迹图已保存: {output_dir}/01_3d_trajectory_visualization.html")

# 保存时间序列分析图
fig_time.write_html(f"{output_dir}/02_time_series_analysis.html")
print(f"📊 时间序列分析图已保存: {output_dir}/02_time_series_analysis.html")

# 保存仪表板
fig_dashboard.write_html(f"{output_dir}/03_shielding_dashboard.html")
print(f"📋 遮蔽效果仪表板已保存: {output_dir}/03_shielding_dashboard.html")

# 保存动画
fig_animation.write_html(f"{output_dir}/04_dynamic_animation.html")
print(f"🎬 动态仿真动画已保存: {output_dir}/04_dynamic_animation.html")

# 保存轨迹数据到CSV文件
df.to_csv(f'{output_dir}/05_trajectory_data.csv', index=False)
print(f"💾 轨迹数据已保存: {output_dir}/05_trajectory_data.csv")

# 保存计算结果摘要
result_summary = {
    "问题": "问题1 - 单弹固定参数分析",
    "有效遮蔽时长(秒)": round(shielding_duration, 3),
    "投放时间(秒)": t_deploy,
    "起爆时间(秒)": t_explode,
    "起爆位置": explode_position.tolist(),
    "遮蔽记录数量": len(shielded_df) if len(shielded_df) > 0 else 0,
    "遮蔽效率(%)": round((shielding_duration/effective_duration*100), 2) if effective_duration > 0 else 0,
    "计算参数": {
        "时间步长": 0.01,
        "有效半径": effective_radius,
        "有效持续时间": effective_duration,
        "导弹速度": missile_speed,
        "无人机速度": drone_speed
    }
}

with open(f'{output_dir}/06_results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=2)
print(f"📋 结果摘要已保存: {output_dir}/06_results_summary.json")

print(f"\n✅ 所有结果已保存到 ImageOutput/01/ 目录")

print(f"\n📁 输出文件:")
print(f"   📈 01_3d_trajectory_visualization.html - 3D轨迹交互图")
print(f"   📊 02_time_series_analysis.html - 时间序列分析")
print(f"   📋 03_shielding_dashboard.html - 遮蔽效果仪表板")
print(f"   🎬 04_dynamic_animation.html - 动态仿真动画")
print(f"   💾 05_trajectory_data.csv - 详细轨迹数据")
print(f"   📋 06_results_summary.json - 完整结果汇总")
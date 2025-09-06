"""
问题2：优化单枚烟幕弹投放策略 - 可视化版本
基于02_problem2_optimization.py的建模思路

包含完整的优化过程可视化、参数空间分析、轨迹对比等功能
使用Plotly和Matplotlib进行美观的数据可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution
from typing import Tuple, List, Dict, Optional
import pandas as pd
import json
import os
from datetime import datetime
import warnings
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# ============================================================================
# 第一步：定义基本参数和常量
# ============================================================================

print("=== 第一步：定义基本参数 ===")

# 物理常量
g = 9.8  # 重力加速度 m/s²
smoke_sink_speed = 3.0  # 烟幕云团下沉速度 m/s
effective_radius = 10.0  # 有效遮蔽半径 m
effective_duration = 20.0  # 有效遮蔽持续时间 s

# 导弹参数（M1）
missile_speed = 300.0  # 导弹速度 m/s
M1_initial = np.array([20000.0, 0.0, 2000.0])  # 导弹M1初始位置

# 无人机参数（FY1）
FY1_initial = np.array([17800.0, 0.0, 1800.0])  # 无人机FY1初始位置

# 目标位置
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
real_target = np.array([0.0, 200.0, 0.0])  # 真目标位置

# 优化约束
theta_range = [0, 2*np.pi]  # 航向角范围 [rad]
v_range = [70, 140]  # 速度范围 [m/s]
t_d_range = [0, 40]  # 投放时间范围 [s]
tau_range = [0, 20]  # 起爆延迟范围 [s]

# 计算导弹速度向量
missile_norm = np.linalg.norm(M1_initial)
missile_velocity = -missile_speed * M1_initial / missile_norm

# 导弹到达假目标的时间
t_max = missile_norm / missile_speed

print(f"导弹初始位置: {M1_initial}")
print(f"导弹速度向量: {missile_velocity}")
print(f"导弹到达假目标时间: {t_max:.2f}s")
print(f"无人机初始位置: {FY1_initial}")
print(f"真目标位置: {real_target}")

# ============================================================================
# 第二步：运动模型函数
# ============================================================================

print("\n=== 第二步：运动模型函数 ===")

def missile_position(t: float) -> np.ndarray:
    """计算导弹M1在时刻t的位置"""
    return M1_initial + missile_velocity * t

def drone_position(t: float, theta: float, v: float) -> np.ndarray:
    """计算无人机FY1在时刻t的位置"""
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    return FY1_initial + velocity * t

def smoke_bomb_position(t: float, t_deploy: float, theta: float, v: float) -> Optional[np.ndarray]:
    """计算烟幕弹在时刻t的位置"""
    if t < t_deploy:
        return None
    
    dt = t - t_deploy
    deploy_pos = drone_position(t_deploy, theta, v)
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    
    position = deploy_pos.copy()
    position[:2] += velocity[:2] * dt
    position[2] -= 0.5 * g * dt**2
    
    return position

def smoke_cloud_position(t: float, t_explode: float, explode_pos: np.ndarray) -> Optional[np.ndarray]:
    """计算烟幕云团在时刻t的位置"""
    if t < t_explode or t > t_explode + effective_duration:
        return None
    
    dt = t - t_explode
    position = explode_pos.copy()
    position[2] -= smoke_sink_speed * dt
    
    return position

def point_to_line_segment_distance(point: np.ndarray, line_start: np.ndarray, 
                                 line_end: np.ndarray) -> Tuple[float, float]:
    """计算点到线段的最短距离和投影参数u"""
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

print("运动模型函数定义完成")

# ============================================================================
# 第三步：遮蔽效果计算函数
# ============================================================================

print("\n=== 第三步：遮蔽效果计算函数 ===")

def calculate_shielding_duration(theta: float, v: float, t_d: float, tau: float, 
                               dt: float = 0.1, verbose: bool = False) -> Tuple[float, Dict]:
    """计算给定参数下的有效遮蔽时长"""
    t_explode = t_d + tau
    
    # 检查物理约束
    if t_explode + effective_duration > t_max:
        return 0.0, {"valid": False, "reason": "时间约束违反"}
    
    # 计算起爆位置
    explode_pos = smoke_bomb_position(t_explode, t_d, theta, v)
    if explode_pos is None or explode_pos[2] <= 0:
        return 0.0, {"valid": False, "reason": "起爆位置无效"}
    
    # 时间采样
    t_start = t_explode
    t_end = t_explode + effective_duration
    time_points = np.arange(t_start, t_end + dt, dt)
    
    shielded_count = 0
    detailed_records = []
    
    for t in time_points:
        missile_pos = missile_position(float(t))
        cloud_pos = smoke_cloud_position(float(t), t_explode, explode_pos)
        
        if cloud_pos is None or cloud_pos[2] <= 0:
            continue
        
        distance, u = point_to_line_segment_distance(cloud_pos, missile_pos, real_target)
        is_shielded = distance <= effective_radius and 0 <= u <= 1
        
        if is_shielded:
            shielded_count += 1
            if verbose:
                detailed_records.append({
                    "time": t,
                    "distance": distance,
                    "projection_u": u,
                    "missile_pos": missile_pos.copy(),
                    "cloud_pos": cloud_pos.copy()
                })
    
    total_duration = shielded_count * dt
    
    info = {
        "valid": True,
        "theta": theta,
        "theta_deg": np.degrees(theta),
        "v": v,
        "t_d": t_d,
        "tau": tau,
        "t_explode": t_explode,
        "explode_pos": explode_pos.tolist(),
        "explode_height": explode_pos[2],
        "shielded_count": shielded_count,
        "total_duration": total_duration,
        "time_step": dt,
        "records": detailed_records if verbose else []
    }
    
    return total_duration, info

# ============================================================================
# 第四步：优化算法实现
# ============================================================================

print("\n=== 第四步：优化算法实现 ===")

def objective_function(params: np.ndarray) -> float:
    """优化目标函数"""
    theta, v, t_d, tau = params
    
    if not (theta_range[0] <= theta <= theta_range[1]):
        return 1000.0
    if not (v_range[0] <= v <= v_range[1]):
        return 1000.0
    if not (t_d_range[0] <= t_d <= t_d_range[1]):
        return 1000.0
    if not (tau_range[0] <= tau <= tau_range[1]):
        return 1000.0
    
    duration, _ = calculate_shielding_duration(theta, v, t_d, tau)
    return -duration

def grid_search_optimization(theta_steps: int = 12, v_steps: int = 8, 
                           t_d_steps: int = 21, tau_steps: int = 21) -> Dict:
    """网格搜索优化"""
    print(f"开始网格搜索优化...")
    print(f"搜索空间: {theta_steps} × {v_steps} × {t_d_steps} × {tau_steps} = {theta_steps*v_steps*t_d_steps*tau_steps:,} 个点")
    
    # 参数网格
    theta_grid = np.linspace(theta_range[0], theta_range[1], theta_steps, endpoint=False)
    v_grid = np.linspace(v_range[0], v_range[1], v_steps)
    t_d_grid = np.linspace(t_d_range[0], t_d_range[1], t_d_steps)
    tau_grid = np.linspace(tau_range[0], tau_range[1], tau_steps)
    
    best_duration = 0.0
    best_params = None
    best_info = None
    
    results = []
    total_points = len(theta_grid) * len(v_grid) * len(t_d_grid) * len(tau_grid)
    processed = 0
    valid_count = 0
    
    for i, theta in enumerate(theta_grid):
        for j, v in enumerate(v_grid):
            for k, t_d in enumerate(t_d_grid):
                for m, tau in enumerate(tau_grid):
                    duration, info = calculate_shielding_duration(theta, v, t_d, tau)
                    
                    results.append({
                        'theta': theta,
                        'theta_deg': np.degrees(theta),
                        'v': v,
                        't_d': t_d,
                        'tau': tau,
                        't_explode': t_d + tau,
                        'duration': duration,
                        'valid': info['valid']
                    })
                    
                    if info['valid']:
                        valid_count += 1
                        if duration > best_duration:
                            best_duration = duration
                            best_params = (theta, v, t_d, tau)
                            best_info = info
                    
                    processed += 1
                    if processed % 5000 == 0:
                        print(f"进度: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%)")
    
    print(f"网格搜索完成！最优遮蔽时长: {best_duration:.3f}s")
    
    return {
        'best_duration': best_duration,
        'best_params': best_params,
        'best_info': best_info,
        'all_results': results,
        'search_stats': {
            'total_points': total_points,
            'valid_points': valid_count,
            'validity_rate': valid_count / total_points if total_points > 0 else 0
        }
    }

# 执行优化
print("执行网格搜索优化...")
grid_result = grid_search_optimization()

# ============================================================================
# 第五步：创建可视化图表
# ============================================================================

print("\n=== 第五步：创建可视化图表 ===")

def create_parameter_space_analysis():
    """创建参数空间分析图表"""
    print("创建参数空间分析图表...")
    
    # 转换结果为DataFrame
    df = pd.DataFrame(grid_result['all_results'])
    df_valid = df[df['valid'] == True].copy()
    
    if len(df_valid) == 0:
        print("警告：没有有效的优化结果")
        return None
    
    # 创建Plotly子图
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '航向角 vs 遮蔽时长', '速度 vs 遮蔽时长', '投放时间 vs 遮蔽时长',
            '起爆延迟 vs 遮蔽时长', '参数相关性分析', '最优解分布'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "heatmap"}, {"type": "scatter3d"}]]
    )
    
    # 1. 航向角 vs 遮蔽时长
    fig.add_trace(
        go.Scatter(
            x=df_valid['theta_deg'],
            y=df_valid['duration'],
            mode='markers',
            name='航向角',
            marker=dict(
                color=df_valid['duration'],
                colorscale='Viridis',
                size=4,
                colorbar=dict(title="遮蔽时长(s)", x=0.32)
            ),
            hovertemplate='航向角: %{x:.1f}°<br>遮蔽时长: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 速度 vs 遮蔽时长
    fig.add_trace(
        go.Scatter(
            x=df_valid['v'],
            y=df_valid['duration'],
            mode='markers',
            name='速度',
            marker=dict(
                color=df_valid['duration'],
                colorscale='Plasma',
                size=4,
                colorbar=dict(title="遮蔽时长(s)", x=0.65)
            ),
            hovertemplate='速度: %{x:.1f}m/s<br>遮蔽时长: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. 投放时间 vs 遮蔽时长
    fig.add_trace(
        go.Scatter(
            x=df_valid['t_d'],
            y=df_valid['duration'],
            mode='markers',
            name='投放时间',
            marker=dict(
                color=df_valid['duration'],
                colorscale='Cividis',
                size=4,
                colorbar=dict(title="遮蔽时长(s)", x=0.98)
            ),
            hovertemplate='投放时间: %{x:.1f}s<br>遮蔽时长: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=3
    )
    
    # 4. 起爆延迟 vs 遮蔽时长
    fig.add_trace(
        go.Scatter(
            x=df_valid['tau'],
            y=df_valid['duration'],
            mode='markers',
            name='起爆延迟',
            marker=dict(
                color=df_valid['duration'],
                colorscale='Turbo',
                size=4,
                colorbar=dict(title="遮蔽时长(s)", x=0.32, y=0.15)
            ),
            hovertemplate='起爆延迟: %{x:.1f}s<br>遮蔽时长: %{y:.3f}s<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 5. 参数相关性热图
    if len(df_valid) > 100:
        df_sample = df_valid.sample(n=min(1000, len(df_valid)), random_state=42)
    else:
        df_sample = df_valid
    
    corr_matrix = df_sample[['theta_deg', 'v', 't_d', 'tau', 'duration']].corr()
    
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=['航向角(°)', '速度(m/s)', '投放时间(s)', '起爆延迟(s)', '遮蔽时长(s)'],
            y=['航向角(°)', '速度(m/s)', '投放时间(s)', '起爆延迟(s)', '遮蔽时长(s)'],
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='相关系数: %{z:.3f}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 6. 最优解3D分布
    top_results = df_valid.nlargest(50, 'duration')
    
    fig.add_trace(
        go.Scatter3d(
            x=top_results['theta_deg'],
            y=top_results['v'],
            z=top_results['duration'],
            mode='markers',
            name='Top 50解',
            marker=dict(
                color=top_results['duration'],
                colorscale='Viridis',
                size=6,
                colorbar=dict(title="遮蔽时长(s)", x=0.98, y=0.15)
            ),
            hovertemplate='航向角: %{x:.1f}°<br>速度: %{y:.1f}m/s<br>遮蔽时长: %{z:.3f}s<extra></extra>'
        ),
        row=2, col=3
    )
    
    # 标记最优解
    if grid_result['best_params']:
        theta_opt, v_opt, t_d_opt, tau_opt = grid_result['best_params']
        best_duration = grid_result['best_duration']
        
        # 在各个子图中标记最优解
        fig.add_trace(
            go.Scatter3d(
                x=[np.degrees(theta_opt)],
                y=[v_opt],
                z=[best_duration],
                mode='markers',
                name='最优解',
                marker=dict(color='red', size=12, symbol='diamond'),
                hovertemplate='最优解<br>航向角: %{x:.1f}°<br>速度: %{y:.1f}m/s<br>遮蔽时长: %{z:.3f}s<extra></extra>'
            ),
            row=2, col=3
        )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': '问题2：单弹最优投放策略 - 参数空间分析',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=False,
        font=dict(size=12)
    )
    
    # 更新轴标签
    fig.update_xaxes(title_text="航向角 (°)", row=1, col=1)
    fig.update_xaxes(title_text="速度 (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="投放时间 (s)", row=1, col=3)
    fig.update_xaxes(title_text="起爆延迟 (s)", row=2, col=1)
    
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=1)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=2)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=3)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=2, col=1)
    
    return fig

def create_trajectory_comparison():
    """创建轨迹对比图"""
    print("创建轨迹对比图...")
    
    if not grid_result['best_params']:
        print("无最优参数，跳过轨迹对比")
        return None
    
    # 最优参数
    theta_opt, v_opt, t_d_opt, tau_opt = grid_result['best_params']
    
    # 默认参数（问题1的参数）
    theta_default = np.pi  # 180度
    v_default = 120.0
    t_d_default = 1.5
    tau_default = 3.6
    
    # 计算轨迹
    t_max_sim = min(30, t_max)
    time_points = np.arange(0, t_max_sim, 0.1)
    
    # 轨迹数据
    missile_traj = np.array([missile_position(t) for t in time_points])
    opt_drone_traj = np.array([drone_position(t, theta_opt, v_opt) for t in time_points])
    def_drone_traj = np.array([drone_position(t, theta_default, v_default) for t in time_points])
    
    # 创建3D图
    fig = go.Figure()
    
    # 导弹轨迹
    fig.add_trace(go.Scatter3d(
        x=missile_traj[:, 0], y=missile_traj[:, 1], z=missile_traj[:, 2],
        mode='lines',
        name='导弹M1轨迹',
        line=dict(color='red', width=6),
        hovertemplate='导弹位置<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
    ))
    
    # 最优无人机轨迹
    fig.add_trace(go.Scatter3d(
        x=opt_drone_traj[:, 0], y=opt_drone_traj[:, 1], z=opt_drone_traj[:, 2],
        mode='lines',
        name=f'最优无人机轨迹 (θ={np.degrees(theta_opt):.1f}°)',
        line=dict(color='blue', width=6),
        hovertemplate='最优无人机位置<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
    ))
    
    # 默认无人机轨迹
    fig.add_trace(go.Scatter3d(
        x=def_drone_traj[:, 0], y=def_drone_traj[:, 1], z=def_drone_traj[:, 2],
        mode='lines',
        name=f'默认无人机轨迹 (θ={np.degrees(theta_default):.1f}°)',
        line=dict(color='gray', width=4, dash='dash'),
        hovertemplate='默认无人机位置<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
    ))
    
    # 关键点标记
    # 假目标
    fig.add_trace(go.Scatter3d(
        x=[fake_target[0]], y=[fake_target[1]], z=[fake_target[2]],
        mode='markers',
        name='假目标',
        marker=dict(size=12, color='orange', symbol='square'),
        hovertemplate='假目标<br>位置: (0, 0, 0)<extra></extra>'
    ))
    
    # 真目标
    fig.add_trace(go.Scatter3d(
        x=[real_target[0]], y=[real_target[1]], z=[real_target[2]],
        mode='markers',
        name='真目标',
        marker=dict(size=12, color='green', symbol='square'),
        hovertemplate='真目标<br>位置: (0, 200, 0)<extra></extra>'
    ))
    
    # 最优投放点
    opt_deploy_pos = drone_position(t_d_opt, theta_opt, v_opt)
    fig.add_trace(go.Scatter3d(
        x=[opt_deploy_pos[0]], y=[opt_deploy_pos[1]], z=[opt_deploy_pos[2]],
        mode='markers',
        name='最优投放点',
        marker=dict(size=10, color='blue', symbol='cross'),
        hovertemplate='最优投放点<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
    ))
    
    # 最优起爆点
    opt_explode_pos = smoke_bomb_position(t_d_opt + tau_opt, t_d_opt, theta_opt, v_opt)
    if opt_explode_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[opt_explode_pos[0]], y=[opt_explode_pos[1]], z=[opt_explode_pos[2]],
            mode='markers',
            name='最优起爆点',
            marker=dict(size=10, color='blue', symbol='diamond'),
            hovertemplate='最优起爆点<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
        ))
    
    # 默认投放点
    def_deploy_pos = drone_position(t_d_default, theta_default, v_default)
    fig.add_trace(go.Scatter3d(
        x=[def_deploy_pos[0]], y=[def_deploy_pos[1]], z=[def_deploy_pos[2]],
        mode='markers',
        name='默认投放点',
        marker=dict(size=8, color='gray', symbol='cross'),
        hovertemplate='默认投放点<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
    ))
    
    # 默认起爆点
    def_explode_pos = smoke_bomb_position(t_d_default + tau_default, t_d_default, theta_default, v_default)
    if def_explode_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[def_explode_pos[0]], y=[def_explode_pos[1]], z=[def_explode_pos[2]],
            mode='markers',
            name='默认起爆点',
            marker=dict(size=8, color='gray', symbol='diamond'),
            hovertemplate='默认起爆点<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '问题2：最优解与默认解轨迹对比',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            yaxis=dict(dtick=100, tickmode='linear'),
            xaxis=dict(dtick=2000, tickmode='linear'),
            zaxis=dict(dtick=200, tickmode='linear'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1000,
        height=800,
        font=dict(size=12)
    )
    
    return fig

def create_optimization_summary():
    """创建优化结果摘要图表"""
    print("创建优化结果摘要图表...")
    
    if not grid_result['best_params']:
        print("无最优参数，跳过摘要图表")
        return None
    
    theta_opt, v_opt, t_d_opt, tau_opt = grid_result['best_params']
    best_duration = grid_result['best_duration']
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            '优化效果对比', '最优参数雷达图', '遮蔽时长分布', '参数敏感性'
        ],
        specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
               [{"type": "histogram"}, {"type": "scatter"}]]
    )
    
    # 1. 优化效果对比
    comparison_data = {
        '方法': ['问题1 (固定参数)', '问题2 (优化参数)'],
        '遮蔽时长': [1.380, best_duration],
        '颜色': ['lightcoral', 'lightblue']
    }
    
    fig.add_trace(
        go.Bar(
            x=comparison_data['方法'],
            y=comparison_data['遮蔽时长'],
            marker_color=comparison_data['颜色'],
            text=[f'{x:.3f}s' for x in comparison_data['遮蔽时长']],
            textposition='auto',
            name='遮蔽时长对比',
            hovertemplate='%{x}<br>遮蔽时长: %{y:.3f}s<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. 最优参数雷达图
    # 归一化参数值
    theta_norm = np.degrees(theta_opt) / 360
    v_norm = (v_opt - v_range[0]) / (v_range[1] - v_range[0])
    t_d_norm = t_d_opt / t_d_range[1]
    tau_norm = tau_opt / tau_range[1]
    
    fig.add_trace(
        go.Scatterpolar(
            r=[theta_norm, v_norm, t_d_norm, tau_norm, theta_norm],
            theta=['航向角', '速度', '投放时间', '起爆延迟', '航向角'],
            fill='toself',
            name='最优参数',
            line_color='blue',
            hovertemplate='%{theta}<br>归一化值: %{r:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. 遮蔽时长分布
    df_valid = pd.DataFrame([r for r in grid_result['all_results'] if r['valid']])
    if len(df_valid) > 0:
        fig.add_trace(
            go.Histogram(
                x=df_valid['duration'],
                nbinsx=30,
                name='遮蔽时长分布',
                marker_color='lightgreen',
                opacity=0.7,
                hovertemplate='遮蔽时长: %{x:.3f}s<br>频次: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 标记最优解
        fig.add_vline(
            x=best_duration,
            line_dash="dash",
            line_color="red",
            annotation_text=f"最优解: {best_duration:.3f}s",
            row=2, col=1
        )
    
    # 4. 参数敏感性分析（简化版）
    if len(df_valid) > 0:
        # 计算各参数与遮蔽时长的相关性
        correlations = df_valid[['theta_deg', 'v', 't_d', 'tau']].corrwith(df_valid['duration'])
        
        fig.add_trace(
            go.Scatter(
                x=['航向角', '速度', '投放时间', '起爆延迟'],
                y=[correlations['theta_deg'], correlations['v'], correlations['t_d'], correlations['tau']],
                mode='markers+lines',
                name='参数相关性',
                marker=dict(size=10, color='red'),
                line=dict(color='red', width=2),
                hovertemplate='%{x}<br>相关系数: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': '问题2：优化结果摘要分析',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=False,
        font=dict(size=12)
    )
    
    # 更新轴标签
    fig.update_xaxes(title_text="方法", row=1, col=1)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=1)
    
    fig.update_xaxes(title_text="遮蔽时长 (s)", row=2, col=1)
    fig.update_yaxes(title_text="频次", row=2, col=1)
    
    fig.update_xaxes(title_text="参数", row=2, col=2)
    fig.update_yaxes(title_text="与遮蔽时长的相关系数", row=2, col=2)
    
    return fig

def create_matplotlib_analysis():
    """使用Matplotlib创建补充分析图表"""
    print("创建Matplotlib补充分析图表...")
    
    if not grid_result['best_params']:
        print("无最优参数，跳过Matplotlib图表")
        return None
    
    df_valid = pd.DataFrame([r for r in grid_result['all_results'] if r['valid']])
    if len(df_valid) == 0:
        print("无有效数据，跳过Matplotlib图表")
        return None
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('问题2：单弹最优投放策略 - 详细分析', fontsize=16, fontweight='bold')
    
    # 1. 航向角分布
    axes[0, 0].hist(df_valid['theta_deg'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(np.degrees(grid_result['best_params'][0]), color='red', linestyle='--', 
                       label=f'最优值: {np.degrees(grid_result["best_params"][0]):.1f}°')
    axes[0, 0].set_xlabel('航向角 (°)')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('航向角分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 速度分布
    axes[0, 1].hist(df_valid['v'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].axvline(grid_result['best_params'][1], color='red', linestyle='--',
                       label=f'最优值: {grid_result["best_params"][1]:.1f} m/s')
    axes[0, 1].set_xlabel('速度 (m/s)')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].set_title('速度分布')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 遮蔽时长热图（航向角 vs 速度）
    pivot_data = df_valid.pivot_table(values='duration', index='theta_deg', columns='v', aggfunc='mean')
    im = axes[0, 2].imshow(pivot_data.values, cmap='viridis', aspect='auto', origin='lower')
    axes[0, 2].set_xlabel('速度索引')
    axes[0, 2].set_ylabel('航向角索引')
    axes[0, 2].set_title('遮蔽时长热图 (航向角 vs 速度)')
    plt.colorbar(im, ax=axes[0, 2], label='遮蔽时长 (s)')
    
    # 4. 投放时间 vs 起爆延迟散点图
    scatter = axes[1, 0].scatter(df_valid['t_d'], df_valid['tau'], 
                                c=df_valid['duration'], cmap='plasma', alpha=0.6, s=20)
    axes[1, 0].scatter(grid_result['best_params'][2], grid_result['best_params'][3], 
                       color='red', s=100, marker='*', label='最优解', edgecolor='black', linewidth=2)
    axes[1, 0].set_xlabel('投放时间 (s)')
    axes[1, 0].set_ylabel('起爆延迟 (s)')
    axes[1, 0].set_title('投放时间 vs 起爆延迟')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='遮蔽时长 (s)')
    
    # 5. 遮蔽时长箱线图（按航向角区间）
    df_valid['theta_bin'] = pd.cut(df_valid['theta_deg'], bins=8, labels=[f'{i*45}-{(i+1)*45}°' for i in range(8)])
    df_valid.boxplot(column='duration', by='theta_bin', ax=axes[1, 1])
    axes[1, 1].set_xlabel('航向角区间')
    axes[1, 1].set_ylabel('遮蔽时长 (s)')
    axes[1, 1].set_title('不同航向角区间的遮蔽时长分布')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. 优化收敛分析（Top N解的分布）
    top_solutions = df_valid.nlargest(20, 'duration')
    axes[1, 2].plot(range(1, len(top_solutions)+1), top_solutions['duration'].values, 
                    'bo-', markersize=8, linewidth=2)
    axes[1, 2].set_xlabel('解的排名')
    axes[1, 2].set_ylabel('遮蔽时长 (s)')
    axes[1, 2].set_title('Top 20 解的遮蔽时长')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# 创建所有可视化图表
print("开始创建可视化图表...")

# Plotly图表
fig_param_space = create_parameter_space_analysis()
fig_trajectory = create_trajectory_comparison()
fig_summary = create_optimization_summary()

# Matplotlib图表
fig_matplotlib = create_matplotlib_analysis()

# 显示图表
if fig_param_space:
    fig_param_space.show()

if fig_trajectory:
    fig_trajectory.show()

if fig_summary:
    fig_summary.show()

if fig_matplotlib:
    plt.show()

# ============================================================================
# 第六步：保存结果和图表
# ============================================================================

print("\n=== 第六步：保存结果和图表 ===")

# 创建输出目录
output_dir = "../ImageOutput/02"
os.makedirs(output_dir, exist_ok=True)

# 保存Plotly图表
if fig_param_space:
    fig_param_space.write_html(f"{output_dir}/01_parameter_space_analysis.html")
    print(f"参数空间分析图已保存: {output_dir}/01_parameter_space_analysis.html")

if fig_trajectory:
    fig_trajectory.write_html(f"{output_dir}/02_trajectory_comparison.html")
    print(f"轨迹对比图已保存: {output_dir}/02_trajectory_comparison.html")

if fig_summary:
    fig_summary.write_html(f"{output_dir}/03_optimization_summary.html")
    print(f"优化摘要图已保存: {output_dir}/03_optimization_summary.html")

# 保存Matplotlib图表
if fig_matplotlib:
    fig_matplotlib.savefig(f"{output_dir}/04_detailed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"详细分析图已保存: {output_dir}/04_detailed_analysis.png")

# 保存优化结果数据
if grid_result['best_params']:
    theta_opt, v_opt, t_d_opt, tau_opt = grid_result['best_params']
    
    results_summary = {
        "问题": "问题2 - 单弹最优投放策略优化可视化",
        "生成时间": datetime.now().isoformat(),
        "最优解": {
            "有效遮蔽时长_s": round(grid_result['best_duration'], 3),
            "航向角_度": round(np.degrees(theta_opt), 2),
            "航向角_弧度": round(theta_opt, 4),
            "速度_m_per_s": round(v_opt, 2),
            "投放时间_s": round(t_d_opt, 2),
            "起爆延迟_s": round(tau_opt, 2),
            "起爆时间_s": round(t_d_opt + tau_opt, 2)
        },
        "优化统计": grid_result.get('search_stats', {}),
        "改进效果": {
            "问题1遮蔽时长_s": 1.380,
            "问题2遮蔽时长_s": round(grid_result['best_duration'], 3),
            "绝对改进_s": round(grid_result['best_duration'] - 1.380, 3),
            "相对改进_percent": round((grid_result['best_duration'] - 1.380) / 1.380 * 100, 1)
        },
        "生成的图表": [
            "01_parameter_space_analysis.html - 参数空间分析",
            "02_trajectory_comparison.html - 轨迹对比",
            "03_optimization_summary.html - 优化摘要",
            "04_detailed_analysis.png - 详细分析"
        ]
    }
    
    with open(f"{output_dir}/05_visualization_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"可视化结果摘要已保存: {output_dir}/05_visualization_results.json")

# 保存网格搜索数据
df_results = pd.DataFrame(grid_result['all_results'])
df_results.to_csv(f"{output_dir}/06_optimization_data.csv", index=False, encoding='utf-8')
print(f"优化数据已保存: {output_dir}/06_optimization_data.csv")

print("="*60)
print("问题2可视化分析完成！")
if grid_result['best_params']:
    theta_opt, v_opt, t_d_opt, tau_opt = grid_result['best_params']
    print(f"最优遮蔽时长: {grid_result['best_duration']:.3f} 秒")
    print(f"最优参数:")
    print(f"  航向角: {np.degrees(theta_opt):.2f}°")
    print(f"  速度: {v_opt:.2f} m/s")
    print(f"  投放时间: {t_d_opt:.2f} s")
    print(f"  起爆延迟: {tau_opt:.2f} s")
    improvement = (grid_result['best_duration'] - 1.380) / 1.380 * 100
    print(f"相比问题1改进: {improvement:+.1f}%")
print(f"所有图表和数据已保存到: {output_dir}/")
print("="*60)
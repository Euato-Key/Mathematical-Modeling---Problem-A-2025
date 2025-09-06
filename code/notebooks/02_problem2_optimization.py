"""
问题2：优化单枚烟幕弹投放策略
基于03-02-A1-P2-单弹最优投放策略.md的建模思路

优化目标：最大化有效遮蔽时长
决策变量：航向角θ、速度v、投放时间t_d、起爆延迟τ
约束条件：θ∈[0,2π)、v∈[70,140]、t_d≥0、τ≥0
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import differential_evolution, minimize
from typing import Tuple, List, Dict, Optional
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# 目标位置
fake_target = np.array([0.0, 0.0, 0.0])  # 假目标位置
real_target = np.array([0.0, 200.0, 0.0])  # 真目标位置

# 优化约束
theta_range = [0, 2*np.pi]  # 航向角范围 [rad]
v_range = [70, 140]  # 速度范围 [m/s]
t_d_range = [0, 40]  # 投放时间范围 [s]
tau_range = [0, 20]  # 起爆延迟范围 [s]

# 导弹到达假目标的时间
missile_norm = np.linalg.norm(M1_initial)
t_max = missile_norm / missile_speed
print(f"导弹到达假目标时间: {t_max:.2f}s")

# 计算导弹速度向量
missile_velocity = -missile_speed * M1_initial / missile_norm
print(f"导弹速度向量: {missile_velocity}")

print(f"优化参数范围:")
print(f"  航向角θ: {theta_range[0]:.2f} - {theta_range[1]:.2f} rad")
print(f"  速度v: {v_range[0]} - {v_range[1]} m/s")
print(f"  投放时间t_d: {t_d_range[0]} - {t_d_range[1]} s")
print(f"  起爆延迟τ: {tau_range[0]} - {tau_range[1]} s")

# ============================================================================
# 第二步：运动模型函数
# ============================================================================

print("\n=== 第二步：运动模型函数 ===")

def missile_position(t: float) -> np.ndarray:
    """计算导弹在时刻t的位置"""
    return M1_initial + missile_velocity * t

def drone_position(t: float, theta: float, v: float) -> np.ndarray:
    """计算无人机在时刻t的位置"""
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
    """计算点到线段的最短距离和投影参数"""
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
    """
    计算给定参数下的有效遮蔽时长
    
    参数:
        theta: 航向角 [rad]
        v: 速度 [m/s]
        t_d: 投放时间 [s]
        tau: 起爆延迟 [s]
        dt: 时间步长 [s]
        verbose: 是否输出详细信息
    
    返回:
        (有效遮蔽时长, 详细信息字典)
    """
    # 计算关键时间点
    t_explode = t_d + tau
    
    # 检查物理约束
    if t_explode + effective_duration > t_max:
        if verbose:
            print(f"时间约束违反: t_explode + 20 = {t_explode + effective_duration:.2f} > t_max = {t_max:.2f}")
        return 0.0, {"valid": False, "reason": "时间约束违反"}
    
    # 计算起爆位置
    explode_pos = smoke_bomb_position(t_explode, t_d, theta, v)
    if explode_pos is None or explode_pos[2] <= 0:
        if verbose:
            print(f"起爆位置无效: {explode_pos}")
        return 0.0, {"valid": False, "reason": "起爆位置无效"}
    
    # 时间采样
    t_start = t_explode
    t_end = t_explode + effective_duration
    time_points = np.arange(t_start, t_end + dt, dt)
    
    shielded_count = 0
    detailed_records = []
    
    for t in time_points:
        # 计算各位置
        missile_pos = missile_position(t)
        cloud_pos = smoke_cloud_position(t, t_explode, explode_pos)
        
        if cloud_pos is None or cloud_pos[2] <= 0:
            continue
        
        # 计算遮蔽条件
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
        "v": v,
        "t_d": t_d,
        "tau": tau,
        "t_explode": t_explode,
        "explode_pos": explode_pos.tolist() if explode_pos is not None else None,
        "shielded_count": shielded_count,
        "total_duration": total_duration,
        "records": detailed_records if verbose else []
    }
    
    return total_duration, info

# 测试函数
print("测试遮蔽效果计算...")
test_theta = np.pi  # 180度
test_v = 120.0
test_t_d = 1.5
test_tau = 3.6

test_duration, test_info = calculate_shielding_duration(test_theta, test_v, test_t_d, test_tau, verbose=True)
print(f"测试结果: 遮蔽时长 = {test_duration:.3f}s")

# ============================================================================
# 第四步：优化算法实现
# ============================================================================

print("\n=== 第四步：优化算法实现 ===")

def objective_function(params: np.ndarray) -> float:
    """
    优化目标函数（最小化负的遮蔽时长）
    
    参数:
        params: [theta, v, t_d, tau]
    
    返回:
        负的遮蔽时长
    """
    theta, v, t_d, tau = params
    
    # 参数约束检查
    if not (theta_range[0] <= theta <= theta_range[1]):
        return 1000.0
    if not (v_range[0] <= v <= v_range[1]):
        return 1000.0
    if not (t_d_range[0] <= t_d <= t_d_range[1]):
        return 1000.0
    if not (tau_range[0] <= tau <= tau_range[1]):
        return 1000.0
    
    duration, _ = calculate_shielding_duration(theta, v, t_d, tau)
    return -duration  # 最小化负值等于最大化正值

def grid_search_optimization(theta_steps: int = 20, v_steps: int = 8, 
                           t_d_steps: int = 41, tau_steps: int = 41) -> Dict:
    """
    网格搜索优化
    
    参数:
        theta_steps: 航向角离散点数
        v_steps: 速度离散点数
        t_d_steps: 投放时间离散点数
        tau_steps: 起爆延迟离散点数
    
    返回:
        优化结果字典
    """
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
    
    for i, theta in enumerate(theta_grid):
        for j, v in enumerate(v_grid):
            for k, t_d in enumerate(t_d_grid):
                for m, tau in enumerate(tau_grid):
                    duration, info = calculate_shielding_duration(theta, v, t_d, tau)
                    
                    results.append({
                        'theta': theta,
                        'v': v,
                        't_d': t_d,
                        'tau': tau,
                        'duration': duration,
                        'theta_deg': np.degrees(theta),
                        'valid': info['valid']
                    })
                    
                    if duration > best_duration:
                        best_duration = duration
                        best_params = (theta, v, t_d, tau)
                        best_info = info
                    
                    processed += 1
                    if processed % 10000 == 0:
                        print(f"进度: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%)")
    
    print(f"网格搜索完成！")
    print(f"最优遮蔽时长: {best_duration:.3f}s")
    print(f"最优参数: θ={np.degrees(best_params[0]):.1f}°, v={best_params[1]:.1f}m/s, t_d={best_params[2]:.1f}s, τ={best_params[3]:.1f}s")
    
    return {
        'best_duration': best_duration,
        'best_params': best_params,
        'best_info': best_info,
        'all_results': results,
        'search_stats': {
            'total_points': total_points,
            'valid_points': sum(1 for r in results if r['valid']),
            'theta_range_deg': [np.degrees(theta_range[0]), np.degrees(theta_range[1])],
            'v_range': v_range,
            't_d_range': t_d_range,
            'tau_range': tau_range
        }
    }

def differential_evolution_optimization() -> Dict:
    """
    差分进化算法优化
    
    返回:
        优化结果字典
    """
    print("开始差分进化优化...")
    
    # 参数边界
    bounds = [
        (theta_range[0], theta_range[1]),  # theta
        (v_range[0], v_range[1]),          # v
        (t_d_range[0], t_d_range[1]),      # t_d
        (tau_range[0], tau_range[1])       # tau
    ]
    
    # 运行优化
    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        disp=True
    )
    
    if result.success:
        theta_opt, v_opt, t_d_opt, tau_opt = result.x
        duration_opt = -result.fun
        
        # 获取详细信息
        _, info = calculate_shielding_duration(theta_opt, v_opt, t_d_opt, tau_opt, verbose=True)
        
        print(f"差分进化优化完成！")
        print(f"最优遮蔽时长: {duration_opt:.3f}s")
        print(f"最优参数: θ={np.degrees(theta_opt):.1f}°, v={v_opt:.1f}m/s, t_d={t_d_opt:.1f}s, τ={tau_opt:.1f}s")
        
        return {
            'success': True,
            'best_duration': duration_opt,
            'best_params': (theta_opt, v_opt, t_d_opt, tau_opt),
            'best_info': info,
            'optimization_result': result
        }
    else:
        print("差分进化优化失败！")
        return {'success': False, 'message': result.message}

# ============================================================================
# 第五步：执行优化
# ============================================================================

print("\n=== 第五步：执行优化 ===")

# 方法1：网格搜索（粗搜索）
print("方法1：网格搜索")
grid_result = grid_search_optimization(theta_steps=12, v_steps=8, t_d_steps=21, tau_steps=21)

# 方法2：差分进化算法
print("\n方法2：差分进化算法")
de_result = differential_evolution_optimization()

# 比较结果
print("\n=== 优化结果比较 ===")
print(f"网格搜索最优解: {grid_result['best_duration']:.3f}s")
if de_result['success']:
    print(f"差分进化最优解: {de_result['best_duration']:.3f}s")
    
    # 选择更好的结果
    if de_result['best_duration'] > grid_result['best_duration']:
        final_result = de_result
        print("采用差分进化结果作为最终解")
    else:
        final_result = grid_result
        print("采用网格搜索结果作为最终解")
else:
    final_result = grid_result
    print("采用网格搜索结果作为最终解")

# 输出最终结果
theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
print(f"\n=== 最终优化结果 ===")
print(f"最大有效遮蔽时长: {final_result['best_duration']:.3f} 秒")
print(f"最优航向角: {np.degrees(theta_opt):.2f}°")
print(f"最优速度: {v_opt:.2f} m/s")
print(f"最优投放时间: {t_d_opt:.2f} s")
print(f"最优起爆延迟: {tau_opt:.2f} s")
print(f"起爆时间: {t_d_opt + tau_opt:.2f} s")

# ============================================================================
# 第六步：结果分析和可视化
# ============================================================================

print("\n=== 第六步：结果分析和可视化 ===")

def create_optimization_analysis():
    """创建优化结果分析图表"""
    
    # 转换网格搜索结果为DataFrame
    df = pd.DataFrame(grid_result['all_results'])
    df_valid = df[df['valid'] == True].copy()
    
    if len(df_valid) == 0:
        print("警告：没有有效的优化结果")
        return None
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            '航向角 vs 遮蔽时长', '速度 vs 遮蔽时长', '投放时间 vs 遮蔽时长',
            '起爆延迟 vs 遮蔽时长', '参数相关性热图', '最优解轨迹'
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
            marker=dict(color=df_valid['duration'], colorscale='Viridis', size=4)
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
            marker=dict(color=df_valid['duration'], colorscale='Viridis', size=4)
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
            marker=dict(color=df_valid['duration'], colorscale='Viridis', size=4)
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
            marker=dict(color=df_valid['duration'], colorscale='Viridis', size=4)
        ),
        row=2, col=1
    )
    
    # 5. 参数相关性分析
    if len(df_valid) > 100:
        # 采样以提高性能
        df_sample = df_valid.sample(n=min(1000, len(df_valid)), random_state=42)
    else:
        df_sample = df_valid
    
    corr_matrix = df_sample[['theta_deg', 'v', 't_d', 'tau', 'duration']].corr()
    
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ),
        row=2, col=2
    )
    
    # 6. 最优解3D可视化
    top_results = df_valid.nlargest(100, 'duration')
    
    fig.add_trace(
        go.Scatter3d(
            x=top_results['theta_deg'],
            y=top_results['v'],
            z=top_results['duration'],
            mode='markers',
            name='Top 100',
            marker=dict(
                color=top_results['duration'],
                colorscale='Viridis',
                size=5
            )
        ),
        row=2, col=3
    )
    
    # 标记最优解
    best_row = df_valid.loc[df_valid['duration'].idxmax()]
    fig.add_trace(
        go.Scatter3d(
            x=[best_row['theta_deg']],
            y=[best_row['v']],
            z=[best_row['duration']],
            mode='markers',
            name='最优解',
            marker=dict(color='red', size=10, symbol='diamond')
        ),
        row=2, col=3
    )
    
    # 添加最优解标记到其他图
    for row, col, x_col in [(1, 1, 'theta_deg'), (1, 2, 'v'), (1, 3, 't_d'), (2, 1, 'tau')]:
        fig.add_hline(y=best_row['duration'], line_dash="dash", line_color="red", row=row, col=col)
        fig.add_vline(x=best_row[x_col], line_dash="dash", line_color="red", row=row, col=col)
    
    # 更新布局
    fig.update_layout(
        title='单弹最优投放策略优化分析',
        height=800,
        showlegend=False
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
    """创建最优解与默认解的轨迹对比"""
    
    # 最优参数
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    
    # 默认参数（问题1的参数）
    theta_default = np.pi  # 180度，朝向假目标
    v_default = 120.0
    t_d_default = 1.5
    tau_default = 3.6
    
    # 计算轨迹
    t_max_sim = min(30, t_max)
    time_points = np.arange(0, t_max_sim, 0.1)
    
    # 最优解轨迹
    opt_missile_traj = np.array([missile_position(t) for t in time_points])
    opt_drone_traj = np.array([drone_position(t, theta_opt, v_opt) for t in time_points])
    
    # 默认解轨迹
    def_missile_traj = np.array([missile_position(t) for t in time_points])
    def_drone_traj = np.array([drone_position(t, theta_default, v_default) for t in time_points])
    
    # 创建3D图
    fig = go.Figure()
    
    # 导弹轨迹
    fig.add_trace(go.Scatter3d(
        x=opt_missile_traj[:, 0], y=opt_missile_traj[:, 1], z=opt_missile_traj[:, 2],
        mode='lines',
        name='导弹M1轨迹',
        line=dict(color='red', width=4)
    ))
    
    # 最优无人机轨迹
    fig.add_trace(go.Scatter3d(
        x=opt_drone_traj[:, 0], y=opt_drone_traj[:, 1], z=opt_drone_traj[:, 2],
        mode='lines',
        name=f'最优无人机轨迹 (θ={np.degrees(theta_opt):.1f}°)',
        line=dict(color='blue', width=4)
    ))
    
    # 默认无人机轨迹
    fig.add_trace(go.Scatter3d(
        x=def_drone_traj[:, 0], y=def_drone_traj[:, 1], z=def_drone_traj[:, 2],
        mode='lines',
        name=f'默认无人机轨迹 (θ={np.degrees(theta_default):.1f}°)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # 关键点标记
    # 假目标
    fig.add_trace(go.Scatter3d(
        x=[fake_target[0]], y=[fake_target[1]], z=[fake_target[2]],
        mode='markers',
        name='假目标',
        marker=dict(size=10, color='orange', symbol='square')
    ))
    
    # 真目标
    fig.add_trace(go.Scatter3d(
        x=[real_target[0]], y=[real_target[1]], z=[real_target[2]],
        mode='markers',
        name='真目标',
        marker=dict(size=10, color='green', symbol='square')
    ))
    
    # 最优投放点
    opt_deploy_pos = drone_position(t_d_opt, theta_opt, v_opt)
    fig.add_trace(go.Scatter3d(
        x=[opt_deploy_pos[0]], y=[opt_deploy_pos[1]], z=[opt_deploy_pos[2]],
        mode='markers',
        name='最优投放点',
        marker=dict(size=8, color='blue', symbol='cross')
    ))
    
    # 最优起爆点
    opt_explode_pos = smoke_bomb_position(t_d_opt + tau_opt, t_d_opt, theta_opt, v_opt)
    if opt_explode_pos is not None:
        fig.add_trace(go.Scatter3d(
            x=[opt_explode_pos[0]], y=[opt_explode_pos[1]], z=[opt_explode_pos[2]],
            mode='markers',
            name='最优起爆点',
            marker=dict(size=8, color='blue', symbol='diamond')
        ))
    
    # 默认投放点
    def_deploy_pos = drone_position(t_d_default, theta_default, v_default)
    fig.add_trace(go.Scatter3d(
        x=[def_deploy_pos[0]], y=[def_deploy_pos[1]], z=[def_deploy_pos[2]],
        mode='markers',
        name='默认投放点',
        marker=dict(size=6, color='gray', symbol='cross')
    ))
    
    fig.update_layout(
        title='最优解与默认解轨迹对比',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            yaxis=dict(dtick=50, tickmode='linear'),
            xaxis=dict(dtick=2000, tickmode='linear'),
            zaxis=dict(dtick=200, tickmode='linear')
        ),
        width=1000,
        height=800
    )
    
    return fig

# 创建分析图表
print("创建优化分析图表...")
fig_analysis = create_optimization_analysis()
if fig_analysis:
    fig_analysis.show()

print("创建轨迹对比图...")
fig_trajectory = create_trajectory_comparison()
fig_trajectory.show()

# ============================================================================
# 第七步：敏感性分析
# ============================================================================

print("\n=== 第七步：敏感性分析 ===")

def sensitivity_analysis():
    """对最优解进行敏感性分析"""
    
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    base_duration = final_result['best_duration']
    
    # 参数扰动范围
    perturbations = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2]
    
    sensitivity_results = {
        'theta': [],
        'v': [],
        't_d': [],
        'tau': []
    }
    
    print("进行敏感性分析...")
    
    # 航向角敏感性
    for p in perturbations:
        theta_new = theta_opt + p * np.pi/6  # ±30度范围
        if theta_range[0] <= theta_new <= theta_range[1]:
            duration, _ = calculate_shielding_duration(theta_new, v_opt, t_d_opt, tau_opt)
            sensitivity_results['theta'].append({
                'perturbation': p,
                'value': np.degrees(theta_new),
                'duration': duration,
                'change': duration - base_duration
            })
    
    # 速度敏感性
    for p in perturbations:
        v_new = v_opt + p * 35  # ±35 m/s范围
        if v_range[0] <= v_new <= v_range[1]:
            duration, _ = calculate_shielding_duration(theta_opt, v_new, t_d_opt, tau_opt)
            sensitivity_results['v'].append({
                'perturbation': p,
                'value': v_new,
                'duration': duration,
                'change': duration - base_duration
            })
    
    # 投放时间敏感性
    for p in perturbations:
        t_d_new = t_d_opt + p * 10  # ±10s范围
        if t_d_range[0] <= t_d_new <= t_d_range[1]:
            duration, _ = calculate_shielding_duration(theta_opt, v_opt, t_d_new, tau_opt)
            sensitivity_results['t_d'].append({
                'perturbation': p,
                'value': t_d_new,
                'duration': duration,
                'change': duration - base_duration
            })
    
    # 起爆延迟敏感性
    for p in perturbations:
        tau_new = tau_opt + p * 5  # ±5s范围
        if tau_range[0] <= tau_new <= tau_range[1]:
            duration, _ = calculate_shielding_duration(theta_opt, v_opt, t_d_opt, tau_new)
            sensitivity_results['tau'].append({
                'perturbation': p,
                'value': tau_new,
                'duration': duration,
                'change': duration - base_duration
            })
    
    return sensitivity_results

def create_sensitivity_plot(sensitivity_results):
    """创建敏感性分析图表"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['航向角敏感性', '速度敏感性', '投放时间敏感性', '起爆延迟敏感性']
    )
    
    # 航向角
    if sensitivity_results['theta']:
        data = sensitivity_results['theta']
        fig.add_trace(
            go.Scatter(
                x=[d['value'] for d in data],
                y=[d['duration'] for d in data],
                mode='lines+markers',
                name='航向角',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    # 速度
    if sensitivity_results['v']:
        data = sensitivity_results['v']
        fig.add_trace(
            go.Scatter(
                x=[d['value'] for d in data],
                y=[d['duration'] for d in data],
                mode='lines+markers',
                name='速度',
                line=dict(color='red')
            ),
            row=1, col=2
        )
    
    # 投放时间
    if sensitivity_results['t_d']:
        data = sensitivity_results['t_d']
        fig.add_trace(
            go.Scatter(
                x=[d['value'] for d in data],
                y=[d['duration'] for d in data],
                mode='lines+markers',
                name='投放时间',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    
    # 起爆延迟
    if sensitivity_results['tau']:
        data = sensitivity_results['tau']
        fig.add_trace(
            go.Scatter(
                x=[d['value'] for d in data],
                y=[d['duration'] for d in data],
                mode='lines+markers',
                name='起爆延迟',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title='参数敏感性分析',
        height=600,
        showlegend=False
    )
    
    # 更新轴标签
    fig.update_xaxes(title_text="航向角 (°)", row=1, col=1)
    fig.update_xaxes(title_text="速度 (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="投放时间 (s)", row=2, col=1)
    fig.update_xaxes(title_text="起爆延迟 (s)", row=2, col=2)
    
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=1)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=1, col=2)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=2, col=1)
    fig.update_yaxes(title_text="遮蔽时长 (s)", row=2, col=2)
    
    return fig

# 执行敏感性分析
sensitivity_results = sensitivity_analysis()
fig_sensitivity = create_sensitivity_plot(sensitivity_results)
fig_sensitivity.show()

# ============================================================================
# 第八步：保存结果
# ============================================================================

print("\n=== 第八步：保存结果 ===")

# 创建输出目录
output_dir = "../../ImageOutput/02"
os.makedirs(output_dir, exist_ok=True)

# 保存图表
if fig_analysis:
    fig_analysis.write_html(f"{output_dir}/01_optimization_analysis.html")
    print(f"优化分析图表已保存: {output_dir}/01_optimization_analysis.html")

fig_trajectory.write_html(f"{output_dir}/02_trajectory_comparison.html")
print(f"轨迹对比图表已保存: {output_dir}/02_trajectory_comparison.html")

fig_sensitivity.write_html(f"{output_dir}/03_sensitivity_analysis.html")
print(f"敏感性分析图表已保存: {output_dir}/03_sensitivity_analysis.html")

# 保存优化结果数据
results_summary = {
    "问题": "问题2 - 单弹最优投放策略优化",
    "优化时间": datetime.now().isoformat(),
    "最优解": {
        "有效遮蔽时长": round(final_result['best_duration'], 3),
        "航向角_度": round(np.degrees(final_result['best_params'][0]), 2),
        "航向角_弧度": round(final_result['best_params'][0], 4),
        "速度_m_per_s": round(final_result['best_params'][1], 2),
        "投放时间_s": round(final_result['best_params'][2], 2),
        "起爆延迟_s": round(final_result['best_params'][3], 2),
        "起爆时间_s": round(final_result['best_params'][2] + final_result['best_params'][3], 2)
    },
    "优化统计": grid_result.get('search_stats', {}),
    "敏感性分析": sensitivity_results
}

with open(f"{output_dir}/04_optimization_results.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)
print(f"优化结果已保存: {output_dir}/04_optimization_results.json")

# 保存网格搜索详细数据
if 'all_results' in grid_result:
    df_results = pd.DataFrame(grid_result['all_results'])
    df_results.to_csv(f"{output_dir}/05_grid_search_data.csv", index=False, encoding='utf-8')
    print(f"网格搜索数据已保存: {output_dir}/05_grid_search_data.csv")

print("="*60)
print("问题2优化完成！")
print(f"最优遮蔽时长: {final_result['best_duration']:.3f} 秒")
theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
print(f"最优参数:")
print(f"  航向角: {np.degrees(theta_opt):.2f}°")
print(f"  速度: {v_opt:.2f} m/s")
print(f"  投放时间: {t_d_opt:.2f} s")
print(f"  起爆延迟: {tau_opt:.2f} s")
print(f"所有结果已保存到: {output_dir}/")
print("="*60)
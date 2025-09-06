"""
问题2：优化单枚烟幕弹投放策略
基于03-02-A1-P2-单弹最优投放策略.md的建模思路

优化目标：最大化有效遮蔽时长
决策变量：航向角θ、速度v、投放时间t_d、起爆延迟τ
约束条件：θ∈[0,2π)、v∈[70,140]、t_d≥0、τ≥0

理论建模过程，不包含可视化
"""

import numpy as np
from scipy.optimize import differential_evolution
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

print(f"\n优化参数范围:")
print(f"  航向角θ: {theta_range[0]:.2f} - {theta_range[1]:.2f} rad ({np.degrees(theta_range[0]):.0f}° - {np.degrees(theta_range[1]):.0f}°)")
print(f"  速度v: {v_range[0]} - {v_range[1]} m/s")
print(f"  投放时间t_d: {t_d_range[0]} - {t_d_range[1]} s")
print(f"  起爆延迟τ: {tau_range[0]} - {tau_range[1]} s")

# ============================================================================
# 第二步：运动模型函数
# ============================================================================

print("\n=== 第二步：运动模型函数 ===")

def missile_position(t: float) -> np.ndarray:
    """
    计算导弹M1在时刻t的位置
    公式: M(t) = M_0 + v_m * t
    """
    return M1_initial + missile_velocity * t

def drone_position(t: float, theta: float, v: float) -> np.ndarray:
    """
    计算无人机FY1在时刻t的位置
    公式: U(t) = U_0 + v_u * t
    其中 v_u = (v*cos(θ), v*sin(θ), 0)
    """
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    return FY1_initial + velocity * t

def smoke_bomb_position(t: float, t_deploy: float, theta: float, v: float) -> Optional[np.ndarray]:
    """
    计算烟幕弹在时刻t的位置
    公式: P_弹(s) = P_d + v_u * s + (1/2) * g * s^2 * k
    其中 s = t - t_deploy，k = (0, 0, -1)
    """
    if t < t_deploy:
        return None
    
    dt = t - t_deploy
    deploy_pos = drone_position(t_deploy, theta, v)
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    
    position = deploy_pos.copy()
    position[:2] += velocity[:2] * dt  # 水平方向保持初速度
    position[2] -= 0.5 * g * dt**2     # 竖直方向受重力影响
    
    return position

def smoke_cloud_position(t: float, t_explode: float, explode_pos: np.ndarray) -> Optional[np.ndarray]:
    """
    计算烟幕云团在时刻t的位置
    公式: C(t) = (x_b, y_b, z_b - 3*(t - t_b))
    """
    if t < t_explode or t > t_explode + effective_duration:
        return None
    
    dt = t - t_explode
    position = explode_pos.copy()
    position[2] -= smoke_sink_speed * dt  # 以3m/s速度下沉
    
    return position

def point_to_line_segment_distance(point: np.ndarray, line_start: np.ndarray, 
                                 line_end: np.ndarray) -> Tuple[float, float]:
    """
    计算点到线段的最短距离和投影参数u
    
    返回: (距离, 投影参数u)
    
    公式:
    u = (AP · AB) / ||AB||^2
    
    距离 d = {
        ||AP||                    if u < 0
        ||P - B||                 if u > 1  
        ||AP × AB|| / ||AB||      if 0 ≤ u ≤ 1
    }
    """
    AB = line_end - line_start
    AP = point - line_start
    
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return float(np.linalg.norm(AP)), 0.0
    
    u = float(np.dot(AP, AB) / AB_squared)
    
    if u < 0:
        # 最近点在线段起点之外
        distance = float(np.linalg.norm(AP))
    elif u > 1:
        # 最近点在线段终点之外
        BP = point - line_end
        distance = float(np.linalg.norm(BP))
    else:
        # 最近点在线段上
        cross_product = np.cross(AP, AB)
        if AB.ndim == 1 and len(AB) == 3:
            distance = float(np.linalg.norm(cross_product) / np.linalg.norm(AB))
        else:
            distance = float(abs(cross_product) / np.linalg.norm(AB))
    
    return distance, u

print("运动模型函数定义完成")

# 验证关键计算
print("\n验证关键计算:")
print(f"导弹速度模长: {np.linalg.norm(missile_velocity):.3f} m/s")
print(f"导弹初始位置模长: {missile_norm:.3f} m")

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
    
    算法流程:
    1. 计算起爆时间 t_b = t_d + τ
    2. 检查物理约束
    3. 计算起爆位置
    4. 在时间区间 [t_b, t_b+20] 内以步长 dt 采样
    5. 对每个时间点检查遮蔽条件
    6. 统计满足条件的时间点数
    """
    # 计算关键时间点
    t_explode = t_d + tau
    
    # 检查时间约束: t_b + 20 ≤ t_max
    if t_explode + effective_duration > t_max:
        if verbose:
            print(f"时间约束违反: t_explode + 20 = {t_explode + effective_duration:.2f} > t_max = {t_max:.2f}")
        return 0.0, {"valid": False, "reason": "时间约束违反"}
    
    # 计算起爆位置
    explode_pos = smoke_bomb_position(t_explode, t_d, theta, v)
    if explode_pos is None:
        if verbose:
            print("起爆位置计算失败")
        return 0.0, {"valid": False, "reason": "起爆位置无效"}
    
    # 检查高度约束: z_b > 0
    if explode_pos[2] <= 0:
        if verbose:
            print(f"高度约束违反: z_b = {explode_pos[2]:.3f} ≤ 0")
        return 0.0, {"valid": False, "reason": "起爆高度过低"}
    
    # 时间采样
    t_start = t_explode
    t_end = t_explode + effective_duration
    time_points = np.arange(t_start, t_end + dt, dt)
    
    shielded_count = 0
    detailed_records = []
    
    for t in time_points:
        # 计算导弹位置
        missile_pos = missile_position(t)
        
        # 计算云团位置
        cloud_pos = smoke_cloud_position(t, t_explode, explode_pos)
        
        if cloud_pos is None or cloud_pos[2] <= 0:
            continue
        
        # 计算遮蔽几何条件
        distance, u = point_to_line_segment_distance(cloud_pos, missile_pos, real_target)
        
        # 检查遮蔽条件
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

# 测试函数
print("测试遮蔽效果计算...")
test_theta = np.pi  # 180度，朝向假目标
test_v = 120.0
test_t_d = 1.5
test_tau = 3.6

test_duration, test_info = calculate_shielding_duration(test_theta, test_v, test_t_d, test_tau, verbose=True)
print(f"测试结果:")
print(f"  参数: θ={np.degrees(test_theta):.1f}°, v={test_v}m/s, t_d={test_t_d}s, τ={test_tau}s")
print(f"  起爆时间: {test_info['t_explode']:.2f}s")
print(f"  起爆位置: {test_info['explode_pos']}")
print(f"  起爆高度: {test_info['explode_height']:.3f}m")
print(f"  遮蔽时长: {test_duration:.3f}s")
print(f"  遮蔽记录数: {test_info['shielded_count']}")

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
        负的遮蔽时长（用于最小化算法）
    """
    theta, v, t_d, tau = params
    
    # 参数边界检查
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
    
    按照建模文档的离散化策略:
    - θ: [0, 2π) 以 Δθ = π/18 离散化（20个点）
    - v: [70, 140] 以 Δv = 10 离散化（8个点）  
    - t_d: [0, 40] 以 Δt_d = 1 离散化（41个点）
    - τ: [0, 20] 以 Δτ = 0.5 离散化（41个点）
    
    总搜索点数: 20 × 8 × 41 × 41 = 268,960 个点
    """
    print(f"开始网格搜索优化...")
    print(f"搜索空间: {theta_steps} × {v_steps} × {t_d_steps} × {tau_steps} = {theta_steps*v_steps*t_d_steps*tau_steps:,} 个点")
    
    # 参数网格（按照建模文档的离散化）
    theta_grid = np.linspace(theta_range[0], theta_range[1], theta_steps, endpoint=False)
    v_grid = np.linspace(v_range[0], v_range[1], v_steps)
    t_d_grid = np.linspace(t_d_range[0], t_d_range[1], t_d_steps)
    tau_grid = np.linspace(tau_range[0], tau_range[1], tau_steps)
    
    print(f"参数网格:")
    print(f"  θ: {len(theta_grid)} 点, 范围 [{np.degrees(theta_grid[0]):.1f}°, {np.degrees(theta_grid[-1]):.1f}°]")
    print(f"  v: {len(v_grid)} 点, 范围 [{v_grid[0]:.1f}, {v_grid[-1]:.1f}] m/s")
    print(f"  t_d: {len(t_d_grid)} 点, 范围 [{t_d_grid[0]:.1f}, {t_d_grid[-1]:.1f}] s")
    print(f"  τ: {len(tau_grid)} 点, 范围 [{tau_grid[0]:.1f}, {tau_grid[-1]:.1f}] s")
    
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
                    if processed % 50000 == 0:
                        print(f"进度: {processed:,}/{total_points:,} ({100*processed/total_points:.1f}%), 有效解: {valid_count}")
    
    print(f"\n网格搜索完成！")
    print(f"总搜索点数: {total_points:,}")
    print(f"有效解数量: {valid_count:,} ({100*valid_count/total_points:.1f}%)")
    print(f"最优遮蔽时长: {best_duration:.3f}s")
    
    if best_params:
        theta_opt, v_opt, t_d_opt, tau_opt = best_params
        print(f"最优参数:")
        print(f"  航向角θ: {np.degrees(theta_opt):.2f}° ({theta_opt:.4f} rad)")
        print(f"  速度v: {v_opt:.2f} m/s")
        print(f"  投放时间t_d: {t_d_opt:.2f} s")
        print(f"  起爆延迟τ: {tau_opt:.2f} s")
        print(f"  起爆时间t_b: {t_d_opt + tau_opt:.2f} s")
        print(f"  起爆高度: {best_info['explode_height']:.3f} m")
    
    return {
        'best_duration': best_duration,
        'best_params': best_params,
        'best_info': best_info,
        'all_results': results,
        'search_stats': {
            'total_points': total_points,
            'valid_points': valid_count,
            'validity_rate': valid_count / total_points,
            'theta_range_deg': [np.degrees(theta_range[0]), np.degrees(theta_range[1])],
            'v_range': v_range,
            't_d_range': t_d_range,
            'tau_range': tau_range,
            'grid_sizes': [theta_steps, v_steps, t_d_steps, tau_steps]
        }
    }

def differential_evolution_optimization() -> Dict:
    """
    差分进化算法优化
    
    使用scipy.optimize.differential_evolution进行全局优化
    """
    print("开始差分进化优化...")
    
    # 参数边界
    bounds = [
        (theta_range[0], theta_range[1]),  # theta
        (v_range[0], v_range[1]),          # v
        (t_d_range[0], t_d_range[1]),      # t_d
        (tau_range[0], tau_range[1])       # tau
    ]
    
    print(f"参数边界:")
    print(f"  θ: [{np.degrees(bounds[0][0]):.1f}°, {np.degrees(bounds[0][1]):.1f}°]")
    print(f"  v: [{bounds[1][0]:.1f}, {bounds[1][1]:.1f}] m/s")
    print(f"  t_d: [{bounds[2][0]:.1f}, {bounds[2][1]:.1f}] s")
    print(f"  τ: [{bounds[3][0]:.1f}, {bounds[3][1]:.1f}] s")
    
    # 运行优化
    result = differential_evolution(
        objective_function,
        bounds,
        maxiter=100,
        popsize=15,
        seed=42,
        disp=True,
        atol=1e-6,
        tol=1e-6
    )
    
    if result.success:
        theta_opt, v_opt, t_d_opt, tau_opt = result.x
        duration_opt = -result.fun
        
        # 获取详细信息
        _, info = calculate_shielding_duration(theta_opt, v_opt, t_d_opt, tau_opt, verbose=True)
        
        print(f"\n差分进化优化完成！")
        print(f"优化成功: {result.success}")
        print(f"函数评估次数: {result.nfev}")
        print(f"迭代次数: {result.nit}")
        print(f"最优遮蔽时长: {duration_opt:.3f}s")
        print(f"最优参数:")
        print(f"  航向角θ: {np.degrees(theta_opt):.2f}° ({theta_opt:.4f} rad)")
        print(f"  速度v: {v_opt:.2f} m/s")
        print(f"  投放时间t_d: {t_d_opt:.2f} s")
        print(f"  起爆延迟τ: {tau_opt:.2f} s")
        print(f"  起爆时间t_b: {t_d_opt + tau_opt:.2f} s")
        print(f"  起爆高度: {info['explode_height']:.3f} m")
        
        return {
            'success': True,
            'best_duration': duration_opt,
            'best_params': (theta_opt, v_opt, t_d_opt, tau_opt),
            'best_info': info,
            'optimization_result': result,
            'function_evaluations': result.nfev,
            'iterations': result.nit
        }
    else:
        print("差分进化优化失败！")
        print(f"失败原因: {result.message}")
        return {'success': False, 'message': result.message}

# ============================================================================
# 第五步：执行优化
# ============================================================================

print("\n=== 第五步：执行优化 ===")

# 方法1：网格搜索（按照建模文档的参数）
print("方法1：网格搜索（按建模文档参数）")
grid_result = grid_search_optimization(theta_steps=20, v_steps=8, t_d_steps=41, tau_steps=41)

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
if final_result['best_params']:
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    print(f"\n=== 最终优化结果 ===")
    print(f"最大有效遮蔽时长: {final_result['best_duration']:.3f} 秒")
    print(f"最优参数:")
    print(f"  航向角θ: {np.degrees(theta_opt):.2f}° ({theta_opt:.4f} rad)")
    print(f"  速度v: {v_opt:.2f} m/s")
    print(f"  投放时间t_d: {t_d_opt:.2f} s")
    print(f"  起爆延迟τ: {tau_opt:.2f} s")
    print(f"  起爆时间t_b: {t_d_opt + tau_opt:.2f} s")
    print(f"  起爆位置: {final_result['best_info']['explode_pos']}")
    print(f"  起爆高度: {final_result['best_info']['explode_height']:.3f} m")
    
    # 与问题1结果对比
    print(f"\n与问题1对比:")
    print(f"  问题1遮蔽时长: 1.380s (固定参数)")
    print(f"  问题2遮蔽时长: {final_result['best_duration']:.3f}s (优化参数)")
    improvement = final_result['best_duration'] - 1.380
    improvement_pct = improvement / 1.380 * 100
    print(f"  改进效果: +{improvement:.3f}s ({improvement_pct:+.1f}%)")
else:
    print("未找到有效的优化解")

# ============================================================================
# 第六步：结果分析
# ============================================================================

print("\n=== 第六步：结果分析 ===")

def analyze_optimization_results():
    """分析优化结果"""
    
    if not final_result['best_params']:
        print("无有效结果可分析")
        return
    
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    
    print("1. 最优参数分析:")
    
    # 航向角分析
    theta_deg = np.degrees(theta_opt)
    if theta_deg < 90:
        direction = "东北方向"
    elif theta_deg < 180:
        direction = "西北方向"
    elif theta_deg < 270:
        direction = "西南方向"
    else:
        direction = "东南方向"
    print(f"   航向角: {theta_deg:.2f}° ({direction})")
    
    # 速度分析
    v_ratio = (v_opt - v_range[0]) / (v_range[1] - v_range[0])
    print(f"   速度: {v_opt:.2f} m/s (范围内{v_ratio:.1%}位置)")
    
    # 时间分析
    print(f"   投放时间: {t_d_opt:.2f}s")
    print(f"   起爆延迟: {tau_opt:.2f}s")
    print(f"   起爆时间: {t_d_opt + tau_opt:.2f}s")
    
    print("\n2. 物理约束检查:")
    explode_height = final_result['best_info']['explode_height']
    print(f"   起爆高度: {explode_height:.3f}m > 0 ✓")
    
    t_explode = t_d_opt + tau_opt
    time_margin = t_max - (t_explode + effective_duration)
    print(f"   时间约束: t_b + 20 = {t_explode + effective_duration:.2f}s < t_max = {t_max:.2f}s ✓")
    print(f"   时间余量: {time_margin:.2f}s")
    
    # 高度约束理论值
    tau_max_theory = np.sqrt(1800 / 4.9)
    print(f"   起爆延迟: τ = {tau_opt:.2f}s < τ_max = {tau_max_theory:.2f}s ✓")
    
    print("\n3. 几何分析:")
    # 计算投放点和起爆点
    deploy_pos = drone_position(t_d_opt, theta_opt, v_opt)
    explode_pos = np.array(final_result['best_info']['explode_pos'])
    
    print(f"   投放点: ({deploy_pos[0]:.1f}, {deploy_pos[1]:.1f}, {deploy_pos[2]:.1f})")
    print(f"   起爆点: ({explode_pos[0]:.1f}, {explode_pos[1]:.1f}, {explode_pos[2]:.1f})")
    
    # 计算与目标的几何关系
    dist_to_real_target = np.linalg.norm(explode_pos[:2] - real_target[:2])
    print(f"   起爆点到真目标水平距离: {dist_to_real_target:.1f}m")
    
    print("\n4. 优化效果:")
    if 'search_stats' in grid_result:
        stats = grid_result['search_stats']
        print(f"   搜索空间大小: {stats['total_points']:,} 个点")
        print(f"   有效解比例: {stats['validity_rate']:.1%}")
    
    print(f"   最优解遮蔽时长: {final_result['best_duration']:.3f}s")
    print(f"   相比问题1改进: {(final_result['best_duration'] - 1.380)/1.380*100:+.1f}%")

# 执行结果分析
analyze_optimization_results()

# ============================================================================
# 第七步：保存结果
# ============================================================================

print("\n=== 第七步：保存结果 ===")

# 创建输出目录
output_dir = "../../ImageOutput/02"
os.makedirs(output_dir, exist_ok=True)

# 保存优化结果数据
if final_result['best_params']:
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    
    results_summary = {
        "问题": "问题2 - 单弹最优投放策略优化",
        "优化时间": datetime.now().isoformat(),
        "最优解": {
            "有效遮蔽时长_s": round(final_result['best_duration'], 3),
            "航向角_度": round(np.degrees(theta_opt), 2),
            "航向角_弧度": round(theta_opt, 4),
            "速度_m_per_s": round(v_opt, 2),
            "投放时间_s": round(t_d_opt, 2),
            "起爆延迟_s": round(tau_opt, 2),
            "起爆时间_s": round(t_d_opt + tau_opt, 2),
            "起爆位置": final_result['best_info']['explode_pos'],
            "起爆高度_m": round(final_result['best_info']['explode_height'], 3)
        },
        "物理约束验证": {
            "时间约束": f"t_b + 20 = {t_d_opt + tau_opt + 20:.2f}s < t_max = {t_max:.2f}s",
            "高度约束": f"z_b = {final_result['best_info']['explode_height']:.3f}m > 0",
            "参数范围": "所有参数均在约束范围内"
        },
        "优化统计": grid_result.get('search_stats', {}),
        "改进效果": {
            "问题1遮蔽时长_s": 1.380,
            "问题2遮蔽时长_s": round(final_result['best_duration'], 3),
            "绝对改进_s": round(final_result['best_duration'] - 1.380, 3),
            "相对改进_percent": round((final_result['best_duration'] - 1.380) / 1.380 * 100, 1)
        }
    }
    
    with open(f"{output_dir}/optimization_results.json", 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"优化结果已保存: {output_dir}/optimization_results.json")

# 保存网格搜索详细数据
if 'all_results' in grid_result:
    df_results = pd.DataFrame(grid_result['all_results'])
    df_results.to_csv(f"{output_dir}/grid_search_data.csv", index=False, encoding='utf-8')
    print(f"网格搜索数据已保存: {output_dir}/grid_search_data.csv")

print("="*60)
print("问题2优化建模完成！")
if final_result['best_params']:
    print(f"最优遮蔽时长: {final_result['best_duration']:.3f} 秒")
    theta_opt, v_opt, t_d_opt, tau_opt = final_result['best_params']
    print(f"最优参数:")
    print(f"  航向角: {np.degrees(theta_opt):.2f}°")
    print(f"  速度: {v_opt:.2f} m/s")
    print(f"  投放时间: {t_d_opt:.2f} s")
    print(f"  起爆延迟: {tau_opt:.2f} s")
    print(f"相比问题1改进: {(final_result['best_duration'] - 1.380)/1.380*100:+.1f}%")
print(f"结果已保存到: {output_dir}/")
print("="*60)
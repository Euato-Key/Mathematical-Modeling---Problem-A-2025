"""
问题3：无人机FY1投放3枚烟幕弹对M1的最优干扰策略建模

基于Problem Analysis/03-03-A1-P3-FY1三弹时序策略.md的建模思路
采用多阶段混合优化策略：遗传算法 + 序列二次规划 + 动态规划
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Tuple, List, Dict, Optional
import json
import os
from datetime import datetime
import warnings
from dataclasses import dataclass

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
tau_max = np.sqrt(2 * 1800 / g)  # 最大起爆延迟（高度约束）

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
print(f"最大起爆延迟（高度约束）: {tau_max:.2f}s")

# ============================================================================
# 第二步：数据结构定义
# ============================================================================

@dataclass
class BombParameters:
    """单枚烟幕弹参数"""
    t_deploy: float  # 投放时间
    tau: float       # 起爆延迟
    
    @property
    def t_explode(self) -> float:
        return self.t_deploy + self.tau

@dataclass
class ThreeBombStrategy:
    """三弹策略参数"""
    theta: float  # 航向角
    v: float      # 速度
    bomb1: BombParameters
    bomb2: BombParameters
    bomb3: BombParameters
    
    def to_array(self) -> np.ndarray:
        """转换为数组形式"""
        return np.array([
            self.theta, self.v,
            self.bomb1.t_deploy, self.bomb1.tau,
            self.bomb2.t_deploy, self.bomb2.tau,
            self.bomb3.t_deploy, self.bomb3.tau
        ])
    
    @classmethod
    def from_array(cls, params: np.ndarray) -> 'ThreeBombStrategy':
        """从数组创建策略"""
        return cls(
            theta=params[0],
            v=params[1],
            bomb1=BombParameters(params[2], params[3]),
            bomb2=BombParameters(params[4], params[5]),
            bomb3=BombParameters(params[6], params[7])
        )

# ============================================================================
# 第三步：运动模型函数
# ============================================================================

print("\n=== 第三步：运动模型函数 ===")

def missile_position(t: float) -> np.ndarray:
    """计算导弹M1在时刻t的位置"""
    return M1_initial + missile_velocity * t

def drone_position(t: float, theta: float, v: float) -> np.ndarray:
    """计算无人机FY1在时刻t的位置"""
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    return FY1_initial + velocity * t

def smoke_bomb_trajectory(t: float, t_deploy: float, theta: float, v: float) -> Optional[np.ndarray]:
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

def explosion_position(bomb: BombParameters, theta: float, v: float) -> np.ndarray:
    """计算烟幕弹起爆点位置"""
    t_explode = bomb.t_explode
    deploy_pos = drone_position(bomb.t_deploy, theta, v)
    velocity = np.array([v * np.cos(theta), v * np.sin(theta), 0.0])
    
    dt = bomb.tau
    position = deploy_pos.copy()
    position[:2] += velocity[:2] * dt
    position[2] -= 0.5 * g * dt**2
    
    return position

def smoke_cloud_position(t: float, bomb: BombParameters, explode_pos: np.ndarray) -> Optional[np.ndarray]:
    """计算烟幕云团在时刻t的位置"""
    t_explode = bomb.t_explode
    
    if t < t_explode or t > t_explode + effective_duration:
        return None
    
    dt = t - t_explode
    position = explode_pos.copy()
    position[2] -= smoke_sink_speed * dt
    
    if position[2] <= 0:
        return None
    
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
# 第四步：约束检查函数
# ============================================================================

print("\n=== 第四步：约束检查函数 ===")

def check_strategy_constraints(strategy: ThreeBombStrategy) -> Tuple[bool, str]:
    """检查三弹策略是否满足所有约束"""
    
    # 基本参数约束
    if not (theta_range[0] <= strategy.theta < theta_range[1]):
        return False, f"航向角约束违反: {strategy.theta}"
    
    if not (v_range[0] <= strategy.v <= v_range[1]):
        return False, f"速度约束违反: {strategy.v}"
    
    # 投放时间顺序约束
    bombs = [strategy.bomb1, strategy.bomb2, strategy.bomb3]
    
    for i, bomb in enumerate(bombs):
        if bomb.t_deploy < 0:
            return False, f"弹{i+1}投放时间为负: {bomb.t_deploy}"
        
        if bomb.tau < 0:
            return False, f"弹{i+1}起爆延迟为负: {bomb.tau}"
        
        if bomb.tau > tau_max:
            return False, f"弹{i+1}起爆延迟超过高度约束: {bomb.tau} > {tau_max}"
    
    # 投放时间间隔约束
    if strategy.bomb2.t_deploy < strategy.bomb1.t_deploy + 1:
        return False, f"弹2投放时间间隔不足: {strategy.bomb2.t_deploy - strategy.bomb1.t_deploy} < 1"
    
    if strategy.bomb3.t_deploy < strategy.bomb2.t_deploy + 1:
        return False, f"弹3投放时间间隔不足: {strategy.bomb3.t_deploy - strategy.bomb2.t_deploy} < 1"
    
    # 起爆高度约束
    for i, bomb in enumerate(bombs):
        explode_pos = explosion_position(bomb, strategy.theta, strategy.v)
        if explode_pos[2] <= 0:
            return False, f"弹{i+1}起爆高度为负: {explode_pos[2]}"
    
    # 时间窗口约束
    for i, bomb in enumerate(bombs):
        if bomb.t_explode + effective_duration > t_max:
            return False, f"弹{i+1}遮蔽结束时间超过导弹到达时间: {bomb.t_explode + effective_duration} > {t_max}"
    
    return True, "所有约束满足"

def apply_constraint_penalty(strategy: ThreeBombStrategy) -> float:
    """计算约束违反的惩罚值"""
    penalty = 0.0
    
    # 基本参数约束惩罚
    if strategy.theta < theta_range[0] or strategy.theta >= theta_range[1]:
        penalty += 1000.0
    
    if strategy.v < v_range[0] or strategy.v > v_range[1]:
        penalty += 1000.0
    
    # 投放时间顺序约束惩罚
    bombs = [strategy.bomb1, strategy.bomb2, strategy.bomb3]
    
    for bomb in bombs:
        if bomb.t_deploy < 0:
            penalty += 1000.0
        if bomb.tau < 0:
            penalty += 1000.0
        if bomb.tau > tau_max:
            penalty += 1000.0
    
    # 时间间隔约束惩罚
    if strategy.bomb2.t_deploy < strategy.bomb1.t_deploy + 1:
        penalty += 500.0 * (strategy.bomb1.t_deploy + 1 - strategy.bomb2.t_deploy)
    
    if strategy.bomb3.t_deploy < strategy.bomb2.t_deploy + 1:
        penalty += 500.0 * (strategy.bomb2.t_deploy + 1 - strategy.bomb3.t_deploy)
    
    # 高度约束惩罚
    try:
        for bomb in bombs:
            explode_pos = explosion_position(bomb, strategy.theta, strategy.v)
            if explode_pos[2] <= 0:
                penalty += 1000.0
    except:
        penalty += 1000.0
    
    # 时间窗口约束惩罚
    for bomb in bombs:
        if bomb.t_explode + effective_duration > t_max:
            penalty += 100.0 * (bomb.t_explode + effective_duration - t_max)
    
    return penalty

print("约束检查函数定义完成")

# ============================================================================
# 第五步：遮蔽效果计算函数
# ============================================================================

print("\n=== 第五步：遮蔽效果计算函数 ===")

def calculate_single_bomb_shielding(t: float, bomb: BombParameters, explode_pos: np.ndarray) -> bool:
    """计算单枚烟幕弹在时刻t是否产生遮蔽"""
    # 检查时间窗口
    if t < bomb.t_explode or t > bomb.t_explode + effective_duration:
        return False
    
    # 计算云团位置
    cloud_pos = smoke_cloud_position(t, bomb, explode_pos)
    if cloud_pos is None:
        return False
    
    # 计算导弹位置
    missile_pos = missile_position(t)
    
    # 计算点到线段距离
    distance, u = point_to_line_segment_distance(cloud_pos, missile_pos, real_target)
    
    # 判断遮蔽条件
    return distance <= effective_radius and 0 <= u <= 1

def calculate_combined_shielding(strategy: ThreeBombStrategy, dt: float = 0.1, 
                               verbose: bool = False) -> Tuple[float, Dict]:
    """计算三弹联合遮蔽时长"""
    
    # 检查约束
    is_valid, reason = check_strategy_constraints(strategy)
    if not is_valid:
        penalty = apply_constraint_penalty(strategy)
        return 0.0, {"valid": False, "reason": reason, "penalty": penalty}
    
    # 计算起爆位置
    bombs = [strategy.bomb1, strategy.bomb2, strategy.bomb3]
    explode_positions = []
    
    for i, bomb in enumerate(bombs):
        try:
            explode_pos = explosion_position(bomb, strategy.theta, strategy.v)
            explode_positions.append(explode_pos)
        except Exception as e:
            return 0.0, {"valid": False, "reason": f"弹{i+1}起爆位置计算失败: {str(e)}"}
    
    # 确定时间范围
    t_start = min(bomb.t_explode for bomb in bombs)
    t_end = min(max(bomb.t_explode + effective_duration for bomb in bombs), t_max)
    
    if t_start >= t_end:
        return 0.0, {"valid": False, "reason": "无有效时间窗口"}
    
    # 时间采样
    time_points = np.arange(t_start, t_end + dt, dt)
    shielded_count = 0
    detailed_records = []
    
    for t in time_points:
        # 检查任意一枚弹是否产生遮蔽（联合遮蔽函数）
        is_shielded = False
        shielding_bombs = []
        
        for i, (bomb, explode_pos) in enumerate(zip(bombs, explode_positions)):
            if calculate_single_bomb_shielding(t, bomb, explode_pos):
                is_shielded = True
                shielding_bombs.append(i + 1)
        
        if is_shielded:
            shielded_count += 1
            if verbose:
                missile_pos = missile_position(t)
                detailed_records.append({
                    "time": t,
                    "shielding_bombs": shielding_bombs,
                    "missile_pos": missile_pos.tolist()
                })
    
    total_duration = shielded_count * dt
    
    # 计算各弹独立遮蔽时长
    individual_durations = []
    for i, (bomb, explode_pos) in enumerate(zip(bombs, explode_positions)):
        individual_count = 0
        for t in time_points:
            if calculate_single_bomb_shielding(t, bomb, explode_pos):
                individual_count += 1
        individual_durations.append(individual_count * dt)
    
    info = {
        "valid": True,
        "strategy": {
            "theta_deg": np.degrees(strategy.theta),
            "v": strategy.v,
            "bomb1": {"t_deploy": strategy.bomb1.t_deploy, "tau": strategy.bomb1.tau, "t_explode": strategy.bomb1.t_explode},
            "bomb2": {"t_deploy": strategy.bomb2.t_deploy, "tau": strategy.bomb2.tau, "t_explode": strategy.bomb2.t_explode},
            "bomb3": {"t_deploy": strategy.bomb3.t_deploy, "tau": strategy.bomb3.tau, "t_explode": strategy.bomb3.t_explode}
        },
        "explode_positions": [pos.tolist() for pos in explode_positions],
        "total_duration": total_duration,
        "individual_durations": individual_durations,
        "time_window": {"start": t_start, "end": t_end},
        "time_step": dt,
        "records": detailed_records if verbose else []
    }
    
    return total_duration, info

print("遮蔽效果计算函数定义完成")

# ============================================================================
# 第六步：优化算法实现
# ============================================================================

print("\n=== 第六步：优化算法实现 ===")

def objective_function(params: np.ndarray) -> float:
    """优化目标函数（最小化负遮蔽时长）"""
    try:
        strategy = ThreeBombStrategy.from_array(params)
        duration, info = calculate_combined_shielding(strategy)
        
        if not info['valid']:
            penalty = apply_constraint_penalty(strategy)
            return 1000.0 + penalty
        
        return -duration  # 最小化负值等于最大化正值
    
    except Exception as e:
        return 10000.0

def genetic_algorithm_optimization(pop_size: int = 100, max_generations: int = 50) -> Dict:
    """遗传算法全局优化"""
    print(f"开始遗传算法优化...")
    print(f"种群大小: {pop_size}, 最大代数: {max_generations}")
    
    # 参数边界
    bounds = [
        (0, 2*np.pi),           # theta
        (70, 140),              # v
        (0, 30),                # t_d1
        (0, tau_max),           # tau1
        (1, 31),                # t_d2 (至少比t_d1大1)
        (0, tau_max),           # tau2
        (2, 32),                # t_d3 (至少比t_d2大1)
        (0, tau_max)            # tau3
    ]
    
    # 使用scipy的differential_evolution
    result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        popsize=pop_size//len(bounds),  # scipy的popsize是每个参数的倍数
        maxiter=max_generations,
        atol=1e-6,
        tol=1e-6,
        polish=True,
        disp=True
    )
    
    best_params = result.x
    best_strategy = ThreeBombStrategy.from_array(best_params)
    best_duration, best_info = calculate_combined_shielding(best_strategy, verbose=True)
    
    print(f"遗传算法优化完成！最优遮蔽时长: {best_duration:.3f}s")
    
    return {
        'best_duration': best_duration,
        'best_strategy': best_strategy,
        'best_info': best_info,
        'optimization_result': result,
        'function_evaluations': result.nfev
    }

def local_refinement_optimization(initial_strategy: ThreeBombStrategy) -> Dict:
    """局部精细化优化（序列二次规划）"""
    print("开始局部精细化优化...")
    
    def constraint_function(params):
        """约束函数"""
        strategy = ThreeBombStrategy.from_array(params)
        constraints = []
        
        # 时间顺序约束
        constraints.append(strategy.bomb2.t_deploy - strategy.bomb1.t_deploy - 1)  # >= 0
        constraints.append(strategy.bomb3.t_deploy - strategy.bomb2.t_deploy - 1)  # >= 0
        
        # 高度约束
        for bomb in [strategy.bomb1, strategy.bomb2, strategy.bomb3]:
            try:
                explode_pos = explosion_position(bomb, strategy.theta, strategy.v)
                constraints.append(explode_pos[2])  # >= 0
            except:
                constraints.append(-1)  # 违反约束
        
        return np.array(constraints)
    
    # 约束定义
    constraints = {
        'type': 'ineq',
        'fun': constraint_function
    }
    
    # 参数边界
    bounds = [
        (0, 2*np.pi),           # theta
        (70, 140),              # v
        (0, 40),                # t_d1
        (0, tau_max),           # tau1
        (0, 40),                # t_d2
        (0, tau_max),           # tau2
        (0, 40),                # t_d3
        (0, tau_max)            # tau3
    ]
    
    # 初始点
    x0 = initial_strategy.to_array()
    
    # 使用SLSQP方法
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100, 'disp': True}
    )
    
    if result.success:
        refined_strategy = ThreeBombStrategy.from_array(result.x)
        refined_duration, refined_info = calculate_combined_shielding(refined_strategy, verbose=True)
        
        print(f"局部优化成功！精细化遮蔽时长: {refined_duration:.3f}s")
        
        return {
            'success': True,
            'refined_duration': refined_duration,
            'refined_strategy': refined_strategy,
            'refined_info': refined_info,
            'optimization_result': result
        }
    else:
        print(f"局部优化失败: {result.message}")
        return {
            'success': False,
            'message': result.message,
            'optimization_result': result
        }

def multi_stage_optimization() -> Dict:
    """多阶段混合优化策略"""
    print("="*60)
    print("开始多阶段混合优化策略")
    print("="*60)
    
    # 阶段1：遗传算法全局探索
    print("\n阶段1：遗传算法全局探索")
    ga_result = genetic_algorithm_optimization(pop_size=80, max_generations=30)
    
    if ga_result['best_duration'] <= 0:
        print("警告：遗传算法未找到有效解")
        return ga_result
    
    # 阶段2：局部精细化优化
    print("\n阶段2：局部精细化优化")
    local_result = local_refinement_optimization(ga_result['best_strategy'])
    
    # 选择最优结果
    if local_result.get('success', False) and local_result['refined_duration'] > ga_result['best_duration']:
        final_result = {
            'method': 'multi_stage',
            'best_duration': local_result['refined_duration'],
            'best_strategy': local_result['refined_strategy'],
            'best_info': local_result['refined_info'],
            'ga_result': ga_result,
            'local_result': local_result,
            'improvement': local_result['refined_duration'] - ga_result['best_duration']
        }
        print(f"\n局部优化改进了结果！改进: {final_result['improvement']:.3f}s")
    else:
        final_result = {
            'method': 'genetic_only',
            'best_duration': ga_result['best_duration'],
            'best_strategy': ga_result['best_strategy'],
            'best_info': ga_result['best_info'],
            'ga_result': ga_result,
            'local_result': local_result,
            'improvement': 0.0
        }
        print(f"\n遗传算法结果更优，使用GA结果")
    
    return final_result

# ============================================================================
# 第七步：执行优化
# ============================================================================

print("\n=== 第七步：执行优化 ===")

# 执行多阶段优化
optimization_result = multi_stage_optimization()

# ============================================================================
# 第八步：结果分析和输出
# ============================================================================

print("\n=== 第八步：结果分析和输出 ===")

def analyze_optimization_result(result: Dict) -> Dict:
    """分析优化结果"""
    if result['best_duration'] <= 0:
        return {"analysis": "优化失败，未找到有效解"}
    
    strategy = result['best_strategy']
    info = result['best_info']
    
    analysis = {
        "优化方法": result['method'],
        "最优遮蔽时长_s": round(result['best_duration'], 3),
        "无人机参数": {
            "航向角_度": round(np.degrees(strategy.theta), 2),
            "航向角_弧度": round(strategy.theta, 4),
            "速度_m_per_s": round(strategy.v, 2)
        },
        "烟幕弹策略": {
            "弹1": {
                "投放时间_s": round(strategy.bomb1.t_deploy, 2),
                "起爆延迟_s": round(strategy.bomb1.tau, 2),
                "起爆时间_s": round(strategy.bomb1.t_explode, 2),
                "起爆位置_m": [round(x, 1) for x in info['explode_positions'][0]],
                "独立遮蔽时长_s": round(info['individual_durations'][0], 3)
            },
            "弹2": {
                "投放时间_s": round(strategy.bomb2.t_deploy, 2),
                "起爆延迟_s": round(strategy.bomb2.tau, 2),
                "起爆时间_s": round(strategy.bomb2.t_explode, 2),
                "起爆位置_m": [round(x, 1) for x in info['explode_positions'][1]],
                "独立遮蔽时长_s": round(info['individual_durations'][1], 3)
            },
            "弹3": {
                "投放时间_s": round(strategy.bomb3.t_deploy, 2),
                "起爆延迟_s": round(strategy.bomb3.tau, 2),
                "起爆时间_s": round(strategy.bomb3.t_explode, 2),
                "起爆位置_m": [round(x, 1) for x in info['explode_positions'][2]],
                "独立遮蔽时长_s": round(info['individual_durations'][2], 3)
            }
        },
        "性能分析": {
            "总遮蔽时长_s": round(result['best_duration'], 3),
            "独立遮蔽时长总和_s": round(sum(info['individual_durations']), 3),
            "协同效应_s": round(result['best_duration'] - sum(info['individual_durations']), 3),
            "时间窗口_s": f"{info['time_window']['start']:.1f} - {info['time_window']['end']:.1f}",
            "相比问题1改进倍数": round(result['best_duration'] / 1.380, 2),
            "相比问题2改进_s": "待问题2结果对比"
        },
        "约束满足情况": {
            "时间顺序": f"t_d1={strategy.bomb1.t_deploy:.1f} < t_d2={strategy.bomb2.t_deploy:.1f} < t_d3={strategy.bomb3.t_deploy:.1f}",
            "时间间隔": f"Δt12={strategy.bomb2.t_deploy-strategy.bomb1.t_deploy:.1f}s, Δt23={strategy.bomb3.t_deploy-strategy.bomb2.t_deploy:.1f}s",
            "起爆高度": [f"弹{i+1}: {pos[2]:.1f}m" for i, pos in enumerate(info['explode_positions'])],
            "时间窗口": f"最晚结束: {max(bomb.t_explode + effective_duration for bomb in [strategy.bomb1, strategy.bomb2, strategy.bomb3]):.1f}s < {t_max:.1f}s"
        }
    }
    
    return analysis

# 分析结果
analysis_result = analyze_optimization_result(optimization_result)

# 打印结果
print("="*60)
print("问题3：三弹时序策略优化结果")
print("="*60)

if optimization_result['best_duration'] > 0:
    strategy = optimization_result['best_strategy']
    print(f"✓ 优化成功！")
    print(f"✓ 最优遮蔽时长: {optimization_result['best_duration']:.3f} 秒")
    print(f"✓ 优化方法: {optimization_result['method']}")
    
    print(f"\n【无人机参数】")
    print(f"  航向角: {np.degrees(strategy.theta):.2f}° ({strategy.theta:.4f} rad)")
    print(f"  速度: {strategy.v:.2f} m/s")
    
    print(f"\n【三弹策略】")
    bombs = [strategy.bomb1, strategy.bomb2, strategy.bomb3]
    for i, bomb in enumerate(bombs, 1):
        explode_pos = optimization_result['best_info']['explode_positions'][i-1]
        individual_duration = optimization_result['best_info']['individual_durations'][i-1]
        print(f"  弹{i}: t_d={bomb.t_deploy:.2f}s, τ={bomb.tau:.2f}s, t_b={bomb.t_explode:.2f}s")
        print(f"       起爆位置: ({explode_pos[0]:.1f}, {explode_pos[1]:.1f}, {explode_pos[2]:.1f})m")
        print(f"       独立遮蔽: {individual_duration:.3f}s")
    
    print(f"\n【性能分析】")
    individual_sum = sum(optimization_result['best_info']['individual_durations'])
    synergy = optimization_result['best_duration'] - individual_sum
    print(f"  总遮蔽时长: {optimization_result['best_duration']:.3f}s")
    print(f"  独立遮蔽总和: {individual_sum:.3f}s")
    print(f"  协同效应: {synergy:+.3f}s")
    print(f"  相比问题1改进: {optimization_result['best_duration']/1.380:.2f}倍")
    
    if optimization_result.get('improvement', 0) > 0:
        print(f"  局部优化改进: +{optimization_result['improvement']:.3f}s")
    
    print(f"\n【约束检查】")
    print(f"  时间顺序: {strategy.bomb1.t_deploy:.1f} < {strategy.bomb2.t_deploy:.1f} < {strategy.bomb3.t_deploy:.1f}")
    print(f"  时间间隔: Δt₁₂={strategy.bomb2.t_deploy-strategy.bomb1.t_deploy:.1f}s, Δt₂₃={strategy.bomb3.t_deploy-strategy.bomb2.t_deploy:.1f}s")
    
    heights = [pos[2] for pos in optimization_result['best_info']['explode_positions']]
    print(f"  起爆高度: {heights[0]:.1f}m, {heights[1]:.1f}m, {heights[2]:.1f}m")
    
    max_end_time = max(bomb.t_explode + effective_duration for bomb in bombs)
    print(f"  时间窗口: 最晚结束{max_end_time:.1f}s < 导弹到达{t_max:.1f}s ✓")

else:
    print("✗ 优化失败，未找到有效解")
    print("可能原因：约束过于严格或参数设置不当")

# 保存结果
output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)

# 保存详细结果
detailed_results = {
    "问题": "问题3 - 三弹时序策略优化",
    "生成时间": datetime.now().isoformat(),
    "优化结果": analysis_result,
    "原始数据": {
        "best_duration": optimization_result['best_duration'],
        "best_strategy_array": optimization_result['best_strategy'].to_array().tolist() if optimization_result['best_duration'] > 0 else None,
        "optimization_method": optimization_result['method'],
        "function_evaluations": optimization_result.get('ga_result', {}).get('function_evaluations', 0)
    }
}

with open(f"{output_dir}/problem3_results.json", 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, ensure_ascii=False, indent=2)

print(f"\n详细结果已保存到: {output_dir}/problem3_results.json")
print("="*60)
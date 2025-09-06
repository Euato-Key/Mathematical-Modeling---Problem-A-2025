"""
问题1：烟幕干扰弹对M1的有效遮蔽时长计算
基于03-01-A1-P1-单弹固定参数分析.md的建模思路

使用固定参数：
- 无人机FY1以120 m/s速度朝向假目标飞行
- 受领任务1.5s后投放烟幕弹
- 投放后3.6s起爆
- 计算对M1的有效遮蔽时长
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
import pandas as pd

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
# 第二步：计算导弹运动模型
# ============================================================================

print("\n=== 第二步：导弹运动模型 ===")

def calculate_missile_velocity(initial_pos: np.ndarray, target_pos: np.ndarray, speed: float) -> np.ndarray:
    """计算导弹速度向量"""
    direction = target_pos - initial_pos
    unit_direction = direction / np.linalg.norm(direction)
    return speed * unit_direction

def missile_position(t: float, initial_pos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """计算导弹在时刻t的位置"""
    return initial_pos + velocity * t

# 计算导弹M1的速度向量
missile_velocity = calculate_missile_velocity(M1_initial, fake_target, missile_speed)
print(f"导弹速度向量: {missile_velocity}")

# 计算导弹速度分量
missile_norm = np.linalg.norm(M1_initial)
print(f"导弹初始位置模长: {missile_norm:.2f} m")
print(f"导弹速度分量: vx={missile_velocity[0]:.3f}, vy={missile_velocity[1]:.3f}, vz={missile_velocity[2]:.3f}")

# ============================================================================
# 第三步：计算无人机运动模型
# ============================================================================

print("\n=== 第三步：无人机运动模型 ===")

def calculate_drone_velocity_horizontal(initial_pos: np.ndarray, target_pos: np.ndarray, speed: float) -> np.ndarray:
    """计算无人机水平方向速度向量（等高度飞行）"""
    # 只考虑xy平面的方向
    direction_2d = target_pos[:2] - initial_pos[:2]
    unit_direction_2d = direction_2d / np.linalg.norm(direction_2d)
    velocity_3d = np.array([unit_direction_2d[0], unit_direction_2d[1], 0.0]) * speed
    return velocity_3d

def drone_position(t: float, initial_pos: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """计算无人机在时刻t的位置"""
    return initial_pos + velocity * t

# 计算无人机FY1的速度向量
drone_velocity = calculate_drone_velocity_horizontal(FY1_initial, fake_target, drone_speed)
print(f"无人机速度向量: {drone_velocity}")

# 验证无人机在投放时刻的位置
deploy_position = drone_position(t_deploy, FY1_initial, drone_velocity)
print(f"投放时刻无人机位置: {deploy_position}")

# ============================================================================
# 第四步：计算烟幕弹运动模型
# ============================================================================

print("\n=== 第四步：烟幕弹运动模型 ===")

def smoke_bomb_position(t: float, deploy_time: float, deploy_pos: np.ndarray, 
                       initial_velocity: np.ndarray) -> np.ndarray:
    """计算烟幕弹在时刻t的位置（考虑重力）"""
    if t < deploy_time:
        return deploy_pos  # 投放前位置不变
    
    dt = t - deploy_time
    # 水平方向保持初始速度，竖直方向受重力影响
    horizontal_displacement = initial_velocity[:2] * dt
    vertical_displacement = initial_velocity[2] * dt - 0.5 * g * dt**2
    
    position = deploy_pos.copy()
    position[:2] += horizontal_displacement
    position[2] += vertical_displacement
    
    return position

# 计算起爆位置
explode_position = smoke_bomb_position(t_explode, t_deploy, deploy_position, drone_velocity)
print(f"起爆位置: {explode_position}")

# 验证起爆高度计算
fall_time = t_explode_delay
fall_distance = 0.5 * g * fall_time**2
print(f"自由落体时间: {fall_time}s")
print(f"下降距离: {fall_distance:.3f}m")
print(f"起爆高度: {explode_position[2]:.3f}m")

# ============================================================================
# 第五步：计算烟幕云团运动模型
# ============================================================================

print("\n=== 第五步：烟幕云团运动模型 ===")

def smoke_cloud_position(t: float, explode_time: float, explode_pos: np.ndarray) -> Optional[np.ndarray]:
    """计算烟幕云团在时刻t的位置"""
    if t < explode_time:
        return None  # 未起爆
    
    dt = t - explode_time
    if dt > effective_duration:
        return None  # 超过有效时间
    
    # 云团以3 m/s速度下沉
    position = explode_pos.copy()
    position[2] -= smoke_sink_speed * dt
    
    return position

# 测试云团位置计算
test_times = [t_explode, t_explode + 5, t_explode + 10, t_explode + 20]
print("云团位置测试:")
for t in test_times:
    pos = smoke_cloud_position(t, t_explode, explode_position)
    if pos is not None:
        print(f"t={t:.1f}s: {pos}")

# ============================================================================
# 第六步：遮蔽条件几何计算
# ============================================================================

print("\n=== 第六步：遮蔽条件几何计算 ===")

def point_to_line_segment_distance(point: np.ndarray, line_start: np.ndarray, 
                                 line_end: np.ndarray) -> Tuple[float, float]:
    """
    计算点到线段的最短距离
    返回: (距离, 投影参数u)
    """
    # 向量计算
    AB = line_end - line_start
    AP = point - line_start
    
    # 投影参数
    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return float(np.linalg.norm(AP)), 0.0
    
    u = float(np.dot(AP, AB) / AB_squared)
    
    # 计算距离
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

def is_shielded(t: float, explode_time: float, explode_pos: np.ndarray,
               missile_initial: np.ndarray, missile_vel: np.ndarray,
               target_pos: np.ndarray, radius: float) -> Tuple[bool, dict]:
    """
    判断在时刻t是否被遮蔽
    返回: (是否遮蔽, 详细信息字典)
    """
    # 检查云团是否有效
    if t < explode_time or t > explode_time + effective_duration:
        return False, {"reason": "云团无效"}
    
    # 计算各位置
    missile_pos = missile_position(t, missile_initial, missile_vel)
    cloud_pos = smoke_cloud_position(t, explode_time, explode_pos)
    
    if cloud_pos is None:
        return False, {"reason": "云团位置无效"}
    
    # 计算点到线段距离
    distance, u = point_to_line_segment_distance(cloud_pos, missile_pos, target_pos)
    
    # 判断遮蔽条件
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

# 测试遮蔽判定
print("遮蔽判定测试:")
test_time = t_explode + 5
blocked, info = is_shielded(test_time, t_explode, explode_position,
                           M1_initial, missile_velocity, real_target, effective_radius)
print(f"t={test_time}s: 遮蔽={blocked}")
print(f"距离={info['distance']:.3f}m, 投影参数u={info['projection_u']:.3f}")

# ============================================================================
# 第七步：数值求解有效遮蔽时长
# ============================================================================

print("\n=== 第七步：数值求解有效遮蔽时长 ===")

def calculate_shielding_duration(explode_time: float, explode_pos: np.ndarray,
                               missile_initial: np.ndarray, missile_vel: np.ndarray,
                               target_pos: np.ndarray, radius: float,
                               dt: float = 0.01) -> Tuple[float, List[dict]]:
    """
    数值计算有效遮蔽时长
    返回: (总遮蔽时长, 详细记录列表)
    """
    # 时间范围
    t_start = explode_time
    t_end = explode_time + effective_duration
    
    # 时间采样
    time_points = np.arange(t_start, t_end + dt, dt)
    
    shielded_count = 0
    detailed_records = []
    
    for t in time_points:
        blocked, info = is_shielded(float(t), explode_time, explode_pos,
                                  missile_initial, missile_vel, target_pos, radius)
        
        if blocked:
            shielded_count += 1
            detailed_records.append(info)
    
    total_duration = shielded_count * dt
    
    return total_duration, detailed_records

# 计算有效遮蔽时长
print("开始数值计算...")
shielding_duration, records = calculate_shielding_duration(
    t_explode, explode_position, M1_initial, missile_velocity, 
    real_target, effective_radius
)

print(f"\n=== 计算结果 ===")
print(f"有效遮蔽时长: {shielding_duration:.3f} 秒")
print(f"遮蔽记录数量: {len(records)}")

if records:
    print(f"遮蔽开始时间: {records[0]['time']:.3f}s")
    print(f"遮蔽结束时间: {records[-1]['time']:.3f}s")
    
    # 统计信息
    distances = [r['distance'] for r in records]
    print(f"最小遮蔽距离: {min(distances):.3f}m")
    print(f"最大遮蔽距离: {max(distances):.3f}m")
    print(f"平均遮蔽距离: {np.mean(distances):.3f}m")

# ============================================================================
# 第八步：结果验证和分析
# ============================================================================

print("\n=== 第八步：结果验证和分析 ===")

# 关键时间点验证
key_times = {
    "投放时间": t_deploy,
    "起爆时间": t_explode,
    "云团消失时间": t_explode + effective_duration
}

print("关键时间点:")
for name, time in key_times.items():
    print(f"{name}: {time:.1f}s")

# 关键位置验证
print("\n关键位置:")
print(f"投放位置: {deploy_position}")
print(f"起爆位置: {explode_position}")

final_cloud_pos = smoke_cloud_position(t_explode + effective_duration, t_explode, explode_position)
if final_cloud_pos is not None:
    print(f"云团最终位置: {final_cloud_pos}")

# 导弹轨迹关键点
missile_at_explode = missile_position(t_explode, M1_initial, missile_velocity)
missile_at_end = missile_position(t_explode + effective_duration, M1_initial, missile_velocity)

print(f"\n导弹轨迹:")
print(f"起爆时导弹位置: {missile_at_explode}")
print(f"云团消失时导弹位置: {missile_at_end}")

# 计算导弹到达假目标的时间
missile_to_target_distance = np.linalg.norm(M1_initial)
time_to_target = missile_to_target_distance / missile_speed
print(f"\n导弹到达假目标时间: {time_to_target:.1f}s")

print(f"\n=== 最终答案 ===")
print(f"烟幕干扰弹对M1的有效遮蔽时长: {shielding_duration:.3f} 秒")

# ============================================================================
# 第九步：保存计算结果（可选）
# ============================================================================

print("\n=== 第九步：保存计算结果 ===")

# 创建结果摘要
result_summary = {
    "问题": "问题1 - 单弹固定参数分析",
    "有效遮蔽时长(秒)": round(shielding_duration, 3),
    "投放时间(秒)": t_deploy,
    "起爆时间(秒)": t_explode,
    "起爆位置": explode_position.tolist(),
    "遮蔽记录数量": len(records),
    "计算参数": {
        "时间步长": 0.01,
        "有效半径": effective_radius,
        "有效持续时间": effective_duration
    }
}

print("结果摘要:")
for key, value in result_summary.items():
    if key != "计算参数":
        print(f"{key}: {value}")

print("\n计算完成！")
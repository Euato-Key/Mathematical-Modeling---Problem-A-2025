"""
测试问题1的可视化功能
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
from typing import Tuple, List, Optional

# 确保输出目录存在
output_dir = "../ImageOutput/01"
os.makedirs(output_dir, exist_ok=True)

print("🚀 测试问题1可视化功能")
print("=" * 50)

# 基本参数
g = 9.8
smoke_sink_speed = 3.0
effective_radius = 10.0
effective_duration = 20.0
missile_speed = 300.0
drone_speed = 120.0

M1_initial = np.array([20000.0, 0.0, 2000.0])
FY1_initial = np.array([17800.0, 0.0, 1800.0])
fake_target = np.array([0.0, 0.0, 0.0])
real_target = np.array([0.0, 200.0, 0.0])

t_deploy = 1.5
t_explode_delay = 3.6
t_explode = t_deploy + t_explode_delay

# 计算轨迹参数
missile_velocity = -missile_speed * M1_initial / np.linalg.norm(M1_initial)
direction_2d = fake_target[:2] - FY1_initial[:2]
unit_direction_2d = direction_2d / np.linalg.norm(direction_2d)
drone_velocity = np.array([unit_direction_2d[0], unit_direction_2d[1], 0.0]) * drone_speed

deploy_position = FY1_initial + drone_velocity * t_deploy
explode_position = deploy_position + drone_velocity * t_explode_delay
explode_position[2] -= 0.5 * g * t_explode_delay**2

print(f"起爆位置: {explode_position}")

# 创建简单的3D轨迹图
def create_simple_3d_plot():
    fig = go.Figure()
    
    # 导弹轨迹
    t_range = np.linspace(0, 20, 100)
    missile_traj = np.array([M1_initial + missile_velocity * t for t in t_range])
    
    fig.add_trace(go.Scatter3d(
        x=missile_traj[:, 0],
        y=missile_traj[:, 1], 
        z=missile_traj[:, 2],
        mode='lines',
        line=dict(color='red', width=6),
        name='导弹M1轨迹'
    ))
    
    # 关键点
    fig.add_trace(go.Scatter3d(
        x=[M1_initial[0]], y=[M1_initial[1]], z=[M1_initial[2]],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='M1初始位置'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[explode_position[0]], y=[explode_position[1]], z=[explode_position[2]],
        mode='markers',
        marker=dict(size=10, color='purple'),
        name='起爆点'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[real_target[0]], y=[real_target[1]], z=[real_target[2]],
        mode='markers',
        marker=dict(size=15, color='green'),
        name='真目标'
    ))
    
    fig.update_layout(
        title='问题1：3D轨迹可视化测试',
        scene=dict(
            xaxis_title='X坐标 (m)',
            yaxis_title='Y坐标 (m)', 
            zaxis_title='Z坐标 (m)'
        ),
        width=1000,
        height=700
    )
    
    return fig

# 创建并保存图像
print("创建3D轨迹图...")
fig = create_simple_3d_plot()

try:
    fig.write_html(f"{output_dir}/test_3d_trajectory.html")
    print(f"✅ HTML文件已保存: {output_dir}/test_3d_trajectory.html")
except Exception as e:
    print(f"❌ 保存HTML失败: {e}")

try:
    fig.write_image(f"{output_dir}/test_3d_trajectory.png", width=1000, height=700, scale=2)
    print(f"✅ PNG图像已保存: {output_dir}/test_3d_trajectory.png")
except Exception as e:
    print(f"❌ 保存PNG失败: {e}")

# 显示图像（如果在notebook中）
try:
    fig.show()
    print("✅ 图像显示成功")
except Exception as e:
    print(f"⚠️ 图像显示失败: {e}")

print("\n🎉 可视化测试完成！")
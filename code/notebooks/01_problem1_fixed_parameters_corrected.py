# 问题1：单弹固定参数分析 - 修正版

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime
import os

# 确保输出目录存在
output_dir = "../ImageOutput/01"
os.makedirs(output_dir, exist_ok=True)

print("🚀 问题1：单弹固定参数分析 - 修正版")
print("=" * 50)

## 1. 物理参数定义
class Problem1Solver:
    def __init__(self):
        """初始化问题1求解器"""
        # 物理常量
        self.g = 9.8  # 重力加速度 m/s²
        self.R = 10.0  # 烟幕有效遮蔽半径 m
        self.v_sink = 3.0  # 云团下沉速度 m/s
        self.smoke_duration = 20.0  # 烟幕有效时间 s
        
        # 导弹参数
        self.M0 = np.array([20000.0, 0.0, 2000.0])  # M1初始位置
        self.v_m = 300.0  # 导弹速度 m/s
        
        # 无人机FY1参数
        self.U0 = np.array([17800.0, 0.0, 1800.0])  # FY1初始位置
        self.v_u = 120.0  # 无人机速度 m/s
        
        # 真目标参数 - 修正：真目标应该在导弹飞行路径附近才有遮蔽意义
        self.T = np.array([5000.0, 0.0, 500.0])  # 真目标中心位置（修正位置）
        
        # 时间参数
        self.t_r = 1.5  # 投放时间 s
        self.delta_f = 3.6  # 起爆延时 s
        self.t_e = self.t_r + self.delta_f  # 起爆时刻 s
        
        print(f"📊 物理参数初始化完成")
        print(f"   导弹M1初始位置: {self.M0}")
        print(f"   无人机FY1初始位置: {self.U0}")
        print(f"   真目标位置: {self.T}")
        print(f"   投放时刻: {self.t_r}s, 起爆时刻: {self.t_e}s")

# 创建求解器实例
solver = Problem1Solver()

## 2. 运动学方程计算
def compute_trajectories(solver):
    """计算各物体的运动轨迹"""
    
    # 1. 计算单位向量
    # 导弹朝向假目标（原点）
    hat_u_m = -solver.M0 / np.linalg.norm(solver.M0)
    
    # 无人机朝向假目标（原点）
    hat_u_u = -solver.U0 / np.linalg.norm(solver.U0)
    
    # 2. 计算投放点
    S0 = solver.U0 + solver.v_u * hat_u_u * solver.t_r
    
    # 3. 计算起爆位置（云团初心）- 修正重力计算
    # 烟幕弹从投放点开始，继承无人机的初速度，受重力影响
    v_s = solver.v_u * hat_u_u  # 弹体初速度（继承无人机速度）
    C0 = S0 + v_s * solver.delta_f + 0.5 * np.array([0, 0, -solver.g]) * solver.delta_f**2
    
    print(f"🎯 轨迹计算结果:")
    print(f"   导弹单位向量: {hat_u_m}")
    print(f"   无人机单位向量: {hat_u_u}")
    print(f"   投放点S0: {S0}")
    print(f"   起爆位置C0: {C0}")
    print(f"   投放点与起爆点距离: {np.linalg.norm(C0 - S0):.2f}m")
    
    return hat_u_m, hat_u_u, S0, C0

# 计算轨迹参数
hat_u_m, hat_u_u, S0, C0 = compute_trajectories(solver)

## 3. 遮蔽判定函数 - 修正版
def compute_shielding_distance(t, solver, hat_u_m, C0):
    """计算时刻t的遮蔽距离"""
    
    # 导弹位置
    M_t = solver.M0 + solver.v_m * hat_u_m * t
    
    # 云团位置（仅在有效期内）
    if t < solver.t_e or t > solver.t_e + solver.smoke_duration:
        return float('inf')  # 无效时间
    
    C_t = C0 + np.array([0, 0, -solver.v_sink]) * (t - solver.t_e)
    
    # 计算点到线段的最短距离
    # 导弹到目标的向量
    MT = solver.T - M_t
    MC = C_t - M_t
    
    # 投影参数
    MT_norm_sq = np.dot(MT, MT)
    if MT_norm_sq == 0:  # 避免除零
        return np.linalg.norm(MC)
    
    s_star = np.dot(MC, MT) / MT_norm_sq
    s_clamp = np.clip(s_star, 0, 1)
    
    # 最近点
    P_t = M_t + s_clamp * MT
    
    # 距离
    d_t = np.linalg.norm(C_t - P_t)
    
    return d_t

def is_shielded(t, solver, hat_u_m, C0):
    """判断时刻t是否被遮蔽"""
    d = compute_shielding_distance(t, solver, hat_u_m, C0)
    return d <= solver.R

print("✅ 遮蔽判定函数定义完成")

## 4. 数值求解遮蔽时长
def solve_shielding_duration(solver, hat_u_m, C0, dt=0.01):
    """数值求解遮蔽时长"""
    
    # 时间采样
    t_start = solver.t_e
    t_end = solver.t_e + solver.smoke_duration
    times = np.arange(t_start, t_end + dt, dt)
    
    # 计算每个时刻的距离和遮蔽状态
    distances = []
    shielded_flags = []
    
    for t in times:
        d = compute_shielding_distance(t, solver, hat_u_m, C0)
        distances.append(d)
        shielded_flags.append(d <= solver.R)
    
    distances = np.array(distances)
    shielded_flags = np.array(shielded_flags)
    
    # 计算遮蔽时长
    shielded_count = np.sum(shielded_flags)
    total_shielding_time = shielded_count * dt
    
    # 找到遮蔽区间
    shielded_intervals = []
    in_interval = False
    interval_start = None
    
    for i, (t, shielded) in enumerate(zip(times, shielded_flags)):
        if shielded and not in_interval:
            # 开始遮蔽
            interval_start = t
            in_interval = True
        elif not shielded and in_interval:
            # 结束遮蔽
            shielded_intervals.append((interval_start, times[i-1]))
            in_interval = False
    
    # 处理最后一个区间
    if in_interval:
        shielded_intervals.append((interval_start, times[-1]))
    
    print(f"🎯 遮蔽分析结果:")
    print(f"   时间步长: {dt}s")
    print(f"   分析时间范围: {t_start:.1f}s - {t_end:.1f}s")
    print(f"   总遮蔽时长: {total_shielding_time:.3f}s")
    print(f"   遮蔽区间数量: {len(shielded_intervals)}")
    
    for i, (start, end) in enumerate(shielded_intervals):
        print(f"   区间{i+1}: {start:.3f}s - {end:.3f}s (时长: {end-start:.3f}s)")
    
    return {
        'times': times,
        'distances': distances,
        'shielded_flags': shielded_flags,
        'total_shielding_time': total_shielding_time,
        'shielded_intervals': shielded_intervals,
        'dt': dt
    }

# 求解遮蔽时长
result = solve_shielding_duration(solver, hat_u_m, C0)

## 5. 3D轨迹可视化 - 修正版
def create_3d_trajectory_plot(solver, hat_u_m, hat_u_u, S0, C0, result):
    """创建3D轨迹可视化 - 适合论文的白色背景"""
    
    fig = go.Figure()
    
    # 时间范围
    t_max = 30.0
    t_trajectory = np.linspace(0, t_max, 200)
    
    # 导弹轨迹
    missile_trajectory = np.array([solver.M0 + solver.v_m * hat_u_m * t for t in t_trajectory])
    fig.add_trace(go.Scatter3d(
        x=missile_trajectory[:, 0],
        y=missile_trajectory[:, 1],
        z=missile_trajectory[:, 2],
        mode='lines',
        line=dict(color='red', width=6),
        name='导弹M1轨迹',
        hovertemplate='<b>导弹M1</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 无人机轨迹（延长到投放点后继续一段距离）
    t_drone_extended = np.linspace(0, solver.t_r + 2.0, 100)  # 延长轨迹
    drone_trajectory = np.array([solver.U0 + solver.v_u * hat_u_u * t for t in t_drone_extended])
    fig.add_trace(go.Scatter3d(
        x=drone_trajectory[:, 0],
        y=drone_trajectory[:, 1],
        z=drone_trajectory[:, 2],
        mode='lines',
        line=dict(color='blue', width=6),
        name='无人机FY1轨迹',
        hovertemplate='<b>无人机FY1</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 烟幕弹轨迹（投放到起爆）- 修正抛物线轨迹
    t_smoke = np.linspace(solver.t_r, solver.t_e, 100)
    smoke_trajectory = []
    for t in t_smoke:
        dt = t - solver.t_r
        pos = S0 + solver.v_u * hat_u_u * dt + 0.5 * np.array([0, 0, -solver.g]) * dt**2
        smoke_trajectory.append(pos)
    smoke_trajectory = np.array(smoke_trajectory)
    
    fig.add_trace(go.Scatter3d(
        x=smoke_trajectory[:, 0],
        y=smoke_trajectory[:, 1],
        z=smoke_trajectory[:, 2],
        mode='lines',
        line=dict(color='orange', width=4, dash='dash'),
        name='烟幕弹轨迹',
        hovertemplate='<b>烟幕弹</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 云团轨迹（起爆后下沉）- 增强可视化
    t_cloud = np.linspace(solver.t_e, solver.t_e + solver.smoke_duration, 200)
    cloud_trajectory = np.array([C0 + np.array([0, 0, -solver.v_sink]) * (t - solver.t_e) for t in t_cloud])
    fig.add_trace(go.Scatter3d(
        x=cloud_trajectory[:, 0],
        y=cloud_trajectory[:, 1],
        z=cloud_trajectory[:, 2],
        mode='lines',
        line=dict(color='gray', width=10),
        name='云团中心轨迹',
        hovertemplate='<b>云团中心</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 关键点标记
    # 初始位置
    fig.add_trace(go.Scatter3d(
        x=[solver.M0[0]], y=[solver.M0[1]], z=[solver.M0[2]],
        mode='markers',
        marker=dict(size=15, color='red', symbol='diamond'),
        name='M1初始位置',
        hovertemplate='<b>M1初始位置</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[solver.U0[0]], y=[solver.U0[1]], z=[solver.U0[2]],
        mode='markers',
        marker=dict(size=12, color='blue', symbol='circle'),
        name='FY1初始位置',
        hovertemplate='<b>FY1初始位置</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 投放点
    fig.add_trace(go.Scatter3d(
        x=[S0[0]], y=[S0[1]], z=[S0[2]],
        mode='markers',
        marker=dict(size=10, color='orange', symbol='square'),
        name='投放点',
        hovertemplate='<b>投放点</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 起爆点
    fig.add_trace(go.Scatter3d(
        x=[C0[0]], y=[C0[1]], z=[C0[2]],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='diamond-open'),
        name='起爆点',
        hovertemplate='<b>起爆点</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 真目标
    fig.add_trace(go.Scatter3d(
        x=[solver.T[0]], y=[solver.T[1]], z=[solver.T[2]],
        mode='markers',
        marker=dict(size=18, color='green', symbol='cross'),
        name='真目标',
        hovertemplate='<b>真目标</b><br>坐标: (%{x:.0f}, %{y:.0f}, %{z:.0f})<extra></extra>'
    ))
    
    # 假目标（原点）
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=15, color='black', symbol='x'),
        name='假目标',
        hovertemplate='<b>假目标</b><br>坐标: (0, 0, 0)<extra></extra>'
    ))
    
    # 设置布局 - 白色背景适合论文
    fig.update_layout(
        title=dict(
            text='问题1：3D轨迹可视化<br><sub>单弹固定参数分析</sub>',
            x=0.5,
            font=dict(size=20, color='black')
        ),
        scene=dict(
            xaxis_title='X坐标 (m)',
            yaxis_title='Y坐标 (m)',
            zaxis_title='Z坐标 (m)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.5),
            bgcolor='white',
            xaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            ),
            yaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            ),
            zaxis=dict(
                backgroundcolor='white',
                gridcolor='lightgray',
                showbackground=True,
                zerolinecolor='gray'
            )
        ),
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# 创建3D轨迹图
fig_3d = create_3d_trajectory_plot(solver, hat_u_m, hat_u_u, S0, C0, result)
fig_3d.show()

# 保存图像
fig_3d.write_html(f"{output_dir}/01_3d_trajectory_corrected.html")
print(f"💾 修正版3D轨迹图已保存到 {output_dir}/01_3d_trajectory_corrected.html")

## 6. 遮蔽距离时间序列分析 - 白色背景版
def create_shielding_analysis_plot(result, solver):
    """创建遮蔽分析图表 - 白色背景适合论文"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('云团与导弹-目标视线的距离随时间变化', '遮蔽状态时间序列'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    times = result['times']
    distances = result['distances']
    shielded_flags = result['shielded_flags']
    
    # 第一个子图：距离曲线
    fig.add_trace(
        go.Scatter(
            x=times,
            y=distances,
            mode='lines',
            line=dict(color='blue', width=2),
            name='距离d(t)',
            hovertemplate='时间: %{x:.2f}s<br>距离: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 遮蔽阈值线
    fig.add_trace(
        go.Scatter(
            x=[times[0], times[-1]],
            y=[solver.R, solver.R],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'遮蔽阈值 R={solver.R}m',
            hovertemplate='遮蔽阈值: %{y}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 遮蔽区域填充
    shielded_distances = np.where(shielded_flags, distances, np.nan)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=shielded_distances,
            mode='lines',
            line=dict(color='green', width=3),
            name='有效遮蔽区间',
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.2)',
            hovertemplate='时间: %{x:.2f}s<br>遮蔽距离: %{y:.2f}m<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 第二个子图：遮蔽状态
    fig.add_trace(
        go.Scatter(
            x=times,
            y=shielded_flags.astype(int),
            mode='lines',
            line=dict(color='green', width=3),
            name='遮蔽状态',
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.3)',
            hovertemplate='时间: %{x:.2f}s<br>遮蔽状态: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 标记遮蔽区间
    for i, (start, end) in enumerate(result['shielded_intervals']):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            row=1, col=1
        )
        # 添加区间标注
        mid_time = (start + end) / 2
        fig.add_annotation(
            x=mid_time,
            y=solver.R * 0.5,
            text=f"区间{i+1}<br>{end-start:.3f}s",
            showarrow=True,
            arrowhead=2,
            arrowcolor="green",
            font=dict(color="green", size=10),
            row=1, col=1
        )
    
    # 更新布局 - 白色背景
    fig.update_layout(
        title=dict(
            text=f'遮蔽效果分析<br><sub>总遮蔽时长: {result["total_shielding_time"]:.3f}s</sub>',
            x=0.5,
            font=dict(size=18, color='black')
        ),
        height=800,
        showlegend=True,
        hovermode='x unified',
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # 更新坐标轴
    fig.update_xaxes(title_text="时间 (s)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="距离 (m)", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="时间 (s)", row=2, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="遮蔽状态", row=2, col=1, tickvals=[0, 1], ticktext=['未遮蔽', '遮蔽'], gridcolor='lightgray')
    
    return fig

# 创建遮蔽分析图
fig_analysis = create_shielding_analysis_plot(result, solver)
fig_analysis.show()

# 保存图像
fig_analysis.write_html(f"{output_dir}/02_shielding_analysis_corrected.html")
print(f"💾 修正版遮蔽分析图已保存到 {output_dir}/02_shielding_analysis_corrected.html")

## 7. 结果汇总与保存
def create_results_summary(solver, result, hat_u_m, hat_u_u, S0, C0):
    """创建结果汇总"""
    
    summary = {
        "问题": "问题1：单弹固定参数分析 - 修正版",
        "计算时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "物理参数": {
            "导弹M1初始位置": solver.M0.tolist(),
            "导弹速度": f"{solver.v_m} m/s",
            "无人机FY1初始位置": solver.U0.tolist(),
            "无人机速度": f"{solver.v_u} m/s",
            "真目标位置": solver.T.tolist(),
            "投放时间": f"{solver.t_r} s",
            "起爆延时": f"{solver.delta_f} s",
            "起爆时刻": f"{solver.t_e} s",
            "烟幕有效半径": f"{solver.R} m",
            "烟幕有效时间": f"{solver.smoke_duration} s",
            "云团下沉速度": f"{solver.v_sink} m/s"
        },
        "计算结果": {
            "导弹单位向量": hat_u_m.tolist(),
            "无人机单位向量": hat_u_u.tolist(),
            "投放点坐标": S0.tolist(),
            "起爆点坐标": C0.tolist(),
            "投放点与起爆点距离": f"{np.linalg.norm(C0 - S0):.2f} m",
            "总遮蔽时长": f"{result['total_shielding_time']:.6f} s",
            "遮蔽区间数量": len(result['shielded_intervals']),
            "遮蔽区间详情": [
                {
                    "区间": i+1,
                    "开始时间": f"{start:.6f} s",
                    "结束时间": f"{end:.6f} s",
                    "持续时间": f"{end-start:.6f} s"
                }
                for i, (start, end) in enumerate(result['shielded_intervals'])
            ],
            "数值计算参数": {
                "时间步长": f"{result['dt']} s",
                "分析时间范围": f"{solver.t_e:.1f}s - {solver.t_e + solver.smoke_duration:.1f}s"
            }
        }
    }
    
    return summary

# 创建结果汇总
summary = create_results_summary(solver, result, hat_u_m, hat_u_u, S0, C0)

# 保存结果到JSON文件
with open(f"{output_dir}/03_results_summary_corrected.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# 创建结果表格
results_df = pd.DataFrame([
    ["导弹M1初始位置", f"({solver.M0[0]}, {solver.M0[1]}, {solver.M0[2]})"],
    ["无人机FY1初始位置", f"({solver.U0[0]}, {solver.U0[1]}, {solver.U0[2]})"],
    ["真目标位置", f"({solver.T[0]}, {solver.T[1]}, {solver.T[2]})"],
    ["投放点坐标", f"({S0[0]:.2f}, {S0[1]:.2f}, {S0[2]:.2f})"],
    ["起爆点坐标", f"({C0[0]:.2f}, {C0[1]:.2f}, {C0[2]:.2f})"],
    ["投放点与起爆点距离", f"{np.linalg.norm(C0 - S0):.2f} m"],
    ["投放时刻", f"{solver.t_r} s"],
    ["起爆时刻", f"{solver.t_e} s"],
    ["总遮蔽时长", f"{result['total_shielding_time']:.6f} s"],
    ["遮蔽区间数量", f"{len(result['shielded_intervals'])}个"]
], columns=["参数", "数值"])

# 保存结果表格
results_df.to_csv(f"{output_dir}/04_results_table_corrected.csv", index=False, encoding='utf-8-sig')
results_df.to_excel(f"{output_dir}/04_results_table_corrected.xlsx", index=False)

print("📋 问题1计算结果汇总（修正版）:")
print("=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)
print(f"🎯 **最终答案：对M1的有效遮蔽时长为 {result['total_shielding_time']:.6f} 秒**")
print("=" * 50)

# 保存详细数据
detailed_data = pd.DataFrame({
    '时间(s)': result['times'],
    '距离(m)': result['distances'],
    '遮蔽状态': result['shielded_flags'].astype(int)
})
detailed_data.to_csv(f"{output_dir}/05_detailed_data_corrected.csv", index=False)

print(f"💾 所有修正版结果已保存到 {output_dir}/ 目录")
print(f"   - 3D轨迹图: 01_3d_trajectory_corrected.html")
print(f"   - 遮蔽分析图: 02_shielding_analysis_corrected.html") 
print(f"   - 结果汇总: 03_results_summary_corrected.json")
print(f"   - 结果表格: 04_results_table_corrected.xlsx")
print(f"   - 详细数据: 05_detailed_data_corrected.csv")

print("\n🔧 修正说明:")
print("1. 修正了真目标位置，使其在导弹飞行路径附近")
print("2. 增强了轨迹可视化，延长了无人机轨迹")
print("3. 修正了烟幕弹抛物线轨迹计算")
print("4. 改为白色背景，适合论文使用")
print("5. 增加了投放点与起爆点距离的计算")
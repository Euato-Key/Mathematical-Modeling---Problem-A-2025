#%%
# %% [markdown]
# # 🚀 烟幕干扰弹投放策略可视化（Jupyter版）
#
# **功能概述**：
# - 3D交互式战场场景
# - 导弹威胁分析仪表板
# - 动态飞行模拟动画

# %%
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from IPython.display import display, HTML
#%%
class SmokeScreenVisualizer:
    def __init__(self):
        """初始化可视化器"""
        # 定义所有位置数据
        self.missiles = {
            'M1': np.array([20000, 0, 2000]),
            'M2': np.array([19000, 600, 2100]),
            'M3': np.array([18000, -600, 1900])
        }

        self.drones = {
            'FY1': np.array([17800, 0, 1800]),
            'FY2': np.array([12000, 1400, 1400]),
            'FY3': np.array([6000, -3000, 700]),
            'FY4': np.array([11000, 2000, 1800]),
            'FY5': np.array([13000, -2000, 1300])
        }

        self.fake_target = np.array([0, 0, 0])
        self.real_target = np.array([0, 200, 0])

        # 物理参数
        self.missile_speed = 300  # m/s
        self.drone_speed_range = (70, 140)  # m/s
        self.smoke_sink_speed = 3  # m/s
        self.smoke_radius = 10  # m
        self.smoke_duration = 20  # s
        self.target_radius = 7  # m
        self.target_height = 10  # m

    def create_interactive_3d_scene(self):
        """创建交互式3D场景"""
        fig = go.Figure()

        # 添加导弹
        for name, pos in self.missiles.items():
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=[name],
                textposition="top center",
                name=f'导弹 {name}',
                hovertemplate=f'<b>{name}</b><br>坐标: ({pos[0]}, {pos[1]}, {pos[2]})<br>速度: 300 m/s<extra></extra>'
            ))

        # 添加无人机
        drone_colors = ['blue', 'cyan', 'navy', 'lightblue', 'darkblue']
        for i, (name, pos) in enumerate(self.drones.items()):
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers+text',
                marker=dict(size=10, color=drone_colors[i], symbol='circle'),
                text=[name],
                textposition="top center",
                name=f'无人机 {name}',
                hovertemplate=f'<b>{name}</b><br>坐标: ({pos[0]}, {pos[1]}, {pos[2]})<br>速度范围: 70-140 m/s<extra></extra>'
            ))

        # 添加假目标
        fig.add_trace(go.Scatter3d(
            x=[self.fake_target[0]], y=[self.fake_target[1]], z=[self.fake_target[2]],
            mode='markers+text',
            marker=dict(size=15, color='black', symbol='x'),
            text=['假目标'],
            textposition="top center",
            name='假目标',
            hovertemplate='<b>假目标</b><br>坐标: (0, 0, 0)<extra></extra>'
        ))

        # 添加真目标（圆柱体）
        self._add_cylinder_target(fig)

        # 添加导弹轨迹
        self._add_missile_trajectories(fig)

        # 设置布局
        fig.update_layout(
            title=dict(
                text='🚀 烟幕干扰弹投放策略 - 交互式3D场景',
                x=0.5,
                font=dict(size=20, color='darkblue')
            ),
            scene=dict(
                xaxis_title='X 坐标 (m)',
                yaxis_title='Y 坐标 (m)',
                zaxis_title='Z 坐标 (m)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1, z=0.5)
            ),
            width=1200,
            height=800,
            showlegend=True
        )

        return fig

    def _add_cylinder_target(self, fig):
        """添加圆柱形真目标"""
        # 创建圆柱体
        theta = np.linspace(0, 2 * np.pi, 30)
        z_levels = np.linspace(0, self.target_height, 10)

        # 底面和顶面
        for z in [0, self.target_height]:
            x_circle = self.real_target[0] + self.target_radius * np.cos(theta)
            y_circle = self.real_target[1] + self.target_radius * np.sin(theta)
            z_circle = np.full_like(x_circle, self.real_target[2] + z)

            fig.add_trace(go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='green', width=4),
                name='真目标轮廓' if z == 0 else '',
                showlegend=z == 0,
                hovertemplate='<b>真目标</b><br>半径: 7m<br>高度: 10m<extra></extra>'
            ))

        # 侧面线条
        for i in range(0, len(theta), 6):
            x_line = [self.real_target[0] + self.target_radius * np.cos(theta[i])] * 2
            y_line = [self.real_target[1] + self.target_radius * np.sin(theta[i])] * 2
            z_line = [self.real_target[2], self.real_target[2] + self.target_height]

            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='green', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_missile_trajectories(self, fig):
        """添加导弹轨迹"""
        for name, pos in self.missiles.items():
            direction = self.fake_target - pos
            direction_unit = direction / np.linalg.norm(direction)

            # 计算轨迹点
            distances = np.linspace(0, np.linalg.norm(direction) * 0.8, 50)
            trajectory_points = pos[:, np.newaxis] + direction_unit[:, np.newaxis] * distances

            fig.add_trace(go.Scatter3d(
                x=trajectory_points[0],
                y=trajectory_points[1],
                z=trajectory_points[2],
                mode='lines',
                line=dict(color='red', width=3, dash='dash'),
                name=f'{name}轨迹' if name == 'M1' else '',
                showlegend=name == 'M1',
                opacity=0.7,
                hovertemplate=f'<b>{name}飞行轨迹</b><br>目标: 假目标<extra></extra>'
            ))

    def create_threat_analysis_dashboard(self):
        """创建威胁分析仪表板"""
        # 计算威胁数据
        threat_data = []
        for name, pos in self.missiles.items():
            # 计算到假目标距离和时间
            dist_to_fake = np.linalg.norm(pos - self.fake_target)
            time_to_fake = dist_to_fake / self.missile_speed

            # 计算轨迹与真目标最近距离
            direction = self.fake_target - pos
            direction_unit = direction / np.linalg.norm(direction)
            to_real_target = self.real_target - pos
            t = np.dot(to_real_target, direction_unit)
            closest_point = pos + t * direction_unit
            min_distance = np.linalg.norm(self.real_target - closest_point)

            threat_level = "高" if min_distance <= 50 else "中" if min_distance <= 200 else "低"

            threat_data.append({
                '导弹': name,
                '到假目标距离(m)': f"{dist_to_fake:.0f}",
                '飞行时间(s)': f"{time_to_fake:.2f}",
                '最近距离(m)': f"{min_distance:.1f}",
                '威胁等级': threat_level
            })

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('导弹威胁等级', '飞行时间对比', '距离分析', '拦截窗口'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        df = pd.DataFrame(threat_data)

        # 威胁等级柱状图
        threat_colors = {'高': 'red', '中': 'orange', '低': 'green'}
        fig.add_trace(
            go.Bar(
                x=df['导弹'],
                y=[3 if x == '高' else 2 if x == '中' else 1 for x in df['威胁等级']],
                marker_color=[threat_colors[x] for x in df['威胁等级']],
                name='威胁等级'
            ),
            row=1, col=1
        )

        # 飞行时间散点图
        fig.add_trace(
            go.Scatter(
                x=df['导弹'],
                y=df['飞行时间(s)'].astype(float),
                mode='markers+lines',
                marker=dict(size=12, color='blue'),
                name='飞行时间'
            ),
            row=1, col=2
        )

        # 距离分析
        fig.add_trace(
            go.Bar(
                x=df['导弹'],
                y=df['最近距离(m)'].astype(float),
                marker_color='purple',
                name='最近距离'
            ),
            row=2, col=1
        )

        # 拦截窗口时间线
        intercept_times = np.array([float(x) for x in df['飞行时间(s)']])
        fig.add_trace(
            go.Scatter(
                x=intercept_times,
                y=df['导弹'],
                mode='markers+text',
                marker=dict(size=15, color='red'),
                text=[f"{t:.1f}s" for t in intercept_times],
                textposition="middle right",
                name='拦截时机'
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text="🎯 导弹威胁分析仪表板",
            showlegend=False,
            height=800
        )

        return fig, df

    def create_animation_simulation(self, duration=70):
        """创建动画模拟"""
        frames = []
        time_steps = np.linspace(0, duration, 100)

        for t in time_steps:
            frame_data = []

            # 导弹位置
            for name, pos in self.missiles.items():
                direction = self.fake_target - pos
                direction_unit = direction / np.linalg.norm(direction)
                current_pos = pos + direction_unit * self.missile_speed * t

                # 检查是否到达目标
                if np.linalg.norm(current_pos - self.fake_target) > np.linalg.norm(pos - self.fake_target):
                    current_pos = pos

                frame_data.append(go.Scatter3d(
                    x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name=name
                ))

            frames.append(go.Frame(data=frame_data, name=f"t={t:.1f}s"))

        # 创建初始图形
        fig = self.create_interactive_3d_scene()
        fig.frames = frames

        # 添加播放控件
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 100, "redraw": True},
                                        "fromcurrent": True}],
                        "label": "▶️ 播放",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "⏸️ 暂停",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

        return fig
#%%
print("🚀 启动交互式烟幕干扰弹可视化系统...")
#%%
visualizer = SmokeScreenVisualizer()
#%%
# 创建3D场景
print("📊 生成3D交互场景...")
scene_fig = visualizer.create_interactive_3d_scene()
#%%
# 创建威胁分析
print("🎯 分析导弹威胁...")
dashboard_fig, threat_df = visualizer.create_threat_analysis_dashboard()
#%%
# 创建动画
print("🎬 准备动画模拟...")
animation_fig = visualizer.create_animation_simulation()
#%%
# 显示结果
print("\n" + "=" * 50)
print("📋 威胁分析报告:")
print(threat_df.to_string(index=False))
print("=" * 50)
#%%

scene_fig.show()

#%%
dashboard_fig.show()
#%%
animation_fig.show()
#%%
# 保存图像
scene_fig.write_html(".././ImageOutput/00/01-interactive_3d_scene.html")
#%%
dashboard_fig.write_html(".././ImageOutput/00/02-threat_analysis_dashboard.html")
#%%
animation_fig.write_html(".././ImageOutput/00/03-missile_animation.html")
#%%

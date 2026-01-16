import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import os

# === 配置 ===
XML_PATH = "simulate_environment.xml"  # 确保文件名一致
DATA_FILE = "motion_data.json"
PLAYBACK_SPEED = 1.0  # 播放速度：0.5 = 半速， 2.0 = 两倍速

def main():
    if not os.path.exists(DATA_FILE):
        print(f"❌ 错误：找不到 {DATA_FILE}，请先运行 record 脚本。")
        return

    # 1. 加载数据
    print(f"正在读取 {DATA_FILE} ...")
    with open(DATA_FILE, "r") as f:
        motion_data = json.load(f)
    
    # 确保按顺序播放
    stage_order = ["INIT", "REACH", "GRASP", "LIFT", "POUR"]
    
    # 2. 加载模型
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("========================================")
    print("【G1 动作回放 - 视觉模式】")
    print("正在演示动作路径 (无物理碰撞)...")
    print("========================================")

    # 启动 Viewer (被动模式)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # === 0. 先复位到初始状态 ===
        if "INIT" in motion_data:
            data.qpos[:] = np.array(motion_data["INIT"])
            mujoco.mj_forward(model, data) # 刷新骨骼
            viewer.sync()
            time.sleep(1.0) # 停顿一下让人看清

        # === 1. 开始循环播放 ===
        prev_stage = "INIT"
        
        for stage in stage_order:
            if stage == "INIT": continue
            if stage not in motion_data: 
                print(f"⚠️ 跳过未录制的动作: {stage}")
                continue

            print(f"▶️ 正在执行: {prev_stage} -> {stage}")

            # 获取起始点和终点
            start_q = np.array(motion_data[prev_stage])
            end_q = np.array(motion_data[stage])

            # 动画参数
            duration = 2.0 / PLAYBACK_SPEED  # 假设每个动作耗时2秒
            fps = 60
            steps = int(duration * fps)

            # 插值循环
            for i in range(steps):
                if not viewer.is_running(): break
                
                # 计算进度 (0.0 到 1.0)
                alpha = i / steps
                
                # 线性插值公式: 当前 = 起点 + (终点-起点)*进度
                current_q = start_q + (end_q - start_q) * alpha
                
                # --- 核心：强制写入位置 ---
                data.qpos[:] = current_q
                
                # --- 核心：刷新计算 (Kinematics) ---
                # 这会让 MuJoCo 根据新的 qpos 计算手脚在哪里，但不会计算重力摔倒
                mujoco.mj_forward(model, data)
                
                # 刷新画面
                viewer.sync()
                
                # 控制帧率
                time.sleep(1.0 / fps)

            prev_stage = stage
            time.sleep(0.5) # 动作之间稍微停顿

        print("✅ 回放结束。")
        while viewer.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()
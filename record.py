import mujoco
import mujoco.viewer
import numpy as np
import json
import os
import threading
import sys

# === 配置 ===
XML_PATH = "simulate_environment.xml" # 确保文件名对
SAVE_FILE = "motion_data.json"
STAGES = ["INIT", "REACH", "GRASP", "LIFT", "POUR"]

# 全局变量，用于线程间通信
recorded_data = {}
keep_running = True

def input_thread(data):
    """
    这个函数会在后台运行，专门负责听你在终端按回车
    """
    print(">>> 输入线程已就绪。")
    
    for stage in STAGES:
        # 1. 提示用户
        print(f"\n------------------------------------------------")
        print(f"👉 下一步目标: 【 {stage} 】")
        print(f"请在 Viewer 里摆好姿势 (建议用右侧 Joints 滑块)")
        print(f"摆好后，请切回此终端窗口，按 【回车键】 保存")
        print(f"------------------------------------------------")
        
        # 2. 等待回车 (这就不会卡住 Viewer 了)
        sys.stdin.readline()
        
        # 3. 偷数据
        # 因为 data 是共享内存，我们直接读就行
        current_qpos = data.qpos.copy().tolist()
        recorded_data[stage] = current_qpos
        print(f"✅ 已捕获: {stage}")

    print("\n🎉 所有动作录制完成！")
    print("请直接关闭 MuJoCo 窗口，文件将自动保存。")

def main():
    if not os.path.exists(XML_PATH):
        print(f"错误：找不到文件 {XML_PATH}")
        return

    # 加载模型
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # === 这里的关键设置 ===
    # 1. 把重力关掉 (设为0)，这样你拖机器人它不会掉下来
    model.opt.gravity[:] = 0
    # 2. 或者我们直接用“暂停+滑块”的战术，这最稳
    
    print("========================================================")
    print("【G1 示教器 - 多线程版】")
    print("1. 窗口打开后，建议按空格【暂停】仿真。")
    print("2. 使用右侧面板的【Joints】滑块来调整关节角度。")
    print("   (这是最精准的方法，因为机器人不会乱跑)")
    print("3. 满意后，点一下这个黑色终端窗口，按【回车】。")
    print("========================================================")

    # 启动后台监听线程
    t = threading.Thread(target=input_thread, args=(data,), daemon=True)
    t.start()

    # 启动标准 Viewer (阻塞式)
    # 这会给你最流畅的原生体验
    mujoco.viewer.launch(model, data)

    # 窗口关闭后，保存文件
    if len(recorded_data) > 0:
        print(f"\n正在保存数据到 {SAVE_FILE} ...")
        with open(SAVE_FILE, "w") as f:
            json.dump(recorded_data, f, indent=4)
        print("保存成功！")

if __name__ == "__main__":
    main()
import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import os

# === 配置 ===
XML_PATH = "simulate_environment.xml"
DATA_FILE = "motion_data.json"
SPEED = 1.0 # 动作速度

# === PD 控制参数 (如果机器人发抖，减小Kp；如果无力，增大Kp) ===
# 这是一个通用的参数，针对 G1 这种体型的机器人
KP = 80.0   # 刚度 (Stiffness)
KD = 5.0    # 阻尼 (Damping)

def main():
    if not os.path.exists(DATA_FILE):
        print("❌ 找不到数据文件")
        return

    # 1. 加载数据
    with open(DATA_FILE, "r") as f:
        motion_data = json.load(f)

    # 2. 加载模型
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 3. 建立映射：执行器(Actuator) -> 关节数据(qpos)
    # 因为 data.ctrl 是按执行器排序的，而 motion_data 是按 qpos 排序的
    # 我们需要知道第 i 个电机控制的是 qpos 里的第几个数
    actuator_to_qpos = []
    
    print("正在映射电机...")
    for i in range(model.nu): # 遍历所有电机
        # 获取该电机控制的关节 ID (joint ID)
        # trnid 格式通常是 [joint_id, type]
        joint_id = model.jnt_qposadr[model.actuator_trnid[i, 0]]
        actuator_to_qpos.append(joint_id)
    
    print(f"✅ 映射完成，共控制 {len(actuator_to_qpos)} 个自由度")

    # 4. 准备动作序列
    stage_order = ["INIT", "REACH", "GRASP", "LIFT", "POUR"]
    
    # 启动 Viewer (主动模式，因为我们要控制物理)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # === 初始化位置 ===
        # 先瞬间移动到 INIT 位置，防止一开始飞出去
        if "INIT" in motion_data:
            data.qpos[:] = np.array(motion_data["INIT"])
            mujoco.mj_forward(model, data)
        
        # 等待物理稳定
        for _ in range(100):
            mujoco.mj_step(model, data)
            viewer.sync()
            
        prev_stage = "INIT"

        for stage in stage_order:
            if stage == "INIT": continue
            if stage not in motion_data: continue

            print(f"⚡ 执行物理动作: {stage}")

            start_q = np.array(motion_data[prev_stage])
            target_q_full = np.array(motion_data[stage])
            
            # 计算这一段需要多少步
            duration = 2.0 / SPEED * 3
            steps = int(duration / model.opt.timestep) # 物理步数


            for i in range(steps):
                if not viewer.is_running(): break
                
                # 1. 计算插值
                alpha = i / steps
                current_target_full = start_q + (target_q_full - start_q) * alpha
                
                # === ✨ 核心修改：捏紧策略 ✨ ===
                # 如果处于抓取、提升、倒水阶段，强行修改手指的目标位置
                if stage in ["GRASP", "LIFT", "POUR"]:
                    # 我们直接遍历所有电机，这样更安全，也能直接找到对应的关节
                    for i in range(model.nu): # model.nu 是电机总数
                        # 1. 获取这个电机控制的关节 ID
                        joint_id = model.actuator_trnid[i, 0]
                        # 2. 获取关节名字
                        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        
                        # 3. 安全检查：如果名字是 None，直接跳过，防止报错
                        if j_name is None:
                            continue

                        # 4. 匹配你的 XML 里的手指关节名字
                        # 你的 XML 里有: left_index_proximal_joint, left_thumb_proximal_joint 等
                        # 只要名字里带 "left" 和 "proximal"，就是我们需要捏紧的手指
                        if "left" in j_name and "proximal" in j_name:
                            # 找到这个关节在 qpos 数组里的位置
                            q_idx = model.jnt_qposadr[joint_id]
                            
                            # 5. 施加捏紧力
                            # 你的范围是 0~1.46，所以我们强行给目标值 +0.8
                            # 这样如果当前是 1.0 (没握紧)，目标变成 1.8 (超过极限)，电机就会拼命输出力矩
                            current_target_full[q_idx] += 0.25
                            
                            # (可选) 拇指可能还需要加上 metacarpal 关节才够紧
                            if "thumb" in j_name and "metacarpal" in j_name:
                                current_target_full[q_idx] += 0.5
                


                # 2. PD 控制循环 (保持不变)
                for act_id, q_idx in enumerate(actuator_to_qpos):
                    target_pos = current_target_full[q_idx] # 这里使用的是刚才被修改过的强力目标
                    current_pos = data.qpos[q_idx]
                    current_vel = data.qvel[q_idx]
                    
                    # 对于手指，甚至可以单独给一个更大的 KP
                    # torque = KP * (target_pos - current_pos) - KD * current_vel
                    # 如果是手指，给双倍力气：
                    local_kp = KP * 2.0 if stage in ["GRASP", "LIFT"] else KP
                    torque = local_kp * (target_pos - current_pos) - KD * current_vel

                    data.ctrl[act_id] = torque

                # 3. 物理步进 (让物理引擎应用这些力)
                mujoco.mj_step(model, data)
                
                # 4. 刷新画面 (每 30 步刷新一次，节省显卡)
                if i % 30 == 0:
                    viewer.sync()

            # === ✨ 修复：抓取后的稳定缓冲 ✨ ===
            # 如果刚刚完成了 GRASP，准备进入 LIFT 之前
            if stage == "GRASP":
                print("🛑 正在加固抓取 (等待物理稳定)...")
                stabilize_steps = 100  # 约 0.2秒 - 0.5秒
                
                # 保持 GRASP 的最后一帧姿态
                last_target = target_q_full.copy()
                
                # 同样要应用“捏紧策略” (这一步很关键，保持捏紧！)
                if "GRASP" in ["GRASP", "LIFT", "POUR"]: # 这里逻辑肯定是 True，为了保持一致性
                     for i in range(model.nu):
                        joint_id = model.actuator_trnid[i, 0]
                        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                        if j_name and "left" in j_name and "proximal" in j_name:
                             q_idx = model.jnt_qposadr[joint_id]
                             # 保持之前的捏紧力度
                             last_target[q_idx] += 0.2  # 这里的数值要和你循环里的一致！

                for _ in range(stabilize_steps):
                    # 仅维持 PD 控制，不移动身体
                    for act_id, q_idx in enumerate(actuator_to_qpos):
                        target_pos = last_target[q_idx]
                        current_pos = data.qpos[q_idx]
                        current_vel = data.qvel[q_idx]
                        
                        # 重新计算力矩
                        finger_kp = 40.0
                        torque = finger_kp * (target_pos - current_pos) - KD * current_vel
                        
                        # 记得加上你的力矩限制 (如果有的话)
                        torque = np.clip(torque, -1.5, 1.5) 
                        data.ctrl[act_id] = torque
                    
                    mujoco.mj_step(model, data)
                    if _ % 20 == 0: viewer.sync()

            prev_stage = stage
            
            # 动作完成后，保持一段时间（Hold）
            print(f"   (保持姿态 {stage})...")
            hold_steps = int(1.0 / model.opt.timestep)
            for _ in range(hold_steps):
                # 保持目标不变，继续维持 PD 控制
                for act_id, q_idx in enumerate(actuator_to_qpos):
                    target_pos = target_q_full[q_idx] # 目标就是终点
                    current_pos = data.qpos[q_idx]
                    current_vel = data.qvel[q_idx]
                    data.ctrl[act_id] = KP * (target_pos - current_pos) - KD * current_vel
                
                mujoco.mj_step(model, data)
                if _ % 30 == 0: viewer.sync()

        print("演示结束。")
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
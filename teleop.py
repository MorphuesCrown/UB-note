import asyncio
import time
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import sys
import cv2 
from revo2_utils import *
MODEL_PATH = "brainco-lefthand-v2.xml"
LOG_FILE = "teleop_touch_data.csv"
PORT_NAME = "/dev/ttyUSB0" 
SLAVE_ID = 0x7e
CONTROL_FREQ = 30 


class TactileDashboard:
    def __init__(self, max_force=5.0):
        """
        :param max_force: 预计最大受力(牛顿)，用于归一化显示柱状图高度
        """
        self.width = 600
        self.height = 300
        self.max_force = max_force
        self.finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        # 初始化画布 (黑色背景)
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def update(self, forces):
        """
        绘制柱状图
        :param forces: 包含5个手指法向力的列表 [thumb, index, middle, ring, pinky]
        """
        # 重置画布
        self.canvas[:] = (30, 30, 30) # 深灰色背景
        
        bar_width = 80
        spacing = 30
        start_x = 40
        bottom_y = self.height - 50

        for i, force in enumerate(forces):
            # 1. 计算高度
            # 限制显示范围，防止爆出屏幕
            display_force = np.clip(force, 0, self.max_force)
            ratio = display_force / self.max_force
            bar_h = int(ratio * (self.height - 100))
            
            x = start_x + i * (bar_width + spacing)
            top_y = bottom_y - bar_h
            
            # 2. 确定颜色 (绿色 -> 红色 渐变)
            # BGR 格式
            color = (0, int(255 * (1 - ratio)), int(255 * ratio))
            
            # 3. 画柱子
            cv2.rectangle(self.canvas, (x, top_y), (x + bar_width, bottom_y), color, -1)
            
            # 4. 画文字 (手指名)
            cv2.putText(self.canvas, self.finger_names[i], (x + 10, bottom_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 5. 画读数 (力的大小)
            force_text = f"{force:.2f}N"
            cv2.putText(self.canvas, force_text, (x + 10, top_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示窗口
        cv2.imshow("Realtime Tactile Data", self.canvas)
        cv2.waitKey(1) 

    def close(self):
        cv2.destroyAllWindows()

class SimRealBridge:
    def __init__(self, model_path):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        
        self.actuator_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.m.nu)]
        self.ctrl_ranges = self.m.actuator_ctrlrange
        
        self.logs = []
        self.start_time = 0
        self.client = None
        self.slave_id = None
        
        self.dashboard = TactileDashboard(max_force=3.0) # 假设最大力为 3N，可根据传感器调整

    def map_sim_to_real(self):
        positions = []
        for i in range(6): 
            sim_val = self.d.ctrl[i]
            min_rad, max_rad = self.ctrl_ranges[i]
            clamped_val = np.clip(sim_val, min_rad, max_rad)
            if (max_rad - min_rad) == 0:
                ratio = 0
            else:
                ratio = (clamped_val - min_rad) / (max_rad - min_rad)
            unit_val = int(ratio * 1000)
            positions.append(unit_val)
        # BrainCo 映射顺序
        positions = [positions[1], positions[0], positions[2], positions[3], positions[4], positions[5]]
        return positions

    async def connect_real_hand(self):
        logger.info(f"Connecting to Revo2 on {PORT_NAME}...")
        try:
            self.client, self.slave_id = await open_modbus_revo2(port_name=PORT_NAME)
        except Exception:
            self.client = await libstark.modbus_open(PORT_NAME, libstark.Baudrate.Baud460800)
            self.slave_id = SLAVE_ID
            
        if not self.client:
            logger.error("Connection failed.")
            sys.exit(1)
            
        await self.client.set_finger_unit_mode(self.slave_id, libstark.FingerUnitMode.Normalized)
        await self.client.touch_sensor_setup(self.slave_id, 0x1F) 
        logger.info("Real hand connected and sensors calibrated.")

    async def loop(self):
        self.start_time = time.time()
        speeds = [1000] * 6 
        
        print("========================================")
        print("系统就绪！")
        print("1. MuJoCo 窗口: 拖动滑条控制")
        print("2. OpenCV 窗口: 显示实时触觉受力")
        print("========================================")

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                loop_start = time.time()
                
                # --- 1. MuJoCo 步进 ---
                mujoco.mj_step(self.m, self.d)
                viewer.sync()
                
                # --- 2. 驱动真机 ---
                target_positions = self.map_sim_to_real()
                await self.client.set_finger_positions_and_speeds(
                    self.slave_id, target_positions, speeds
                )
                
                # --- 3. 读取触觉 & 可视化 ---
                try:
                    touch_status = await self.client.get_touch_sensor_status(self.slave_id)
                    
                    # 提取法向力用于显示
                    normal_forces = [
                        touch_status[0].normal_force1, # Thumb
                        touch_status[1].normal_force1, # Index
                        touch_status[2].normal_force1, # Middle
                        touch_status[3].normal_force1, # Ring
                        touch_status[4].normal_force1  # Pinky
                    ]
                    
                    self.dashboard.update(normal_forces)

                    # 记录数据
                    current_t = time.time() - self.start_time
                    log_entry = {'time': current_t}
                    for i, name in enumerate(self.actuator_names):
                        log_entry[f'cmd_{name}'] = target_positions[i]
                        
                    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
                    for i, finger in enumerate(touch_status):
                        fname = finger_names[i]
                        log_entry[f'{fname}_force_n'] = finger.normal_force1
                        log_entry[f'{fname}_force_t'] = finger.tangential_force1
                    
                    self.logs.append(log_entry)
                    
                except Exception as e:
                    # 偶尔的串口超时不要崩溃，打印警告即可
                    pass 
                    # logger.warning(f"Sensor read error: {e}")

                # --- 4. 频率控制 ---
                elapsed = time.time() - loop_start
                wait_time = (1.0 / CONTROL_FREQ) - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

        # 退出处理
        self.dashboard.close() # [新增] 关闭窗口
        self.save_data()
        libstark.modbus_close(self.client)

    def save_data(self):
        if not self.logs:
            print("No data collected.")
            return
        df = pd.DataFrame(self.logs)
        df.to_csv(LOG_FILE, index=False)
        print(f"\n采集结束。数据已保存至: {LOG_FILE}")

async def main():
    bridge = SimRealBridge(MODEL_PATH)
    await bridge.connect_real_hand()
    await bridge.loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
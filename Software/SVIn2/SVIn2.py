import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# ===================== 基础工具函数（对应数学建模） =====================
def quat2rot(q):
    """四元数转旋转矩阵（对应{}_W R_I）
    q: 四元数 [w, x, y, z]
    """
    w, x, y, z = q
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])
    return R

def rot2quat(R):
    """旋转矩阵转四元数"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def imu_preintegrate(imu_data, dt):
    """IMU预积分（对应e_s误差项的核心）
    imu_data: [gyro, acc] 陀螺(rad/s)、加速度计(m/s²)
    dt: 时间步长(s)
    返回：预积分后的相对位置/姿态/速度
    """
    gyro, acc = imu_data
    # 初始状态
    R = np.eye(3)  # 旋转矩阵
    p = np.zeros(3) # 位置
    v = np.zeros(3) # 速度
    # 离散化积分（对应IMU预积分公式）
    for i in range(len(gyro)):
        # 陀螺积分更新姿态
        w = gyro[i]
        R = R @ (np.eye(3) + dt * np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ]))
        # 加速度计积分更新速度和位置
        a = R @ acc[i] + np.array([0, 0, 9.8]) # 加重力
        v += a * dt
        p += v * dt + 0.5 * a * dt**2
    return p, rot2quat(R), v

# ===================== 传感器数据模拟 =====================
def generate_sensor_data(t_total=10, dt=0.01):
    """模拟水下机器人轨迹+传感器数据（带噪声）
    返回：真实轨迹、IMU数据、视觉特征、声纳/深度数据
    """
    t = np.arange(0, t_total, dt)
    N = len(t)
    
    # 1. 真实轨迹（水下螺旋轨迹，模拟机器人运动）
    true_p = np.zeros((N, 3))  # 位置
    true_q = np.zeros((N, 4))  # 姿态四元数
    true_v = np.zeros((N, 3))  # 速度
    for i in range(N):
        # 螺旋轨迹：x=cos(t), y=sin(t), z=t/10（水下深度缓慢增加）
        true_p[i] = [np.cos(t[i]), np.sin(t[i]), t[i]/10]
        true_v[i] = [-np.sin(t[i]), np.cos(t[i]), 0.1]  # 速度
        true_q[i] = rot2quat(np.array([                # 姿态（缓慢旋转）
            [np.cos(t[i]/10), -np.sin(t[i]/10), 0],
            [np.sin(t[i]/10), np.cos(t[i]/10), 0],
            [0, 0, 1]
        ]))
    
    # 2. IMU数据（加高斯噪声，模拟真实传感器）
    gyro_noise = np.random.normal(0, 0.01, (N, 3))  # 陀螺噪声
    acc_noise = np.random.normal(0, 0.1, (N, 3))    # 加速度计噪声
    gyro = np.zeros((N, 3))
    acc = np.zeros((N, 3))
    for i in range(N):
        # 真实陀螺测量（角速度）
        gyro[i] = [0, 0, 0.1] + gyro_noise[i]  # 绕z轴旋转
        # 真实加速度计测量（去除重力）
        R = quat2rot(true_q[i])
        acc[i] = R.T @ (np.array([-np.cos(t[i]), -np.sin(t[i]), 0]) + acc_noise[i])
    
    # 3. 视觉特征（模拟立体相机观测，每10帧一个特征）
    visual_feat = []
    for i in range(0, N, 10):
        # 特征在世界坐标系下的位置（轨迹附近）
        feat_p = true_p[i] + np.random.normal(0, 0.05, 3)
        # 像素坐标（简化投影模型）
        u = 320 + 100 * feat_p[0] + np.random.normal(0, 2)
        v = 240 + 100 * feat_p[1] + np.random.normal(0, 2)
        visual_feat.append([i, u, v, feat_p[0], feat_p[1], feat_p[2]])
    
    # 4. 声纳+深度数据（模拟测距+深度测量）
    sonar = true_p[:, 2] + np.random.normal(0, 0.02, N)  # 声纳测距（深度方向）
    depth = true_p[:, 2] + np.random.normal(0, 0.01, N)  # 深度传感器（更精准）
    
    return {
        "t": t, "dt": dt,
        "true_p": true_p, "true_q": true_q, "true_v": true_v,
        "imu": {"gyro": gyro, "acc": acc},
        "visual": visual_feat,
        "sonar": sonar, "depth": depth
    }

# ===================== SVIn2核心融合逻辑 =====================
def reprojection_error(feat, robot_p, robot_q, K=np.array([[500,0,320],[0,500,240],[0,0,1]])):
    """视觉重投影误差（对应e_r）
    feat: [帧号, u, v, x, y, z] 视觉特征
    robot_p: 机器人位置 {}_W p_I
    robot_q: 机器人姿态 {}_W q_I
    K: 相机内参
    """
    _, u_obs, v_obs, x_w, y_w, z_w = feat
    # 世界坐标转相机坐标
    R = quat2rot(robot_q)
    p_c = R.T @ (np.array([x_w, y_w, z_w]) - robot_p)
    # 投影到像素平面
    u_pred = K[0,0] * p_c[0]/p_c[2] + K[0,2]
    v_pred = K[1,1] * p_c[1]/p_c[2] + K[1,2]
    # 重投影误差
    return np.sqrt((u_obs - u_pred)**2 + (v_obs - v_pred)**2)

def cost_function(x, data):
    """多传感器融合代价函数（对应J(x)）
    x: 优化变量 [p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z]
    data: 传感器数据
    """
    # 解包优化变量
    p = x[:3]
    q = x[3:7]
    v = x[7:]
    q = q / np.linalg.norm(q)  # 四元数归一化
    
    # 1. 视觉重投影误差项
    err_visual = 0
    for feat in data["visual"]:
        err_visual += reprojection_error(feat, p, q)
    
    # 2. IMU预积分误差项
    imu_data = (data["imu"]["gyro"], data["imu"]["acc"])
    p_pre, q_pre, v_pre = imu_preintegrate(imu_data, data["dt"])
    err_imu = np.linalg.norm(p - p_pre) + np.linalg.norm(q - q_pre) + np.linalg.norm(v - v_pre)
    
    # 3. 声纳+深度误差项
    err_sonar = np.linalg.norm(p[2] - data["sonar"][-1])
    err_depth = np.linalg.norm(p[2] - data["depth"][-1])
    
    # 加权求和（对应P_r/P_s/P_t/P_u信息矩阵）
    total_err = 0.5 * err_visual + 1.0 * err_imu + 0.3 * err_sonar + 0.7 * err_depth
    return total_err

# ===================== 主函数：模拟演示 =====================
if __name__ == "__main__":
    print("代码开始执行...")
    # 1. 生成模拟数据
    print("正在生成模拟数据...")
    data = generate_sensor_data(t_total=10, dt=0.01)
    N = len(data["t"])
    print(f"模拟数据生成完成，共 {N} 个时间步")
    
    # 2. 初始化估计状态（带初始偏差）
    print("正在初始化估计状态...")
    init_p = np.array([0.1, 0.1, 0.0])  # 初始位置偏差
    init_q = rot2quat(np.eye(3))       # 初始姿态
    init_v = np.array([0.0, 0.0, 0.0]) # 初始速度
    x0 = np.concatenate([init_p, init_q, init_v])
    print(f"初始位置: {init_p}, 初始姿态: {init_q}, 初始速度: {init_v}")
    
    # 3. 非线性优化（最小化代价函数）
    print("正在进行非线性优化...")
    res = minimize(cost_function, x0, args=(data,), method="L-BFGS-B")
    est_p = res.x[:3]    # 估计位置
    est_q = res.x[3:7]   # 估计姿态
    est_q = est_q / np.linalg.norm(est_q)
    est_v = res.x[7:]    # 估计速度
    print(f"优化完成，代价函数值: {res.fun}")
    
    # 4. 可视化结果
    try:
        print("正在生成可视化结果...")
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1：3D轨迹对比
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(data["true_p"][:,0], data["true_p"][:,1], data["true_p"][:,2], label="真实轨迹", color="blue")
        ax1.scatter(est_p[0], est_p[1], est_p[2], label="估计位置", color="red", s=100, marker="*")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("水下机器人3D轨迹（真实vs估计）")
        ax1.legend()
        
        # 子图2：深度方向误差（声纳+深度传感器）
        ax2 = fig.add_subplot(132)
        ax2.plot(data["t"], data["true_p"][:,2], label="真实深度", color="blue")
        ax2.plot(data["t"], data["sonar"], label="声纳测量（噪声）", color="orange", alpha=0.5)
        ax2.plot(data["t"], data["depth"], label="深度传感器（精准）", color="green", alpha=0.8)
        ax2.axhline(est_p[2], color="red", linestyle="--", label="估计深度")
        ax2.set_xlabel("时间 (s)")
        ax2.set_ylabel("深度 (m)")
        ax2.set_title("深度测量与估计")
        ax2.legend()
        
        # 子图3：代价函数收敛曲线
        ax3 = fig.add_subplot(133)
        # 模拟优化过程的代价函数值
        cost_history = [cost_function(x0, data)]
        for i in range(10):
            x_temp = x0 - 0.1 * res.jac  # 简化梯度下降
            cost_history.append(cost_function(x_temp, data))
        ax3.plot(range(len(cost_history)), cost_history, color="purple")
        ax3.set_xlabel("迭代次数")
        ax3.set_ylabel("代价函数值")
        ax3.set_title("融合优化收敛过程")
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('simulation_result.png')
        print("图形已保存为 simulation_result.png")
        plt.close()  # 关闭图形，释放资源
        print("可视化完成")
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        print("跳过可视化步骤")
    
    # 输出关键结果
    print("\n===== 模拟结果 =====")
    print(f"真实最终位置: {data['true_p'][-1]}")
    print(f"估计最终位置: {est_p}")
    print(f"位置误差 (m): {np.linalg.norm(data['true_p'][-1] - est_p)}")
    print(f"代价函数最终值: {res.fun}")
    print("\n代码执行完成！")

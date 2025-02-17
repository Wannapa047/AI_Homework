from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv
import gym

# สร้าง environment
env = gym.make('CartPole-v1')  # เปลี่ยนเป็น environment ที่เหมาะสมกับโปรเจกต์ของคุณ

# ทำให้ environment เป็น vectorized environment (สำหรับการฝึก)
env = DummyVecEnv([lambda: env])

# สร้าง DQN โมเดล
model = DQN('MlpPolicy', env, verbose=1)

# ฝึกโมเดล
model.learn(total_timesteps=10000)

# ทดสอบโมเดล
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

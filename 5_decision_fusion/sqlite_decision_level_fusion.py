# ==================== 2. 加载LSTM模型并生成P1序列 ====================
print("加载LSTM模型...")
checkpoint = torch.load(lstm_model_path, map_location=torch.device('cpu'), weights_only=False)
seq_len = checkpoint['seq_len']  # 12
scaler_lstm = checkpoint['scaler']

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.squeeze(1)

lstm_model = LSTM_Classifier()
lstm_model.load_state_dict(checkpoint['model_state_dict'])
lstm_model.eval()

X_pcs_scaled = scaler_lstm.transform(X_pcs_all).astype(np.float32)

print("生成P1序列...")
P1 = np.full(len(time_index), np.nan, dtype=np.float32)
with torch.no_grad():
    for i in range(seq_len, len(X_pcs_scaled)):
        seq = X_pcs_scaled[i-seq_len:i]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        prob = lstm_model(seq_tensor).item()
        P1[i] = prob

# ==================== 3. 加载SVM模型并生成P2序列 ====================
print("加载SVM模型...")
svm_data = joblib.load(svm_model_path)
svm_model = svm_data['model']
scaler_svm = svm_data['scaler']

X_svm = np.column_stack([X_pcs_all, X_patrol_all, v_mining_all])
X_svm_scaled = scaler_svm.transform(X_svm)
print("生成P2序列...")
P2 = svm_model.predict_proba(X_svm_scaled)[:, 1].astype(np.float32)

# ==================== 4. 生成P3序列 ====================
P3 = df_p3_aligned['P3'].values.astype(np.float32)

# ==================== 5. 贝叶斯决策级融合 ====================
print("进行贝叶斯融合...")
p_H1 = 0.05
p_H0 = 1 - p_H1

s1, f1 = 0.9568, 0.1078
s2, f2 = 0.9159, 0.1042
s3, f3 = 0.8, 0.2

I1 = (P1 >= 0.5).astype(int)
I2 = (P2 >= 0.5).astype(int)
I3 = (P3 >= 0.5).astype(int)

posterior = np.full(len(time_index), np.nan, dtype=np.float32)
for i in range(len(time_index)):
    if np.isnan(P1[i]) or np.isnan(P2[i]) or np.isnan(P3[i]):
        continue
    lik_H1 = (s1**I1[i] * (1-s1)**(1-I1[i])) * \
             (s2**I2[i] * (1-s2)**(1-I2[i])) * \
             (s3**I3[i] * (1-s3)**(1-I3[i]))
    lik_H0 = (f1**I1[i] * (1-f1)**(1-I1[i])) * \
             (f2**I2[i] * (1-f2)**(1-I2[i])) * \
             (f3**I3[i] * (1-f3)**(1-I3[i]))
    numerator = p_H1 * lik_H1
    denominator = numerator + p_H0 * lik_H0
    posterior[i] = numerator / denominator if denominator > 0 else 0.0

# 预警等级划分
levels = np.full(len(posterior), np.nan)
for i in range(len(posterior)):
    p = posterior[i]
    if np.isnan(p):
        levels[i] = -1
    elif p < 0.3:
        levels[i] = 0
    elif p < 0.6:
        levels[i] = 1
    elif p < 0.8:
        levels[i] = 2
    else:
        levels[i] = 3

level_map = {-1: '未知', 0: '低风险(蓝)', 1: '中风险(黄)', 2: '高风险(橙)', 3: '极高风险(红)'}
level_colors = {-1: 'gray', 0: 'blue', 1: 'yellow', 2: 'orange', 3: 'red'}

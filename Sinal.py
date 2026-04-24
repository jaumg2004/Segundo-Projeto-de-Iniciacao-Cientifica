# --- Bibliotecas ---
import os
import numpy as np
import matplotlib as mpl
from matplotlib.lines import lineStyles

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['axes.formatter.limits'] = (-99, 99)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt



# --- Configurações ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
random.seed(10)
torch.manual_seed(10)


############################################################################################
# --- Funções de Canal ---
def rice_channel(N, kappa): #Calcúlo do canal Rice
    h_los = np.exp(1j * np.random.uniform(0, 2 * np.pi, N)) #componente LoS do vetor de canal
    h_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2) #componente NLoS do vetor de canal
    return np.sqrt(kappa / (1 + kappa)) * h_los + np.sqrt(1 / (1 + kappa)) * h_nlos


# Calcula a Perda de Percursso
def calculate_beta_mk(frequency, d_mk, alpha, temp_celsius, wind_speed):
    wavelength = 3e8 / frequency #calculo do compimento da onda
    beta = (wavelength ** 2) / ((4 * np.pi) ** 2 * (d_mk ** alpha))  #calcúlo do sinal beta
    temp_loss = (0.0002 * (temp_celsius - 25) ** 2 + 1) #calculo da perda pela temperatura
    if wind_speed <= 31.7:
        wind_loss = (np.exp(0.01 * 31.7) - 1) * wind_speed/31.7 + 1 #calculo da perda pela velocidade do vento se ela for menor que 31,7 km/h
    else:
        wind_loss = np.exp(0.01 * wind_speed)  #calculo da perda pela velocidade do vento se ela for maior que 31,7 km/h
    env_loss = temp_loss * wind_loss #calculo da perda total
    return beta / env_loss



# Calcula a Potência Recebida
def received_power(beta_mk, psi_mk, h_mk):
    inner = np.vdot(psi_mk, h_mk)
    return float(beta_mk * np.abs(inner) ** 2)


# --- Gerador de cenário aleatório ---
def generate_scenario(K, M, N, bounds, kappa=1.0):
    # IoTs (Kx3) em coordenadas absolutas [0, bounds[1]], z=1.0
    iot_xy = np.column_stack([
        np.random.uniform(0.0, bounds[1], K),
        np.random.uniform(0.0, bounds[1], K),
    ])
    iot_positions = np.hstack([iot_xy, np.full((K, 1), 1.0)])

    pb_positions = np.column_stack([
        np.random.uniform(0.0, bounds[1], size=M),
        np.random.uniform(0.0, bounds[1], size=M),
        np.full(M, 5.0)
    ])

    pb_positions = pb_positions.astype(float)
    pb_positions[:, 0:2] /= bounds[1]

    # Canais de Rice: (M, K, N)
    chans = np.zeros((M, K, N), dtype=complex)
    for m in range(M):
        for k in range(K):
            chans[m, k] = rice_channel(N, kappa=kappa)

    # Temperatura/vento: um valor escalar para todo o cenário
    temperature = np.random.uniform(-29.0, 62.3, K).astype(float)  # °C
    wind_speed  = np.random.uniform(0.0, 90.0, K).astype(float)    # km/h

    return iot_positions, pb_positions, chans, temperature, wind_speed


###################################################################################################
# --- Ambiente ---
class EnergyHarvestingEnv(gym.Env):
    def __init__(self, tau_k, mu, a, b, Omega, K, M, N, PT, frequency, alpha, bounds):
        super().__init__()

        self.K = K
        self.M = M

        self.N = N
        self.PT = PT

        self.frequency = frequency
        self.alpha = alpha
        self.bounds = bounds

        self.mu = mu
        self.a = a
        self.b = b
        self.Omega = Omega
        self.tau_k = tau_k[:K]

        # Inicializados com None; serão definidos a cada episódio via set_scenario()
        self.iot_positions = None
        self.pb_positions = None
        self.pb_positions_init = None
        self.realization_channels = None
        self.temperature = None
        self.wind_speed = None

        self.betas = np.zeros((M, K))
        self.collected_energies = np.zeros(K, dtype=np.float32)

        # métricas
        self.E_min = 1e-6  # 1 microjoule (limiar por passo)
        self.ever_harvested = np.zeros(self.K, dtype=bool)  # IoTs que já atingiram E_min em algum passo do episódio

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(M * 2,),
                                            dtype=np.float32)  # [0,1] -- Considera a posição dos PBs normalizada
        self.action_delta = 0.001  # Move o PB no espaço -- 1 - 30 m / 0.05 - 15 m / 0.001 - 0.3 m
        self.action_space = spaces.Box(low=-self.action_delta, high=self.action_delta, shape=(M * 2,), dtype=np.float32)

    # CALCULA O GANHO MÉDIO DE POT~ENCIA DO CANAL ENTRE O PB E OS DISPOSITIVOS IoT --- Para o calculo do beta as posições não podem ser normalizadas
    def _calculate_betas(self):
        pb_positions_denorm = np.hstack((self.pb_positions[:, :2] * self.bounds[1], self.pb_positions[:, 2:3]))

        for m in range(self.M):
            for k in range(self.K):
                dx = pb_positions_denorm[m, 0] - self.iot_positions[k, 0]
                dy = pb_positions_denorm[m, 1] - self.iot_positions[k, 1]
                dz = pb_positions_denorm[m, 2] - self.iot_positions[k, 2]

                d_mk = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 0.1
                self.betas[m, k] = calculate_beta_mk(self.frequency, d_mk, self.alpha, float(self.temperature[0]),
                                                     float(self.wind_speed[0]))

    # Determina a posição dos dispositivos IoT
    def set_scenario(self, iot_positions, pb_positions, channels, temperature, wind_speed):
        self.iot_positions = iot_positions[:self.K, :]
        self.realization_channels = channels

        # Usar um único valor de temperatura e vento para todos os IoTs --- Os dispositivos estão no mesmo cenário, então vento e temperatura são iguais para todos!
        self.temperature = np.full(self.K, temperature[0])
        self.wind_speed = np.full(self.K, wind_speed[0])

        # PB inicial vindo do dataset
        self.pb_positions = pb_positions.copy()
        self.pb_positions_init = self.pb_positions.copy()

        self._calculate_betas()
        self.collected_energies = np.zeros(self.K, dtype=np.float32)

    # Reiniciliza o ambiente em cada episódio
    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.collected_energies = np.zeros(self.K, np.float32)
        self.pb_positions = self.pb_positions_init.copy()
        self._calculate_betas()
        self.ever_harvested = np.zeros(self.K, dtype=bool)
        return (self.pb_positions[:, :2]).flatten().astype(np.float32), {}

    # Determina a ação do PB e calcula a Recompensa
    def step(self, action):
        action = np.array(action).reshape(self.M, 2)

        self.pb_positions[:, :2] = np.clip(self.pb_positions[:, :2] + action, 0.0, 1.0)#Normaliza as posições dos dispositvos IoT utilizando a ação
        self._calculate_betas() #Chama a função de calcúlo do ganho médio de potência do canal entre o m-ésimo PB e o l-ésimo dispositivo IoT

        #Cria um array da potência recebida por cada dispositivo
        P_k_array = np.zeros(self.K, dtype=np.float32)
        for m in range(self.M):
            for k in range(self.K):
                h_mk = self.realization_channels[m, k] #determina o vetor de canal entre o m-ésimo PB e o k-ésimo dispositivo IoT
                psi_mk = np.sqrt(self.PT / self.N) * (h_mk / np.abs(h_mk)) #vetor beamforming entre o M-ésimo PB e o k-ésimo dispositivo IoT
                P_k_array[k] += received_power(self.betas[m, k], psi_mk, h_mk)

        harvested = (
                self.tau_k *
                ((self.mu / (1 + np.exp(-self.a * (P_k_array - self.b)))) - (self.mu * self.Omega))
                / (1 - self.Omega)
        ) #CALCÚLO DA ENEGIA COLETADA PELO K-ÉSIMO DISPOSITIVO IoT

        # Métrica por passo (igual sua reward atual)
        step_loaded = (harvested >= self.E_min)
        reward = int(np.sum(step_loaded))  # soma por passo

        # Métrica de "únicos no episódio"
        self.ever_harvested |= step_loaded
        unique_loaded = int(np.sum(self.ever_harvested))

        done = False
        info = {
            "step_loaded": reward,  # quantos passaram E_min nesse passo
            "unique_loaded": unique_loaded  # quantos  já passaram E_min em algum passo do episódio
        }
        return (self.pb_positions[:, :2]).flatten().astype(np.float32), reward, done, False, info


#########################################################################################################
# --- Redes Neurais ---
class Actor(nn.Module):
    """
    Rede do ator (μ(s)): recebe o estado e produz a ação contínua normalizada em [-1, 1].
    No DDPG, o ator é ajustado para maximizar o Q(s, μ(s)) avaliado pelo crítico.
    """
    # Ação = vetor de deslocamentos(Δx, Δy) para cada Power Beacon, por passo de tempo, em
    # coordenadas normalizadas, que o ambiente converte em movimento real na área de 30×30 m
    def __init__(self, state_size, action_size, hidden1, hidden2):
        super(Actor, self).__init__()
        # Camadas totalmente conectadas (perceptron) que mapeiam estado -> ação
        self.fc1 = nn.Linear(state_size, hidden1)   # extração inicial de características do estado
        self.fc2 = nn.Linear(hidden1, hidden2)      # transformação não linear intermediária
        self.out = nn.Linear(hidden2, action_size)  # última camada gera a ação (ainda sem restrição)

    def forward(self, state):
        # Ativações ReLU para introduzir não linearidade
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh para limitar cada dimensão da ação em [-1, 1].
        # Fora daqui, o agente escala por "action_limit".
        return torch.tanh(self.out(x))


class Critic(nn.Module):
    """
    Rede do crítico (Q(s, a)): estima o valor-ação contínuo.
    Recebe o par (estado, ação) e retorna um escalar Q.
    """
    def __init__(self, state_size, action_size, hidden1, hidden2):
        super(Critic, self).__init__()
        # Entrada é a concatenação [estado, ação]
        self.fc1 = nn.Linear(state_size + action_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)  # saída escalar: valor Q(s,a)

    def forward(self, state, action):
        # Concatena ao longo da dimensão de features (B, S+A)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Sem ativação final: Q pode ser qualquer valor real
        return self.out(x)


# --- Replay Buffer ---
class ReplayBuffer:
    """
    Memória de repetição (off-policy): armazena transições (s, a, r, s', done)
    para amostragem aleatória em minibatches, quebrando correlação temporal.
    """
    def __init__(self, capacity):
        # deque com tamanho máximo: quando enche, descarta o mais antigo
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Armazena uma transição completa. Tipos/formatos são mantidos como vieram.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Amostragem i.i.d. uniforme do buffer
        batch = random.sample(self.buffer, batch_size)
        # Separa e empilha por coluna -> arrays numpy (batch, ·)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        # Permite checar rapidamente se já há amostras suficientes para treino
        return len(self.buffer)


# --- Agente DDPG ---
class DDPGAgent:
    """
    Implementação básica do DDPG:
      - Dois pares de redes (ator/Crítico e seus alvos "target")
      - Atualização suave (Polyak) dos alvos
      - Política determinística com ruído gaussiano para exploração
      - Aprendizado off-policy a partir do ReplayBuffer
    """
    def __init__(self, state_size, action_size, action_limit, actor_lr, critic_lr, gamma, tau, buffer_capacity,
                 batch_size, hidden1, hidden2, noise_std=0.2, noise_clip=0.4):
        # Dimensões e hiperparâmetros principais
        self.state_size = state_size
        self.action_size = action_size
        self.action_limit = action_limit  # escala física das ações (multiplica a saída tanh do ator)
        self.gamma = gamma                # fator de desconto
        self.tau = tau                    # taxa da atualização suave (0 < tau << 1)
        self.batch_size = batch_size

        # Redes online (treináveis)
        self.actor = Actor(state_size, action_size, hidden1, hidden2).to(device)
        self.critic = Critic(state_size, action_size, hidden1, hidden2).to(device)

        # Redes-alvo: começam idênticas e seguem as online por Polyak averaging
        self.actor_target = Actor(state_size, action_size, hidden1, hidden2).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_size, action_size, hidden1, hidden2).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Otimizadores independentes para cada rede
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Memória de experiências e ruído de exploração
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.noise_std = noise_std      # desvio padrão do ruído gaussiano
        self.noise_clip = noise_clip    # recorte do ruído para evitar saturação extrema

    def select_action(self, state, noise=True):
        """
        Gera uma ação determinística μ(s) e opcionalmente adiciona ruído gaussiano
        (exploração). A ação final é limitada a [-action_limit, action_limit].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        # modo eval desabilita camadas como dropout/batchnorm (não usadas aqui, mas é boa prática)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()  # [-1, 1]
        self.actor.train()

        if noise:
            # Ruído N(0, noise_std) por dimensão, recortado para estabilidade
            noise_sample = np.clip(
                np.random.randn(self.action_size) * self.noise_std,
                -self.noise_clip, self.noise_clip
            )
            action = action + noise_sample

        # Escala para o range físico permitido pelo ambiente
        return np.clip(action, -1, 1) * self.action_limit

    def update(self):
        """
        Uma iteração de atualização:
          1) Amostra um minibatch do replay
          2) Atualiza o crítico por MSE entre Q atual e alvo de Bellman
          3) Atualiza o ator maximizando Q(s, μ(s)) (via gradiente da política determinística)
          4) Atualiza suavemente as redes-alvo
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # espera até haver amostras suficientes

        # ---- Amostragem e tensores ----
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)  # (B, 1)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(device)

        # ---- Alvo (target) do crítico: y = r + γ * Q'(s', μ'(s')) * (1 - done) ----
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q  # (B,1)

        # ---- Perda do crítico: MSE(Q(s,a), y) ----
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Perda do ator: maximizar Q(s, μ(s))  ≡  minimizar -Q(s, μ(s)) ----
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Atualização suave das redes-alvo (Polyak averaging) ----
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)


    def soft_update(self, net, net_target):
        """
        Copia lentamente os pesos da rede online para a rede-alvo:
            θ' ← τ θ + (1 - τ) θ'
        Isso estabiliza o alvo no treinamento.
        """
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


# --- Média Móvel ---
def moving_average(values, window):
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


#########################################################################################################################
# --- Função Principal ---
def main():
    print("Iniciando simulação (cenários aleatórios por episódio)...")

    # -----------------------
    # Hiperparâmetros DDPG
    # -----------------------
    hyperparams = {
        'actor_lr': 1e-3,
        'critic_lr': 2e-3, #crítico aprende um pouco mais rápido
        'hidden1': 64,
        'hidden2': 128,
        'gamma': 0.95, #considera recompensas futuras, mas ainda com foco forte em horizonte relativamente curto.
        'tau': 0.001, #atualização suave (Polyak) muito lenta para estabilizar treinamento.
        'buffer_capacity': 256_109,
        'batch_size': 32, #tamanho padrão para minibatch
        'noise_std': 0.1, #ruído moderado
        'noise_clip': 0.2, #e recortado, garantindo exploração sem ações malucas
        'max_steps': 200 #cada episódio tem, no máximo, 200 passos (movimentos de PB)
    }

    # Onde salvar os gráficos
    diretorio = os.path.join(r"D:\INATEL\WET\plots\resultados primeiro dataset")
    os.makedirs(diretorio, exist_ok=True)

    # -----------------------
    # Parâmetros do ambiente
    # -----------------------
    bounds = (0, 30)   # (min, max) no plano x-y (m)
    # esses parâmetros são variáveis, dependem da quantidade de dispositivos
    K = 200            # número de dispositivos IoT
    M = 3              # número de PBs (drones)
    N = 4              # antenas por PB

    PT = 2.0           # potência Tx
    frequency = 915e6  # Hz
    alpha = 1.5        # expoente de perda

    # Receptor EH
    mu = 10.73e-3 #potência máxima coletada pelo dispositivo quando o circuito do dispositivo está saturado
    b = 0.2308
    a = 5.365
    Omega = 1 / (1 + np.exp(a * b)) #constante que garante uma resposta de entrada/saída zero para o circuíto

    # Tempo ativo (tau_k) dos IoTs
    tau_k = np.ones(K, dtype=float)

    # Janela da média móvel para o gráfico de convergência
    moving_avg_window = 50

    # -----------------------
    # Instancia o ambiente
    # (sem cenário fixo; o cenário é injetado a cada episódio via set_scenario)
    # -----------------------
    env = EnergyHarvestingEnv(
        tau_k,
        mu, a, b, Omega,
        K, M, N, PT, frequency, alpha,
        bounds
    )

    # -----------------------
    # Agente DDPG
    # -----------------------
    state_size = M * 2
    action_size = M * 2
    action_limit = 0.001

    agent = DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        action_limit=action_limit,
        actor_lr=hyperparams['actor_lr'],
        critic_lr=hyperparams['critic_lr'],
        gamma=hyperparams['gamma'],
        tau=hyperparams['tau'],
        buffer_capacity=hyperparams['buffer_capacity'],
        batch_size=hyperparams['batch_size'],
        hidden1=hyperparams['hidden1'],
        hidden2=hyperparams['hidden2'],
        noise_std=hyperparams['noise_std'],
        noise_clip=hyperparams['noise_clip']
    )

    # -----------------------
    # Treinamento
    # -----------------------
    total_training_episodes = 500

    rewards_per_episode = []
    unique_per_episode = []

    print(f"Iniciando fase de treinamento ({total_training_episodes} episódios, cenários aleatórios)...")

    for episode in range(total_training_episodes):

        iot_positions, pb_positions, chans, temperature_scalar, wind_scalar = generate_scenario(K, M, N, bounds)

        env.set_scenario(
            iot_positions,
            pb_positions,
            chans,
            temperature_scalar,
            wind_scalar
        )

        total_reward = 0
        state, _ = env.reset()
        last_info = {"unique_loaded": 0}

        for step in range(hyperparams['max_steps']):

            # No treinamento, mantém ruído para exploração
            action = agent.select_action(state, noise=True)

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            total_reward += reward

            agent.replay_buffer.push(
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state
            last_info = info

            agent.update()

            if done:
                break

        rewards_per_episode.append(total_reward)
        unique_per_episode.append(last_info["unique_loaded"])

        if (episode + 1) % 100 == 0:
            recent_mean = np.mean(rewards_per_episode[-100:])
            print(
                f"\tEpisódio {episode + 1}/{total_training_episodes} | "
                f"Reward total: {total_reward} | "
                f"Únicos: {last_info['unique_loaded']} | "
                f"Média últimos 100: {recent_mean:.2f}"
            )

    # -----------------------
    # Plot de convergência
    # -----------------------
    episodes_axis = np.arange(1, total_training_episodes + 1)

    ma_reward = moving_average(rewards_per_episode, moving_avg_window)

    ma_ep = np.arange(moving_avg_window, total_training_episodes + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(episodes_axis, rewards_per_episode, alpha=0.25, linestyle="--", label="Reward por episódio")

    plt.plot(ma_ep, ma_reward, linewidth=2, label=f"Média móvel reward ({moving_avg_window})")

    plt.xlabel("Episódio de treinamento")
    plt.ylabel("Valor")
    plt.title(
        "Convergência do DDPG ao longo da linha do tempo de treinamento\n"
        f"K={K}, M={M}, steps={hyperparams['max_steps']}, com ruído de exploração"
    )

    plt.ylim(0, max(max(rewards_per_episode), max(unique_per_episode)) * 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_plot_treino = os.path.join(
        diretorio,
        f'Convergencia_DDPG_linha_tempo_{M}PB_{K}_dispositivos.png'
    )

    plt.savefig(caminho_plot_treino, dpi=150, bbox_inches='tight')
    plt.show()

    # -----------------------
    # Avaliação (com e sem ruído)
    # -----------------------
    eval_episodes = 500

    eval_rewards_noise = []
    eval_rewards_no_noise = []

    eval_unique_noise = []
    eval_unique_no_noise = []

    print("\nIniciando avaliação com e sem ruído...")

    for episode in range(eval_episodes):

        # Gera o mesmo cenário para as duas avaliações
        iot_positions, pb_positions, chans, temperature_scalar, wind_scalar = generate_scenario(K, M, N, bounds)

        # =====================================================
        # Avaliação com ruído
        # =====================================================
        env.set_scenario(iot_positions, pb_positions, chans, temperature_scalar, wind_scalar)

        total_reward_noise = 0
        state, _ = env.reset()
        last_info_noise = {"unique_loaded": 0}

        for step in range(hyperparams['max_steps']):
            action = agent.select_action(state, noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward_noise += reward
            last_info_noise = info
            state = next_state

            if terminated or truncated:
                break

        eval_rewards_noise.append(total_reward_noise)
        eval_unique_noise.append(last_info_noise["unique_loaded"])

        # =====================================================
        # Avaliação sem ruído
        # =====================================================
        env.set_scenario(iot_positions, pb_positions, chans, temperature_scalar, wind_scalar)

        total_reward_no_noise = 0
        state, _ = env.reset()
        last_info_no_noise = {"unique_loaded": 0}

        for step in range(hyperparams['max_steps']):
            action = agent.select_action(state, noise=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward_no_noise += reward
            last_info_no_noise = info
            state = next_state

            if terminated or truncated:
                break

        eval_rewards_no_noise.append(total_reward_no_noise)
        eval_unique_no_noise.append(last_info_no_noise["unique_loaded"])

        if (episode + 1) % 100 == 0:
            print(
                f"\tEpisódio eval {episode + 1}/{eval_episodes} | "
                f"Reward com ruído: {total_reward_noise} | "
                f"Reward sem ruído: {total_reward_no_noise} | "
                f"Únicos com ruído: {last_info_noise['unique_loaded']} | "
                f"Únicos sem ruído: {last_info_no_noise['unique_loaded']}"
            )

    # -----------------------
    # Plot 1: Reward com e sem ruído
    # -----------------------
    eval_window = min(moving_avg_window, eval_episodes // 5)

    ep_eval = np.arange(1, eval_episodes + 1)
    ma_eval_ep = np.arange(eval_window, eval_episodes + 1)

    ma_reward_noise = moving_average(eval_rewards_noise, eval_window)
    ma_reward_no_noise = moving_average(eval_rewards_no_noise, eval_window)

    plt.figure()

    plt.plot(ep_eval, eval_rewards_noise, alpha=0.25, linestyle="--", label="Reward com ruído")

    plt.plot(ep_eval, eval_rewards_no_noise, alpha=0.25, label="Reward sem ruído")

    plt.plot(ma_eval_ep, ma_reward_noise, linewidth=2, linestyle="g--", label=f"Média móvel com ruído")

    plt.plot(ma_eval_ep, ma_reward_no_noise, linewidth=2, linestyle="b--", label=f"Média móvel sem ruído")

    plt.xlabel("Episódio de avaliação")
    plt.ylabel("Reward acumulada")
    plt.title(
        "Comparação da recompensa acumulada: com ruído vs sem ruído\n"
        f"K={K}, M={M}, steps={hyperparams['max_steps']}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    caminho_plot = os.path.join(diretorio, f'Comparacao_recompensa_com_sem_ruido_{M}PB_{K}_dispositivos.png')

    plt.savefig(caminho_plot, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

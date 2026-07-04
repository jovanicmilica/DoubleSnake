import random
import numpy as np
import torch
import torch.nn as nn

from agents.rl_agent.dqn_model import DQN
from agents.rl_agent.replay_buffer import ReplayBuffer
from config.config import (
    REPLAY_BUFFER_SIZE,
    BATCH_SIZE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
    LEARNING_RATE,
    GAMMA,
    TARGET_UPDATE_FREQ,
    HEURISTIC_SAMPLE_PROB_START,
    HEURISTIC_SAMPLE_DECAY_STEPS,
)


class DQNAgent:
    '''
    DQN agent koji koristi epsilon-greedy strategiju za odabir akcija i trenira se koristeci replay buffer.
    '''
    def __init__(self):
        self.model        = DQN() # glavna neuronska mreza koja se trenira
        self.target_model = DQN() # target mreza koja se koristi za stabilnije treniranje
        self.target_model.load_state_dict(self.model.state_dict())  # inicijalizacija target mreze sa istim tezinskim parametrima kao glavna mreza
        self.target_model.eval()  # target se nikad ne trenira direktno

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE) # sluzi za optimizaciju tezinskih parametara glavne mreze
        self.loss_fn   = nn.MSELoss() # koliko model odstupa od cilja 

        self.buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)  # self-play iskustva 

        # Poseban buffer sa heuristic vs heuristic iskustvima 
        # Ne mesa se sa self-play bufferom - train_step bira iz kog buffera uzima
        # batch na osnovu heuristic_sample_prob, koja linearno opada do 0.
        self.heuristic_buffer = None
        self.heuristic_sample_prob = HEURISTIC_SAMPLE_PROB_START
        self._heuristic_prob_decay = HEURISTIC_SAMPLE_PROB_START / HEURISTIC_SAMPLE_DECAY_STEPS

        self.epsilon = EPSILON_START  # vjerovatnoca random poteza
        self.steps   = 0             # ukupan broj koraka, za TARGET_UPDATE_FREQ

    def set_heuristic_buffer(self, buffer):
        """Postavlja odvojeni buffer sa heuristic demonstracijama za train_step."""
        self.heuristic_buffer = buffer


    def select_action(self, obs):
        """
        Epsilon-greedy: s verovatnocom epsilon biramo random akciju
        (istrazivanje), inace biramo akciju s najvecom Q-vrijednosti
        (iskoristavanje).

        obs[0:4] su danger flagovi (gore, dole, levo, desno) - isti redosled
        kao akcije. Kod random poteza izbegavamo akcije koje odmah ubijaju
        zmiju (zid/telo), jer takvi potezi samo skracuju epizodu bez ikakve
        korisne informacije za trening dok je epsilon jos visok.
        """
        if random.random() < self.epsilon: # ako je random broj manji od epsilon, biramo random akciju
            safe_actions = [a for a in range(4) if obs[a] == 0.0]
            if safe_actions:
                return random.choice(safe_actions)
            return random.randint(0, 3)  # nema sigurnog poteza, smrt je neizbezna

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # pretvaramo u tensor i dodajemo batch dimenziju (1, 19)
        with torch.no_grad():
            q_values = self.model(obs_tensor)  
        return q_values.argmax(dim=1).item()   # indeks najvece Q-vrednosti


    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)


    def train_step(self):
        """
        Uzima jedan batch i azurira tezine modela.
        Na pocetku treninga (dok je heuristic_sample_prob > 0) batch se s tom
        vjerovatnocom uzima iz heuristic buffera umjesto iz self-play buffera -
        sto je vise koraka odigrano, ta vjerovatnoca je sve manja (do 0).
        Ne radi nista ako odabrani buffer jos nema dovoljno iskustava.
        """
        use_heuristic = (
            self.heuristic_buffer is not None
            and len(self.heuristic_buffer) >= BATCH_SIZE
            and random.random() < self.heuristic_sample_prob
        )
        source = self.heuristic_buffer if use_heuristic else self.buffer

        if len(source) < BATCH_SIZE:  # premalo iskustava, preskoci
            return None

        obs, actions, rewards, next_obs, dones = source.sample(BATCH_SIZE)

        # pretvori numpy arraye u PyTorch tensore
        obs_t      = torch.tensor(obs,     dtype=torch.float32)  # (64, 19)
        actions_t  = torch.tensor(actions, dtype=torch.long)     # (64,)
        rewards_t  = torch.tensor(rewards, dtype=torch.float32)  # (64,)
        next_obs_t = torch.tensor(next_obs,dtype=torch.float32)  # (64, 19)
        dones_t    = torch.tensor(dones,   dtype=torch.float32)  # (64,)

        # uzima Q-vrednosti za sve akcije, a zatim odabire one koje odgovaraju odabranim akcijama u batchu
        current_q = self.model(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: online mreza bira akciju, target mreza evaluira njenu vrednost
        # (smanjuje overestimation bias standardnog DQN-a)
        with torch.no_grad():
            next_actions = self.model(next_obs_t).argmax(dim=1)
            next_q = self.target_model(next_obs_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_t + GAMMA * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # azurianje epsilon-a (istrazivanje se smanjuje s vremenom)
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # linearno smanjivanje oslanjanja na heuristic buffer do 0
        if self.heuristic_sample_prob > 0:
            self.heuristic_sample_prob = max(0.0, self.heuristic_sample_prob - self._heuristic_prob_decay)

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def save(self, path, episode, scores):
        torch.save({
            "episode":         episode,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon":                self.epsilon,
            "steps":                  self.steps,
            "scores":                 scores,
            "heuristic_sample_prob":  self.heuristic_sample_prob,
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model_state"])
        self.target_model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epsilon = ckpt["epsilon"]
        self.steps   = ckpt["steps"]
        # stariji checkpointi nemaju ovo polje - u tom slucaju vise ne oslanjamo se na heuristic buffer
        self.heuristic_sample_prob = ckpt.get("heuristic_sample_prob", 0.0)
        return ckpt["episode"], ckpt["scores"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import random
from collections import deque
from services.qgen import environment

cont = """
Alexander Graham Bell was born in Edinburgh, Scotland on March 3, 1847. When he was only eleven years old, he invented a machine that could clean wheat. Graham studied anatomy and physiology at the University of London, but moved with his family to Quebec, Canada in 1870.
Bell soon moved to Boston, Massachusetts. In 1871, he began working with deaf people and published the system of Visible Hearing that was developed by his father. Visible hearing illustrated how the tongue, lips, and throat are used to produce vocal sounds. In 1872, Bell founded a school for the deaf which soon became part of Boston University. Alexander Graham Bell is best known for his invention of the telephone. While trying to discover the secret of transmitting multiple messages on a single wire, Bell heard the sound of a plucked string along some of the electrical wire. One of Bell's assistants, Thomas A. Watson, was trying to reactivate a telephone transmitter. After hearing the sound, Bell believed he could send the sound of a human voice over the wire. After receiving a patent on March 7, 1876 for transmitting sound along a single wire, he successfully transmitted human speech on March 10th. Bell's telephone patent was one of the most valuable patents ever issued. He started the Bell Telephone Company in 1877. Bell went on to invent a precursor to the modern day air conditioner, and a device, called a "photophone", that enabled sound to be transmitted on a beam of light. Today's fiber optic and laser communication systems are based on Bell's photophone research. In 1898, Alexander Graham Bell and his son-in-law took over the National Geographic Society and built it into one of the most recognized magazines in the world. Bell also helped found Science Magazine, one of the most respected research journals in the world.
Alexander Graham Bell died August 2, 1922. On the day of his burial, in honor of Bell, all telephone services in the United States were stopped for one minute.
"""


def lambda_select_actions(last_hidden_state: torch.Tensor, actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    """
    Masked mean pooling per action range.

    Args:
        last_hidden_state: (batch, seq_len, 768)
        actions: (batch, num_actions, 2)  where [..., 0] = start, [..., 1] = end (inclusive)
        num_actions: int

    Returns:
        select_actions: (batch, num_actions, 768)
    """
    b = last_hidden_state.size(0)
    l = last_hidden_state.size(1)

    # positions: (1, 1, l)
    positions = torch.arange(l, device=last_hidden_state.device).reshape(1, 1, l)
    # tile to (b, num_actions, l)
    positions = positions.expand(b, num_actions, l)

    # a_min, a_max: (b, num_actions, 1)
    a_min = actions[:, :, 0].unsqueeze(-1)  # (b, num_actions, 1)
    a_max = actions[:, :, 1].unsqueeze(-1)  # (b, num_actions, 1)

    # mask_actions: (b, num_actions, l)
    mask_actions = (positions >= a_min) & (positions <= a_max)

    # reshaped_mask: (b * num_actions, l)
    reshaped_mask = mask_actions.reshape(-1, l)

    # last_hidden_state: (b, l, 768) -> tile to (b, num_actions, l, 768)
    reshaped_lhs = last_hidden_state.unsqueeze(1).expand(b, num_actions, l, 768)
    # (b * num_actions, l, 768)
    reshaped_tiled_lhs = reshaped_lhs.reshape(-1, l, 768)

    # Masked mean pooling: sum over masked positions, divide by count
    # reshaped_mask: (b * num_actions, l) -> (b * num_actions, l, 1)
    mask_float = reshaped_mask.unsqueeze(-1).float()
    masked_sum = (reshaped_tiled_lhs * mask_float).sum(dim=1)   # (b * num_actions, 768)
    mask_count = mask_float.sum(dim=1).clamp(min=1e-9)           # (b * num_actions, 1)
    pool = masked_sum / mask_count                                # (b * num_actions, 768)

    # (b, num_actions, 768)
    select_actions = pool.reshape(b, num_actions, 768)
    return select_actions


class DQNModel(nn.Module):
    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions

        self.bert = BertModel.from_pretrained(environment.model_name)

        self.pos_embedding = nn.Embedding(len(environment.pos_all_tags), 8)

        # Shared projection layer (768 -> 32)
        self.projection_layer = nn.Linear(768, 32)

        # Can-stop branch (move: num_actions-1 actions)
        # Input: 32 (projection) + 1 (subtree) + 8 (pos_emb) = 41
        self.move_dense2 = nn.Linear(32 + 1 + 8, 16)
        self.move_output = nn.Linear(16, 1)

        # Can-stop branch (stay: 1 action)
        self.stay_dense2 = nn.Linear(32 + 1 + 8, 16)
        self.stay_output = nn.Linear(16, 1)

        # Cannot-stop branch (all num_actions actions)
        # Input: 32 (projection) + 1 (subtree) + 8 (pos_emb) = 41
        self.dense2 = nn.Linear(32 + 1 + 8, 16)
        self.dense_output = nn.Linear(16, 1)

    def forward(
        self,
        input_ids: torch.Tensor,       # (b, seq_len)
        attention_mask: torch.Tensor,  # (b, seq_len)
        actions: torch.Tensor,         # (b, num_actions, 2)
        pos: torch.Tensor,             # (b, num_actions, 1)
        subtree: torch.Tensor,         # (b, num_actions, 1)
        allow_stop: torch.Tensor,      # (b,) or (b, 1), bool
    ) -> torch.Tensor:
        num_actions = self.num_actions

        # BERT encoding
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_out.last_hidden_state  # (b, seq_len, 768)

        # Masked mean pooling per action range -> (b, num_actions, 768)
        select_actions = lambda_select_actions(last_hidden_state, actions, num_actions)

        # Split stay (1) and move (num_actions-1)
        stay_actions = select_actions[:, :1, :]           # (b, 1, 768)
        move_actions = select_actions[:, 1:, :]           # (b, num_actions-1, 768)

        stay_subtree = subtree[:, :1, :]                  # (b, 1, 1)
        move_subtree = subtree[:, 1:, :]                  # (b, num_actions-1, 1)

        # POS embeddings: pos is (b, num_actions, 1)
        pos_emb = self.pos_embedding(pos.squeeze(-1))     # (b, num_actions, 8)
        stay_pos_emb = pos_emb[:, :1, :]                  # (b, 1, 8)
        move_pos_emb = pos_emb[:, 1:, :]                  # (b, num_actions-1, 8)

        # --- Can-stop branch ---
        move_dense1 = F.relu(self.projection_layer(move_actions))   # (b, num_actions-1, 32)
        move_in2 = torch.cat([move_dense1, move_subtree, move_pos_emb], dim=2)  # (b, num_actions-1, 41)
        move_dense2 = F.relu(self.move_dense2(move_in2))            # (b, num_actions-1, 16)
        move_out = self.move_output(move_dense2)                     # (b, num_actions-1, 1)

        stay_dense1 = F.relu(self.projection_layer(stay_actions))   # (b, 1, 32)
        stay_in2 = torch.cat([stay_dense1, stay_subtree, stay_pos_emb], dim=2)  # (b, 1, 41)
        stay_dense2 = F.relu(self.stay_dense2(stay_in2))            # (b, 1, 16)
        stay_out = self.stay_output(stay_dense2)                     # (b, 1, 1)

        # Concatenate stay + move and flatten -> (b, num_actions)
        move_stay = torch.cat([stay_out, move_out], dim=1)           # (b, num_actions, 1)
        move_stay_flat = move_stay.squeeze(-1)                        # (b, num_actions)

        # --- Cannot-stop branch ---
        dense1 = F.relu(self.projection_layer(select_actions))       # (b, num_actions, 32)
        dense_in2 = torch.cat([dense1, subtree, pos_emb], dim=2)     # (b, num_actions, 41)
        dense2 = F.relu(self.dense2(dense_in2))                      # (b, num_actions, 16)
        dense_out = self.dense_output(dense2)                         # (b, num_actions, 1)
        flat_out = dense_out.squeeze(-1)                              # (b, num_actions)

        # allow_stop branching: (b,) bool
        if allow_stop.dim() > 1:
            allow_stop = allow_stop.squeeze(-1)                       # (b,)
        # Expand to (b, num_actions) for torch.where
        allow_stop_expanded = allow_stop.unsqueeze(-1).expand_as(move_stay_flat)

        output = torch.where(allow_stop_expanded, move_stay_flat, flat_out)  # (b, num_actions)
        return output


class DQAgent:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.model = DQNModel(num_actions=environment.MAX_ACTIONS)
        self.target_model = DQNModel(num_actions=environment.MAX_ACTIONS)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, eps=1e-8)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_distribution(self, state: environment.Node):
        all_nodes = state.subtree
        if len(state.children) == 0:
            return [1]
        if state.allow_stop == False:
            return [1.0 / len(state.children)] * len(state.children)
        subtree_children = [child.subtree for child in state.children]
        distribution = [1.0 * sc / all_nodes for sc in subtree_children]
        distribution = [1.0 / all_nodes] + distribution
        return distribution

    def act(self, state: environment.Node, index=-1):
        if state.allow_stop == False and index != -1:
            return index, None, None
        allowed_actions = len([x for x in state.actions if x is not None])
        a = [1] * len(state.token_ids)

        input_ids = torch.tensor([state.token_ids], dtype=torch.long)
        attention_mask = torch.tensor([a], dtype=torch.long)
        actions = torch.tensor([state.reshape_actions()], dtype=torch.long)
        pos = torch.tensor([state.reshape_pos()], dtype=torch.long)
        subtree = torch.tensor([state.reshape_subtree()], dtype=torch.float)
        allow_stop = torch.tensor([state.allow_stop], dtype=torch.bool)

        self.model.eval()
        with torch.no_grad():
            act_values = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                actions=actions,
                pos=pos,
                subtree=subtree,
                allow_stop=allow_stop,
            )  # (1, num_actions)

        act_values_np = act_values[0]  # (num_actions,)
        distributions = torch.softmax(act_values_np[:allowed_actions], dim=0)
        choose = random.choices(list(range(0, allowed_actions)), weights=distributions.tolist(), k=1)[0]
        return choose, distributions, act_values_np[choose]

    def load(self, name):
        state_dict = torch.load(name, map_location="cpu")
        self.model.load_state_dict(state_dict)

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def get_largest_indexes(lst, choices):
    indexes = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return indexes[:choices]

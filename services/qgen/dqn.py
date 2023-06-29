import tensorflow as tf
from transformers import TFBertModel
import random
from collections import deque
from services.qgen import environment
# import warnings
# warnings.filterwarnings('ignore')

cont = """
Alexander Graham Bell was born in Edinburgh, Scotland on March 3, 1847. When he was only eleven years old, he invented a machine that could clean wheat. Graham studied anatomy and physiology at the University of London, but moved with his family to Quebec, Canada in 1870.
Bell soon moved to Boston, Massachusetts. In 1871, he began working with deaf people and published the system of Visible Hearing that was developed by his father. Visible hearing illustrated how the tongue, lips, and throat are used to produce vocal sounds. In 1872, Bell founded a school for the deaf which soon became part of Boston University. Alexander Graham Bell is best known for his invention of the telephone. While trying to discover the secret of transmitting multiple messages on a single wire, Bell heard the sound of a plucked string along some of the electrical wire. One of Bell's assistants, Thomas A. Watson, was trying to reactivate a telephone transmitter. After hearing the sound, Bell believed he could send the sound of a human voice over the wire. After receiving a patent on March 7, 1876 for transmitting sound along a single wire, he successfully transmitted human speech on March 10th. Bell's telephone patent was one of the most valuable patents ever issued. He started the Bell Telephone Company in 1877. Bell went on to invent a precursor to the modern day air conditioner, and a device, called a "photophone", that enabled sound to be transmitted on a beam of light. Today's fiber optic and laser communication systems are based on Bell's photophone research. In 1898, Alexander Graham Bell and his son-in-law took over the National Geographic Society and built it into one of the most recognized magazines in the world. Bell also helped found Science Magazine, one of the most respected research journals in the world.
Alexander Graham Bell died August 2, 1922. On the day of his burial, in honor of Bell, all telephone services in the United States were stopped for one minute.
"""

def lambda_select_actions(last_hidden_state: tf.Tensor, actions: tf.Tensor, num_actions: int) -> tf.Tensor:
    l = tf.shape(last_hidden_state)[1]  # sequence length (number of tokens)
    b = tf.shape(last_hidden_state)[0]  # batch size

    # Range tensor for mask
    range = tf.reshape(tf.range(l), [1, 1, l])
    range = tf.tile(range, [b, num_actions, 1])

    # Action ranges
    a_min = tf.expand_dims(actions[:, :, 0], axis=-1)
    a_max = tf.expand_dims(actions[:, :, 1], axis=-1)

    # Create mask
    y_ge_min = tf.greater_equal(range, a_min)
    y_le_max = tf.less_equal(range, a_max)
    mask_actions = tf.logical_and(y_ge_min, y_le_max)

    # Reshape mask for average pooling
    reshaped_mask = tf.reshape(mask_actions, [-1, l])
    reshaped_lhs = tf.reshape(last_hidden_state, [b, 1, l, 768])
    tiled_lhs = tf.tile(reshaped_lhs, [1, num_actions, 1, 1])
    reshaped_tiled_lhs = tf.reshape(tiled_lhs, [-1, l, 768])
    pool = tf.keras.layers.GlobalAveragePooling1D()(reshaped_tiled_lhs, reshaped_mask)

    select_actions = tf.reshape(pool, [-1, num_actions, 768])
    return select_actions


def create_model(num_actions: int = 4) -> tf.keras.Model:
    bert_input1 = tf.keras.Input(
        shape=(None,), dtype="int32", name="input_ids")
    bert_input2 = tf.keras.Input(
        shape=(None,), dtype="int32", name="attention_mask")
    actions = tf.keras.Input(shape=(num_actions, 2),
                             dtype="int32", name="actions")
    subtree = tf.keras.Input(shape=(num_actions, 1),
                             dtype="float32", name="subtree")
    pos = tf.keras.Input(shape=(num_actions, 1), dtype="int32", name="pos")
    allow_stop = tf.keras.Input(shape=(1), dtype="bool", name="allow_stop")
    bert_model = TFBertModel.from_pretrained(environment.model_name)

    bert_model.trainable = True
    base_outputs = bert_model.bert(
        {"input_ids": bert_input1, "attention_mask": bert_input2})
    last_hidden_state = base_outputs.last_hidden_state

    select_actions = lambda_select_actions(
        last_hidden_state, actions, num_actions)
    stay_actions, move_actions = tf.split(
        select_actions, [1, num_actions-1], axis=1)

    stay_subtree, move_subtree = tf.split(subtree, [1, num_actions-1], axis=1)

    pos_embedding = tf.keras.layers.Embedding(
        len(environment.pos_all_tags), 8, name="pos_embedding")(pos)
    reshaped_embedding = tf.reshape(
        pos_embedding, shape=(-1, num_actions, 8))  # flattening
    stay_pos_embedding, move_pos_embedding = tf.split(
        reshaped_embedding, [1, num_actions-1], axis=1)

    projection_layer = tf.keras.layers.Dense(
        32, activation='relu', name='projection_layer')

    # Can stop branch
    move_dense1 = projection_layer(move_actions)
    move_dense2 = tf.keras.layers.Dense(16, activation='relu', name='move_dense2')(
        tf.concat([move_dense1, move_subtree, move_pos_embedding], axis=2))
    move_output = tf.keras.layers.Dense(1, name='move_output')(move_dense2)

    stay_dense1 = projection_layer(stay_actions)
    stay_dense2 = tf.keras.layers.Dense(16, activation='relu', name='stay_dense2')(
        tf.concat([stay_dense1, stay_subtree, stay_pos_embedding], axis=2))
    stay_output = tf.keras.layers.Dense(1, name='stay_output')(stay_dense2)

    move_stay_flattened_output = tf.keras.layers.Flatten(
        name='move_stay_flattened_output')(tf.concat([stay_output, move_output], axis=1))

    # Cannot stop branch
    dense1 = projection_layer(select_actions)
    dense2 = tf.keras.layers.Dense(16, activation='relu', name='dense2')(
        tf.concat([dense1, subtree, reshaped_embedding], axis=2))
    dense_output = tf.keras.layers.Dense(1, name='dense_output')(dense2)

    flattened_output = tf.keras.layers.Flatten(
        name='flattened_output')(dense_output)

    # Output
    output = tf.where(allow_stop, move_stay_flattened_output, flattened_output)

    model = tf.keras.Model(
        inputs={
            "input_ids": bert_input1,
            "attention_mask": bert_input2,
            "actions": actions,
            "pos": pos,
            "subtree": subtree,
            "allow_stop": allow_stop,
        },
        outputs=output,
    )
    return model

class DQAgent:
    def __init__(self):
        self.memory = deque(maxlen=1000)
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=1e-05, epsilon=1e-08)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.model = create_model(num_actions=environment.MAX_ACTIONS)
        self.target_model = create_model(num_actions=environment.MAX_ACTIONS)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
        act_values = self.model.predict(
            {
                "input_ids": tf.constant([state.token_ids]),
                "attention_mask": tf.constant([a]),
                "actions": tf.constant([state.reshape_actions()]),
                "pos": tf.constant([state.reshape_pos()], dtype=tf.int32),
                "subtree": tf.constant([state.reshape_subtree()], dtype=tf.float32),
                "allow_stop": tf.constant([state.allow_stop], dtype=tf.bool),
            },
            verbose=0,
        )
        
        distributions = tf.nn.softmax(act_values[0][:allowed_actions])
        choose = random.choices(list(range(0, allowed_actions)), weights=distributions, k=1)[0]
        return choose, distributions, act_values[0][choose]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def get_largest_indexes(lst, choices):
    indexes = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)
    return indexes[:choices]

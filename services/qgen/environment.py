import tensorflow as tf

from transformers import BertTokenizerFast
import spacy
import benepar
import string
import copy
from nltk import sent_tokenize
import nltk
import json
nltk.download('punkt')
benepar.download('benepar_en3')
model_name = 'bert-base-cased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
pos_all_tags = list(spacy.glossary.GLOSSARY.keys()) + ['ZZZ_SENTENCE', 'S', 'PRN', 'SQ', 'LST', 'SBARQ', 'ZZZ', 'NML', 'QP', 'WHNP', 'CONJP', 'SINV', 'WHADVP', 'WHPP', 'FRAG', 'UCP', 'WHADJP', 'RRC']
pos_all_tags.sort()

# fd = open('dict_correct_aq_loss.json', 'r')
# reward_dict = json.load(fd)


MAX_ACTIONS = 30

N_TOKEN = "[unused1]"
N_TOKEN_ID = tokenizer.convert_tokens_to_ids(N_TOKEN)
N_END_TOKEN = "[unused2]"
N_END_TOKEN_ID = tokenizer.convert_tokens_to_ids(N_END_TOKEN)
C_TOKEN = "[unused3]"
C_TOKEN_ID = tokenizer.convert_tokens_to_ids(C_TOKEN)
C_END_TOKEN = "[unused4]"
C_END_TOKEN_ID = tokenizer.convert_tokens_to_ids(C_END_TOKEN)

CLS_TOKEN = "[CLS]"
CLS_TOKEN_ID = tokenizer.convert_tokens_to_ids(CLS_TOKEN)

SEP_TOKEN = "[SEP]"
SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

PAD_TOKEN = "[PAD]"
PAD_TOKEN_ID = tokenizer.convert_tokens_to_ids(PAD_TOKEN)

print(f"{N_TOKEN} - {N_TOKEN_ID}")
print(f"{N_END_TOKEN} - {N_END_TOKEN_ID}")
print(f"{C_TOKEN} - {C_TOKEN_ID}")
print(f"{C_END_TOKEN} - {C_END_TOKEN_ID}")
print(f"{CLS_TOKEN} - {CLS_TOKEN_ID}")
print(f"{SEP_TOKEN} - {SEP_TOKEN_ID}")
print(f"{PAD_TOKEN} - {PAD_TOKEN_ID}")

class Node:
  def get_numeric_state_actions(self, token_ids, sequence, children, parent_start_idx, parent_end_idx): 
    batch = tokenizer([sequence] + children, return_tensors="np", padding='max_length', max_length=512)['input_ids'].tolist()
    
    for i in range(len(batch)):
      # Remove PAD, CLS and SEP
      batch[i] = [x for x in batch[i] if x != tokenizer.pad_token_id]
      batch[i] = batch[i][1:len(batch[i]) - 1]
    state_ids = batch[0]
    children_ids = batch[1:]
    state_indexes = None
    for i in range(len(token_ids)):
      if i >= parent_start_idx and i + len(state_ids) - 1 <= parent_end_idx and token_ids[i:i+len(state_ids)] == state_ids:
        state_indexes = [i, i + len(state_ids) - 1]
        break
    if state_indexes == None:
      return (None, None)
    
    children_indexes_list = []
    for child in children_ids:
      child_indexes = None
      for i in range(len(token_ids)):
        if i >= state_indexes[0] and i + len(child) - 1 <= state_indexes[1] and token_ids[i:i+len(child)] == child:
          child_indexes = [i, i + len(child) - 1]
          break
      if child_indexes == None:
        children_indexes_list = None
        break
      children_indexes_list.append(child_indexes)
    
    return (state_indexes, children_indexes_list)
  
  def reshape_actions(self):
    first = [x[0] if x != None else 0 for x in self.actions]
    second = [x[1] if x != None else 0 for x in self.actions]

    return [[x[0], x[1]] if x != None else [0, 0] for x in self.actions]
    #return [first, second]
  def reshape_pos(self):
    if self.allow_stop == True:
      pos_list = [self.pos] + self.children_pos
    else:
      pos_list = self.children_pos
    while len(pos_list) < MAX_ACTIONS:
      pos_list.append(0)
    pos_list = [[x] for x in pos_list]
    return pos_list
  
  def reshape_subtree(self):
    if self.allow_stop == True:
      subtree_list = [self.subtree] + [child.subtree for child in self.children]
    else:
      subtree_list = [child.subtree for child in self.children]
    while len(subtree_list) < MAX_ACTIONS:
      subtree_list.append(-1)
    subtree_list = [[x] for x in subtree_list]
    return subtree_list

  def __str__(self) -> str:
    return f"TOK_IDS: {self.token_ids} \n ACTIONS: {self.actions} \n CHILDREN: {len(self.children)} \n TEXT: {self.text} \n SUBTREE: {self.subtree}"

  def __init__(self, token_ids, sequence, children, parent_start_idx, parent_end_idx, allow_stop, pos, start_char_idx, end_char_idx): # token_ids pentru toata propozitia, nu se schimba; state: (start, end); actions: lista de (start, end)
    state, actions = self.get_numeric_state_actions(token_ids[1:len(token_ids) - 1], sequence, children, parent_start_idx, parent_end_idx)
    self.allow_stop = allow_stop
    self.text = sequence
    self.pos = pos
    self.parent = None
    self.internal_id = -1
    self.reward = -1000.0
    self.gen_question = ""
    self.token_ids = []
    self.token_ids = copy.deepcopy(token_ids)
    self.token_ids.insert(state[0] + 1, N_TOKEN_ID)
    self.token_ids.insert(state[1] + 3, N_END_TOKEN_ID)
    self.subtree = 0
    self.sentence_index = -1
    children_tokens = 0
    self.start_char_idx = start_char_idx
    self.end_char_idx = end_char_idx
    self.actions = []
    if actions != None:
      for action in actions:
        self.token_ids.insert(action[0] + 1 + 1 + children_tokens * 2, C_TOKEN_ID)
        self.token_ids.insert(action[1] + 1 + 1 + children_tokens * 2 + 1 + 1, C_END_TOKEN_ID)
        children_tokens += 1

      start_act = -1
      for i in range(len(self.token_ids)):
        if self.token_ids[i] == C_TOKEN_ID:
          start_act = i
        if self.token_ids[i] == C_END_TOKEN_ID:
          self.actions.append([start_act, i])

    start_act = -1
    for i in range(len(self.token_ids)):
      if self.token_ids[i] == N_TOKEN_ID:
        start_act = i
      if self.token_ids[i] == N_END_TOKEN_ID:
        self.actions.insert(0, [start_act, i])
        break
    while len(self.actions) < MAX_ACTIONS:
      self.actions.append(None)
    self.children = []
    self.children_pos = []

class MyEnvironment:
  def __init__(self, context):
    self.MAX_LEN_SENTENCE = 1000
    self.context = context
    self.last_id = 0
    self.sentences = None

    self.root = None
    self.create_sentences(context)
    self.compute_subtree(self.root)
    self.initial_state = self.root
    self.current_state = self.initial_state

  def compute_subtree(self, node: Node):
    node.internal_id = self.last_id
    self.last_id += 1
    if len(node.children) == 0:
      node.subtree = 1
      return
    sum = 0
    for child in node.children:
      self.compute_subtree(child)
      sum += child.subtree
    node.subtree = sum + 1

  def export_nodes(self, external_id, node: Node, root: Node):
    result = [(f"{external_id}_{node.internal_id}", root.text, node.text)]
    for child in node.children:
      result += self.export_nodes(external_id, child, root)
    return result
  
  def assign_reward(self, external_id, node: Node):
    reward_entry = reward_dict[f"{external_id}_{node.internal_id}"]
    if reward_entry[1] == node.text:
      node.reward = reward_entry[3] #+ reward_entry[4]
      node.gen_question = reward_entry[2]
    else:
      print("AOLEUUUUUUUUUU!!!!!")
      print(reward_entry[1])
      print(node.text)
    for child in node.children:
      self.assign_reward(external_id, child)

  def export_states(self, node: Node):
    result = [node]
    for child in node.children:
      result += self.export_states(child)
    return result
  
  def create_sentences(self, context):
    sentences = sent_tokenize(context)
    self.sentences = sentences
    context_token_ids = tokenizer(context, return_tensors="np")['input_ids'].tolist()[0]
    self.root = Node(context_token_ids, context, sentences, 0, len(context_token_ids), False, pos_all_tags.index('ZZZ_SENTENCE'), 0, len(context))

    self.root.actions.pop(0)
    self.root.actions.append(None)

    index = -1
    for sentence in sentences:
      index += 1
      all_token_ids = tokenizer(sentence, return_tensors="np")['input_ids'].tolist()[0]
      doc = list(nlp(sentence).sents)[0]
      self.create_graph(doc, None, all_token_ids, index)

  def just_punkt(self, seq):
    if not seq or (len(str(seq)) == 1 and str(seq)[0] in string.punctuation):
      return True
    return False

  def create_graph(self, seq, prev_node, all_token_ids, index):
    if seq and not self.just_punkt(seq):
      children_str_list = [str(child) for child in seq._.children if not self.just_punkt(child)]
      if prev_node == None:
        (s_idx, e_idx) = (0, self.MAX_LEN_SENTENCE)
      else:
        list_with_just_N = [x for x in prev_node.token_ids[1:len(prev_node.token_ids) - 1] if x != C_TOKEN_ID and x != C_END_TOKEN_ID]
        (s_idx, e_idx) = (list_with_just_N.index(N_TOKEN_ID), list_with_just_N.index(N_END_TOKEN_ID) - 1)

      if len(seq._.labels) == 0:
        pos = pos_all_tags.index('ZZZ')
      else:
        pos = pos_all_tags.index(seq._.labels[0])

      if prev_node != None and prev_node.actions[1] == None:
        return
      
      curr_node = Node(all_token_ids, str(seq), children_str_list, s_idx, e_idx, True, pos, seq.start_char, seq.end_char)

      if prev_node == None:
        curr_node.sentence_index = index
        self.root.children.append(curr_node)
        self.root.children_pos.append(pos)
      else:
        curr_node.sentence_index = prev_node.sentence_index
        prev_node.children.append(curr_node)
        prev_node.children_pos.append(pos)

      for child in seq._.children:
        if not self.just_punkt(child):
          self.create_graph(child, curr_node, all_token_ids, index)
  
  def step_next_state(self, action_idx):
    if self.current_state.allow_stop:
      if action_idx == 0:
        return True # done
      self.current_state = self.current_state.children[action_idx - 1]
      return False
    else:
      self.current_state = self.current_state.children[action_idx] # Sentence children-actions start with 0
      return False
  
  def assign_parents(self, state: Node):
    for ch in state.children:
      ch.parent = state
      self.assign_parents(ch)
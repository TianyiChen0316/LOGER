import torch, torch.nn.functional as F
import dgl
import random
import typing
import math
from collections import Iterable

from core import database, Sql, Plan
from lib.timer import timer

from .explorer import HalfTimeExplorer
from .memory import Memory, BestCache
from .step1 import Step1, PredictTail, UseGeneratedPredict
from .step2 import Step2

class _dummy:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DeepQNet:
    def __init__(self, device=torch.device('cpu'), half=200, out_dim=4, num_table_layers=1,
                 use_value_predict=True, restricted_operator=True, reward_weighting=0.1, _dummy=False, log_cap=1):
        self.model_step1 = Step1(num_table_layers=num_table_layers)
        self.model_step2 = Step2(squeeze=True)
        self.OUT_DIM = out_dim
        self.restricted_operator = restricted_operator
        if not restricted_operator:
            self.OUT_DIM = 3

        self.gt_cap = log_cap

        self._tail_out_dim = 4
        self.model_tail = PredictTail(
            out_dim=self._tail_out_dim,
            aggr='sum',
        )

        self.optimizer_reset()

        self.use_value_predict = use_value_predict
        self.memory = Memory(size=database.config.memory_size)
        self.best_values = BestCache()

        self.explorer = HalfTimeExplorer(start=0.8, end=0.2, half_steps=half)

        self.train_resample = 0.25

        self.use_baseline_reward = False
        self.baseline_dict = {}
        self.baseline_reward_explorer = HalfTimeExplorer(start=0.2, end=0, half_steps=800)

        self.reward_weighting = reward_weighting

        self.device = device

    def optimizer_reset(self):
        self.optim = torch.optim.Adam([
            *self.model_step1.parameters(),
            *self.model_step2.parameters(),
            *self.model_tail.parameters(),
        ], lr=3e-4)
        self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, [
            *range(50, 100),
            *range(100, 200, 2),
        ], (0.1) ** (1 / 50))

    def reset(self):
        self.explorer.reset()
        self.memory.clear()
        self.best_values.clear()

        self.baseline_dict.clear()

    def train_mode(self):
        self.model_step1.train()
        self.model_step2.train()
        self.model_tail.train()

    def eval_mode(self):
        self.model_step1.eval()
        self.model_step2.eval()
        self.model_tail.eval()

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.to(value)

    def to(self, device):
        self.__device = device
        self.model_step1.to(device)
        self.model_step2.to(device)
        self.model_tail.to(device)
        return self

    def model_recover(self, state_dict, strict=True):
        if 'step1' in state_dict:
            self.model_step1.load_state_dict(state_dict['step1'], strict=strict)
        if 'step2' in state_dict:
            self.model_step2.load_state_dict(state_dict['step2'], strict=strict)
        if 'tail' in state_dict:
            self.model_tail.load_state_dict(state_dict['tail'], strict=strict)
        if 'memory' in state_dict:
            self.memory.load_state_dict(state_dict['memory'], load_config=False)
        if 'best_values' in state_dict:
            self.best_values.cache = state_dict['best_values']
        if 'explorer' in state_dict:
            self.explorer.count = state_dict['explorer']
        if 'baseline_dict' in state_dict:
            self.baseline_dict = state_dict['baseline_dict']
        if 'baseline_explorer' in state_dict:
            self.baseline_reward_explorer.count = state_dict['baseline_explorer']

    def model_export(self):
        return {
            'step1': self.model_step1.state_dict(),
            'step2': self.model_step2.state_dict(),
            'tail': self.model_tail.state_dict(),
            'memory': self.memory.state_dict(),
            'best_values': self.best_values.cache,
            'explorer': self.explorer.count,
            'baseline_dict': self.baseline_dict,
            'baseline_explorer': self.baseline_reward_explorer.count,
        }

    def init(self, state : typing.Union[Plan, Sql], grad=False, return_graph=False):
        if isinstance(state, Sql):
            plan = Plan(state)
            return self.init(plan, grad=grad, return_graph=return_graph)
        graph, _, node_indexes, *_ = state.sql.to_hetero_graph_dgl()
        with _dummy() if grad else torch.no_grad():
            graph : dgl.DGLHeteroGraph = self.model_step1(graph)
        hiddens = graph.nodes['table'].data['res']
        cells = torch.zeros(*hiddens.shape[:-1], database.config.feature_size, device=self.device)

        res = torch.cat([hiddens, cells], dim=-1)
        state.root_node_emb_init(res, node_indexes)
        if return_graph:
            return state, graph
        return state

    def topk_search(self, states : typing.Iterator[Plan], k=1, exploration=False, detail=False, exp_bias=0, use_beam=True, bushy=False):
        all_candidates = []
        all_res = []
        exploration_mask = []

        for state_index, state in enumerate(states):
            if isinstance(state, tuple):
                state, is_explore = state
            else:
                is_explore = False
            table_candidates, root_candidates = state.candidates(bushy=bushy)
            root_candidates = sorted(root_candidates, key=str)
            all_candidates.extend(map(lambda x: (state_index, state, x), root_candidates))
            with torch.no_grad():

                global_emb = state.root_node_emb_all_hidden()

                _len = len(state.sql.from_tables)

                global_emb = self.model_tail.aggregate(global_emb)
                state_join_encodings = state.join_encodings()

                embs = []
                for left_alias, right_alias in root_candidates:
                    left_emb, right_emb = state.root_node_emb(left_alias), state.root_node_emb(right_alias)
                    embs.append((left_emb, right_emb, _len))
                if not embs:
                    print(state.sql.filename, state._hash_str(hint=False))
                    print(state.direct_parent)
                    print(state.left_children)
                    print(state.right_children)
                    print('bushy:', bushy)
                    raise Exception('Empty candidate')
                left_emb, right_emb, lens = zip(*embs)
                left_emb = torch.stack(left_emb, dim=0)
                right_emb = torch.stack(right_emb, dim=0)
                lens = torch.tensor([_len], device=self.device)#torch.tensor(lens, device=self.device).unsqueeze(-1)
                parent_emb = self.model_step2(left_emb, right_emb, input=None)
                parent_emb = parent_emb.chunk(2, dim=-1)[0]
                left_emb = left_emb.chunk(2, dim=-1)[0]
                right_emb = right_emb.chunk(2, dim=-1)[0]

                res = self.model_tail(global_emb, lens, parent_emb, left_emb, right_emb, state_join_encodings)[..., :self.OUT_DIM]

                res_size = res.shape[0] * res.shape[1]
                res = res.reshape(res_size)
            all_res.append(res)
            exploration_mask.extend((is_explore for i in range(res_size)))

        all_res = torch.cat(all_res, dim=-1)
        exploration_mask = torch.tensor(exploration_mask, device=self.device)
        explore = 0

        prob = self.explorer.prob
        prob_coef = 0.5
        prob = prob + (1 - prob) * (1 - math.exp(-prob_coef * exp_bias))

        if use_beam:
            k = min(k, len(all_candidates))

            if exploration:
                if k > 1:
                    explore = 1
                for i in range(k - 2):
                    if random.random() < prob:
                        explore += 1

            _max = 255

            selected_res = torch.topk(all_res, k - explore, dim=-1, largest=False, sorted=True)
            selected_items = selected_res.values.tolist()
            selected_res = selected_res.indices.tolist()

            exploration_mask[selected_res] = False

            if explore > 0:
                explore_res = random.sample(range(len(all_candidates) * self.OUT_DIM), k)
                explore_res = list(filter(lambda x: x not in selected_res, explore_res))[:explore * 2]

                exploration_mask[explore_res] = True

                _max_exploration_mask = _max * exploration_mask
                explore_all_res = all_res - _max_exploration_mask
                explore_res = torch.topk(explore_all_res, explore, dim=-1, largest=False, sorted=True).indices.tolist()
            else:
                explore_res = []

            for i in explore_res:
                selected_items.append(all_res[i].item())
            selected_res.extend(explore_res)
        else:
            if random.random() < prob:
                # explore
                explore = 1
                selected_res = random.randrange(len(all_candidates) * self.OUT_DIM)
            else:
                selected_res = torch.argmin(all_res, dim=-1).item()
            selected_items = [all_res[selected_res].item()]
            selected_res = [selected_res]

        selected = []
        for index in selected_res:
            _selected, _join = index // self.OUT_DIM, index % self.OUT_DIM
            selected.append((all_candidates[_selected], _join))

        res = selected

        if detail:
            all_res_stats = (all_res.min().item(), all_res.max().item(), all_res.mean().item())
            return res, explore, selected_items, all_res_stats, prob
        return res

    def step(self, state : Plan, action, join=0):
        left_alias, right_alias = action
        left_alias, right_alias = state.find_parent(left_alias), state.find_parent(right_alias)
        if self.restricted_operator:
            join_method = Plan.ALL_JOIN if join == 0 else \
                Plan.NO_NEST_LOOP_JOIN if join == 1 else \
                Plan.NO_HASH_JOIN if join == 2 else \
                Plan.NO_MERGE_JOIN if join == 3 else \
                Plan.ALL_JOIN
        else:
            join_method = Plan.NEST_LOOP_JOIN if join == 0 else \
                Plan.HASH_JOIN if join == 1 else \
                Plan.MERGE_JOIN if join == 2 else \
                Plan.NEST_LOOP_JOIN
        parent_alias = state.join(left_alias, right_alias, join_method=join_method)

        left_emb, right_emb = state.root_node_emb(left_alias), state.root_node_emb(right_alias)
        parent_emb = self.model_step2(left_emb, right_emb, input=None)
        state.root_node_emb_set(parent_alias, parent_emb)
        return state

    def beam_plan(self, sql : Sql, bushy=False, judge=False):
        plan, graph = self.init(sql, grad=False, return_graph=True)
        plans = [plan]
        beam_width = database.config.beam_width
        use_beam = beam_width >= 1
        if not use_beam:
            beam_width = 1
        while not plans[0].completed:
            selected = self.topk_search(plans, beam_width, exploration=False, use_beam=use_beam, bushy=bushy)
            plans = []
            for (_, plan, action), join in selected:
                plan = plan.clone(deep=True)
                self.step(plan, action, join=join)
                plans.append(plan)
        plan = plans[0]
        _timer = timer()
        with _timer:
            use_generated = False
        return plan, use_generated, _timer.time

    def clear_baseline_dict(self, sql):
        self.baseline_dict[str(sql)] = set()

    def add_memory(self, state : Plan, value : float, memory_value=None, info=None, is_baseline=False):
        if memory_value is None:
            memory_value = value
        state_hash_prev = f'{state.sql.filename} {state._hash_str(hint=False)}'
        state_hash_prev_hint = f'{state.sql.filename} {state._hash_str(hint=True)}'
        state_hash = f'{state_hash_prev} {info[0]} {info[1]}'
        state_hash_hint = f'{state_hash_prev_hint} {info[0]} {info[1]}'
        if info is None:
            mem = state
        else:
            mem = (state, *info)
        self.memory.push(mem)
        self.best_values[state_hash_prev] = value
        self.best_values[state_hash] = value
        self.best_values[state_hash_prev_hint] = value
        self.best_values[state_hash_hint] = value
        if self.use_baseline_reward and is_baseline:
            baseline_set = self.baseline_dict.setdefault(str(state.sql), set())
            baseline_set.add(state_hash_prev)
            baseline_set.add(state_hash)

    def __batch_embedding_update(self, states : typing.List[Plan], return_graph=False):

        _timer = timer()

        with _timer:
            gs = []
            node_indexes = []
            _states = []
            for state in states:
                g, _, _node_indexes, *_ = state.sql.to_hetero_graph_dgl()
                gs.append(g)
                node_indexes.append(_node_indexes)

                _states.append(Plan(state.sql))
            gs = self.model_step1(dgl.batch(gs))
            hiddens = gs.nodes['table'].data['res']
            cells = torch.zeros(*hiddens.shape[:-1], database.config.feature_size, device=self.device)
            gs.nodes['table'].data['res_lstm'] = torch.cat([hiddens, cells], dim=-1)

        with _timer:
            layer_data = []

            for state, init_state, g, _node_indexes in zip(states, _states, dgl.unbatch(gs), node_indexes):
                action_seq = state.action_sequence_by_layer()

                init_state.root_node_emb_init(g.nodes['table'].data['res_lstm'], _node_indexes)
                while len(layer_data) < len(action_seq):
                    layer_data.append([])
                for layer, actions in enumerate(action_seq):
                    _layer_data = layer_data[layer]
                    for action in actions:
                        _layer_data.append((init_state, action))

        with _timer:
            for layer, _layer_data in enumerate(layer_data):
                layer_tensors = []
                for state, (left, right, join) in _layer_data:
                    state : Plan
                    left_emb, right_emb = state.root_node_emb(left), state.root_node_emb(right)
                    layer_tensors.append(torch.cat([left_emb, right_emb], dim=-1))
                left_emb, right_emb = torch.stack(layer_tensors, dim=0).chunk(2, dim=-1)
                res = self.model_step2(left_emb, right_emb)
                for (state, action), parent_emb in zip(_layer_data, res):
                    parent_alias = state.join(*action)
                    state.root_node_emb_set(parent_alias, parent_emb)

        if return_graph:
            return _states, gs
        return _states

    def train(self, detail=False):
        _timer = timer()

        with _timer:
            batch = self.memory.sample(database.config.batch_size)
            batch, batch_action, joins, *_ = zip(*batch)
            batch = self.__batch_embedding_update(batch)
        _Time_batch_update = _timer.time

        with _timer:
            gts = []
            embs = []
            batched = list(zip(batch, joins, batch_action))
            for state, join, action in batched:
                left_alias, right_alias = action
                state : Plan

                state_hash = f'{state.sql.filename} {state._hash_str(hint=False)}'
                state_hash_this = f'{state_hash} {action} {join}'
                state_hash_hint = f'{state.sql.filename} {state._hash_str(hint=True)}'
                state_hash_this_hint = f'{state_hash_hint} {action} {join}'

                groundtruth = self.best_values.get(state_hash_this)
                prev_groundtruth = self.best_values.get(state_hash)
                groundtruth_hint = self.best_values.get(state_hash_this_hint)
                prev_groundtruth_hint = self.best_values.get(state_hash_hint)

                assert groundtruth is not None

                global_embs = state.root_node_emb_all_hidden()
                _len = len(state.sql.from_tables)
                global_embs = self.model_tail.aggregate(global_embs)

                state_join_encodings = state.join_encodings()

                left_emb = state.root_node_emb(left_alias)
                right_emb = state.root_node_emb(right_alias)

                parent_emb = self.model_step2(left_emb, right_emb, input=None)
                parent_emb = parent_emb.chunk(2, dim=-1)[0]
                left_emb = left_emb.chunk(2, dim=-1)[0]
                right_emb = right_emb.chunk(2, dim=-1)[0]

                embs.append((global_embs, _len, parent_emb, left_emb, right_emb, state_join_encodings))
                gts.append(((prev_groundtruth, prev_groundtruth_hint), (groundtruth, groundtruth_hint)))

            prev_gt, gt = zip(*gts)
            if self.use_value_predict:
                gt = gt + prev_gt

            global_embs, lens, parent_emb, left_emb, right_emb, state_join_encodings = zip(*embs)

            global_embs = torch.stack(global_embs, dim=0)
            lens = torch.tensor(lens, device=self.device).unsqueeze(-1)
            parent_emb = torch.stack(parent_emb, dim=0)
            left_emb = torch.stack(left_emb, dim=0)
            right_emb = torch.stack(right_emb, dim=0)
            state_join_encodings = torch.stack(state_join_encodings, dim=0)

            joins_tensor = [i * self._tail_out_dim + join for i, join in enumerate(joins)]
            joins_tensor = torch.tensor(joins_tensor, dtype=torch.long, device=self.device)

            this_pred = self.model_tail(global_embs, lens, parent_emb, left_emb, right_emb, state_join_encodings).take(joins_tensor)
            if self.use_value_predict:
                prev_pred = self.model_tail.prev_predict(global_embs, lens, state_join_encodings).view(-1)
                pred = torch.cat([this_pred, prev_pred], dim=-1)
            else:
                pred = this_pred

            gt_cap = self.gt_cap
            if gt_cap < 0:
                _gt = [_no_hint * (1 - self.reward_weighting) + _use_hint * self.reward_weighting for _no_hint, _use_hint in gt]
                gt = torch.tensor(_gt, device=self.device)
            else:
                _lambda_gt_value = lambda i: i if i < gt_cap else gt_cap + math.log(1 + (i - gt_cap))
                _gt = [_lambda_gt_value(_no_hint) * (1 - self.reward_weighting) + _lambda_gt_value(_use_hint) * self.reward_weighting for _no_hint, _use_hint in gt]
                gt = (1 + torch.tensor(_gt, device=self.device)).log()

        _Time_predict = _timer.time

        with _timer:
            use_generated_plan_loss = torch.tensor([0], device=self.device)
        _Time_use_generated = _timer.time

        with _timer:
            losses = F.mse_loss(pred, gt, reduction='none')
            loss = losses.mean()
            self.optim.zero_grad()
            loss.backward()
            self.explorer.step()

            self.optim.step()

            self.baseline_reward_explorer.step()

        _Time_train = _timer.time

        if detail:
            return loss, use_generated_plan_loss, (_Time_batch_update, _Time_predict, _Time_train, _Time_use_generated)
        return loss, use_generated_plan_loss

    def schedule(self, epoch=None):
        self.sched.step(epoch)

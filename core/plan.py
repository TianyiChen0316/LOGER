import torch
import math

from core.sql import Sql, Baseline, database

class Plan:
    _use_cache = False
    _global_cache = {}

    ALL_JOIN = 0
    NO_NEST_LOOP_JOIN = 1
    NO_MERGE_JOIN = 2
    NO_HASH_JOIN = 4
    NEST_LOOP_JOIN = 6
    MERGE_JOIN = 5
    HASH_JOIN = 3

    @classmethod
    def str_join_method(cls, type):
        if type == cls.NO_NEST_LOOP_JOIN:
            return "NoNestLoop"
        elif type == cls.NO_MERGE_JOIN:
            return "NoMergeJoin"
        elif type == cls.NO_HASH_JOIN:
            return "NoHashJoin"
        elif type == cls.NEST_LOOP_JOIN:
            return "NestLoop"
        elif type == cls.MERGE_JOIN:
            return "MergeJoin"
        elif type == cls.HASH_JOIN:
            return "HashJoin"
        return None

    @classmethod
    def use_cache(cls, value):
        cls._use_cache = value
        cls._global_cache = {}

    default_feature_size = 128
    default_feature_length = 5

    def __init__(self, sql: Sql):
        self.sql = sql
        self.__reset()

    def __reset(self):
        self.root_nodes = set(self.sql.aliases)
        self.parent = {}
        self.direct_parent = {}
        self.left_children = {}
        self.right_children = {}
        self.total_branch_nodes = 0
        self.join_on_left = {}
        self.join_on_right = {}
        self.children_table_aliases = {}
        self.join_aliases = {}
        self.__candidates_cache = {}
        self.__eval_mode = False
        self.__baseline_attributes = {}
        self.__bushy = 0

        self.__mapping_attributes = None

        self._join_method = {}

        self._descendants = {}
        self._cmps = {}
        for alias in self.root_nodes:
            self._cmps[alias] = set()
            self._descendants[alias] = {alias}
            self.children_table_aliases[alias] = {alias}
            for v, e in self.sql.join_edges[alias]:
                s = self.join_aliases.setdefault(alias, set())
                s.add(v)

        self.edge_dict = {}
        for left_alias, right_alias, cmp in self.sql.edge_list:
            edge = (left_alias, right_alias)
            s = self.edge_dict.setdefault(left_alias, set())
            s.add(edge)
            edge = (right_alias, left_alias)
            s = self.edge_dict.setdefault(right_alias, set())
            s.add(edge)

        self.root_node_embeddings = {}
        for alias in self.root_nodes:
            self.root_node_embeddings[alias] = torch.zeros(database.config.feature_size * 4, device=self.sql.device)

    @property
    def completed(self):
        return len(self.sql.from_tables) == self.total_branch_nodes + 1

    def join_encodings(self):
        len_tables = len(database.schema.tables)
        encodings = torch.zeros(len_tables, 4)

        table_indexes = {}
        for index, table in enumerate(database.schema.tables):
            table_indexes[table.name] = index
        for alias in self.sql.aliases - self.root_nodes:
            _, _, table = self.sql.get_table(alias)
            index = table_indexes[table.name]
            direct_parent = self.direct_parent.get(alias, None)
            assert direct_parent is not None
            join = self._join_method.get(direct_parent, self.ALL_JOIN)
            join = 0 if join == self.ALL_JOIN else \
                1 if join == self.NO_NEST_LOOP_JOIN else \
                2 if join == self.NO_HASH_JOIN else \
                3 if join == self.NO_MERGE_JOIN else \
                1 if join == self.NEST_LOOP_JOIN else \
                2 if join == self.HASH_JOIN else \
                3 if join == self.MERGE_JOIN else \
                0
            encodings[index, join] = 1
        return encodings.view(-1).to(self.sql.device)

    def clone(self, deep=True):
        plan = self.__class__(self.sql)
        plan.root_nodes = set(self.root_nodes)
        plan.parent = dict(self.parent)
        plan.direct_parent = dict(self.direct_parent)
        plan.left_children = dict(self.left_children)
        plan.right_children = dict(self.right_children)
        plan.total_branch_nodes = self.total_branch_nodes
        plan.join_on_left = dict(self.join_on_left)
        plan.join_on_right = dict(self.join_on_right)
        plan.children_table_aliases = dict(self.children_table_aliases)
        plan.join_aliases = {}
        plan._join_method = dict(self._join_method)
        if deep:
            plan.root_node_embeddings = {k : v.clone().detach()
                                         for k, v in self.root_node_embeddings.items()}
        else:
            plan.root_node_embeddings = dict(self.root_node_embeddings)
        for k, v in self.join_aliases.items():
            plan.join_aliases[k] = set(v)
        plan._descendants = {}
        for k, v in self._descendants.items():
            plan._descendants[k] = set(v)
        plan._cmps = {}
        for k, v in self._cmps.items():
            plan._cmps[k] = set(v)
        plan.edge_dict = {}
        for k, v in self.edge_dict.items():
            plan.edge_dict[k] = set(v)
        return plan

    def find_parent(self, node):
        parent = node
        while parent in self.parent:
            t = parent
            parent = self.parent[parent]
            if t == parent:
                print(self.parent)
                raise Exception
        if not self.__eval_mode:
            while node in self.parent:
                temp = self.parent[node]
                self.parent[node] = parent
                node = temp
        return parent

    def can_join(self, left, right):
        return self.find_parent(left) != self.find_parent(right)

    def __join(self, left, right, join_method=ALL_JOIN, eval=False):
        parent_index = self.total_branch_nodes
        self.total_branch_nodes += 1

        left_parent, right_parent = self.find_parent(left), self.find_parent(right)

        if not isinstance(left_parent, int) and not isinstance(right_parent, int):
            self.__bushy += 1

        self.parent[left_parent] = parent_index
        self.parent[right_parent] = parent_index
        self.direct_parent[left_parent] = parent_index
        self.direct_parent[right_parent] = parent_index
        self.left_children[parent_index] = left_parent
        self.right_children[parent_index] = right_parent
        self.root_nodes.add(parent_index)
        self.root_nodes.remove(left_parent)
        self.root_nodes.remove(right_parent)

        self._join_method[parent_index] = join_method

        self.join_on_left[parent_index] = left
        self.join_on_right[parent_index] = right

        self._descendants[parent_index] = self._descendants[left_parent] | self._descendants[right_parent]

        self.__candidates_cache = {}

        s = set()
        self._cmps[parent_index] = s
        for left_alias in self._descendants[left_parent]:
            for _left_alias, right_alias in self.edge_dict.get(left_alias, ()):
                assert left_alias == _left_alias
                if not right_alias in self._descendants[right_parent]:
                    continue
                s.add((left_alias, right_alias))

        if not eval:
            children = self.children_table_aliases[left_parent] | \
                       self.children_table_aliases[right_parent]
            self.children_table_aliases[parent_index] = children
            self.join_aliases[parent_index] = \
                (self.join_aliases[left_parent] |
                 self.join_aliases[right_parent]) - children
        else:
            self.__eval_mode = True

        return parent_index

    def join(self, left, right, join_method=ALL_JOIN):
        return self.__join(left, right, join_method=join_method, eval=False)

    def join_eval(self, left, right, join_method=ALL_JOIN):
        return self.__join(left, right, join_method=join_method, eval=True)

    def join_eval_undo(self):
        self.__candidates_cache = {}
        self.__eval_mode = False

        self.total_branch_nodes -= 1
        self.root_nodes.remove(self.total_branch_nodes)
        left_child = self.left_children[self.total_branch_nodes]
        right_child = self.right_children[self.total_branch_nodes]
        self.root_nodes.add(left_child)
        self.root_nodes.add(right_child)

        if not isinstance(left_child, int) and not isinstance(right_child, int):
            self.__bushy -= 1
        self._join_method.pop(self.total_branch_nodes)

        try:
            self.__baseline_attributes.pop(self.total_branch_nodes)
        except:
            pass
        try:
            self._global_cache.pop(self.total_branch_nodes)
        except:
            pass
        self.parent.pop(self.left_children[self.total_branch_nodes])
        self.parent.pop(self.right_children[self.total_branch_nodes])
        self.direct_parent.pop(self.left_children[self.total_branch_nodes])
        self.direct_parent.pop(self.right_children[self.total_branch_nodes])

    @property
    def bushy(self):
        # at least two leaf-leaf joins
        return self.__bushy > 1

    def __attribute_update(self):
        baseline = Baseline(self)
        b2p, p2b = baseline.mapping(self)
        for plan_alias, baseline_alias in p2b.items():
            attr = baseline.attributes.get(baseline_alias, None)
            assert attr is not None
            self.__baseline_attributes.setdefault(plan_alias, attr)

    def baseline_attribute(self, node):
        attr = self.__baseline_attributes.get(node, None)
        if attr is None:
            self.__attribute_update()
            attr = self.__baseline_attributes.get(node, None)
        assert attr is not None, f"{node} {self.parent}\n{Baseline(self)}\n{self}"
        # attr: rows, columns
        rows, columns, *_ = attr
        attr = torch.tensor([
            math.log(rows + 1),
            columns,
        ], device=self.sql.device).unsqueeze(0)
        return attr

    def candidates(self, bushy=False, cache=True, no_left_right=False):
        cache_keys = (bushy, no_left_right)
        if cache:
            _cache = self.__candidates_cache.get(cache_keys, None)
            if _cache is not None:
                return _cache
        candidates = []
        edges = set()
        for left, right in self.sql.join_candidates:
            left_parent, right_parent = self.find_parent(left), self.find_parent(right)
            if no_left_right:
                if right < left and left_parent == left and right_parent == right\
                        or isinstance(right_parent, int) and (
                        not isinstance(left_parent, int) or right_parent < left_parent):
                    left, right = right, left
                    left_parent, right_parent = right_parent, left_parent
            if left_parent == right_parent:
                continue
            if not bushy and not (
                    self.total_branch_nodes == 0 or (left_parent != left and
                                                     right_parent == right)):  # left has to be a tree and right has to be a node
                continue
            candidates.append((left_parent, right_parent))
            edges.add((left, right))
        res = list(edges), set(candidates)
        self.__candidates_cache[cache_keys] = res
        return res

    def hidden_candidate_edges(self, bushy=False):
        candidates = []
        edges = []

        for eq_set in self.sql.hidden_edge_sets.values():
            eq_set = list(eq_set)
            for i, left_expr in enumerate(eq_set):
                for j in range(i + 1, len(eq_set)):
                    right_expr = eq_set[j]
                    left_alias = left_expr.alias
                    right_alias = right_expr.alias
                    left_parent = self.find_parent(left_alias)
                    right_parent = self.find_parent(right_alias)
                    if left_parent == right_parent:
                        continue
                    if not bushy:
                        if not (self.total_branch_nodes == 0 or
                                (left_parent == left_alias ^
                                right_parent == right_alias)):  # left has to be a tree and right has to be a node
                            continue
                        if right_parent != right_alias:
                            left_alias, right_alias = right_alias, left_alias
                            left_parent, right_parent = right_parent, left_parent
                    candidates.append((left_parent, right_parent))
                    edges.append((left_alias, right_alias))

        return edges, set(candidates)

    def __to_sql(self, oracle=False):
        all_cmps = set()
        tables = []
        for i in range(0, self.total_branch_nodes):
            if i in self.parent:
                continue
            table_part, cmps = self.table_sql(i, parenthesis=False, oracle=oracle)
            all_cmps.update(cmps)
            tables.append(table_part)
        rest = list(self.sql.edges - all_cmps)
        for alias in self.sql.aliases:
            if not alias in self.parent:
                if oracle:
                    _alias = self.sql.alias_to_table[alias].oracle()
                else:
                    _alias = str(self.sql.alias_to_table[alias])
                tables.append(_alias)
                rest.extend(self.sql.filters.get(alias, ()))

        if oracle:
            _str = lambda x: x.oracle()
        else:
            _str = str

        if rest:
            return ''.join((
                'select ' if not oracle else '',
                ',\n'.join((_str(x) for x in self.sql.target_tables)),
                '\nfrom ',
                ',\n'.join(x for x in tables),
                '\nwhere ',
                '\nAND '.join(map(_str, rest)),
                self.sql.tail_clauses(oracle=oracle),
                ';' if not oracle else '',
            ))
        else:
            return ''.join((
                'select ' if not oracle else '',
                ',\n'.join((_str(x) for x in self.sql.target_tables)),
                '\nfrom ',
                self.table_sql(self.total_branch_nodes - 1, parenthesis=False, oracle=oracle)[0],
                self.sql.tail_clauses(oracle=oracle),
                ';' if not oracle else '',
            ))

    def table_sql(self, node, parenthesis=True, oracle=False):
        if isinstance(node, int):
            # branch
            res = ['('] if parenthesis else []
            left_child = self.left_children[node]
            right_child = self.right_children[node]
            left, left_cmps = self.table_sql(left_child, left_child in self.sql.aliases, oracle=oracle)
            right, right_cmps = self.table_sql(right_child, oracle=oracle)

            res.append(left)
            res.append('\n')

            cmps, ons, filters = [], [], []
            # compare between column and value
            if left_child in self.sql.filters:
                for cmp in self.sql.filters[left_child]:
                    filters.append(cmp)
            if right_child in self.sql.filters:
                for cmp in self.sql.filters[right_child]:
                    filters.append(cmp)

            # compare between columns
            join_on_left, join_on_right = self.join_on_left[node], self.join_on_right[node]
            for left_table in self.children_table_aliases[left_child]:
                for right_table, cmp in self.sql.join_edges[left_table]:
                    if not right_table in self.children_table_aliases[right_child]:
                        continue
                    lcol, rcol = cmp.aliasname_list
                    if lcol == join_on_left and rcol == join_on_right or \
                            lcol == join_on_right and rcol == join_on_left:
                        cmps.append(cmp)
                    else:
                        ons.append(cmp)

            if oracle:
                _str = lambda x: x.oracle()
            else:
                _str = str

            between_columns = cmps + ons
            lst = between_columns + filters
            if lst:
                res.extend(('inner join ', right, '\non ', ' AND '.join(map(_str, lst))))
            else:
                res.extend(('cross join ', right))
            if parenthesis:
                res.append(')')

            return ''.join(res), between_columns + left_cmps + right_cmps
        else:
            # leaf
            if oracle:
                return self.sql.alias_to_table[node].oracle(), []
            return str(self.sql.alias_to_table[node]), []

    def __hint(self, node, hints : list):
        if isinstance(node, int):
            # branch node
            left_alias = self.left_children[node]
            right_alias = self.right_children[node]
            left = self.__hint(left_alias, hints)
            right = self.__hint(right_alias, hints)

            join_method = self.str_join_method(self._join_method.get(node, self.ALL_JOIN))
            if join_method is not None:
                hints.append(f'{join_method}({" ".join(sorted(self._descendants[left_alias]))} {" ".join(sorted(self._descendants[right_alias]))})')

            return f'({left} {right})'
        # leaf
        return node

    def _hint_str(self, hints=None):
        if hints is None:
            hints = []
        return self.__hint(self.total_branch_nodes - 1, hints)

    def hint_str(self, sql=True):
        hints = []
        leading = f'Leading({self._hint_str(hints)})'
        hints.append(leading)

        res = f'/*+ {" ".join(hints)} */'
        if sql:
            return f'{res} {str(self.sql)}'
        return res

    def _hash_str(self, hint=True):
        res = []
        dummy = []
        for root_alias in self.root_nodes:
            res.append(self.__hint(root_alias, dummy))
        if hint:
            res.extend(dummy)
        return ' '.join(sorted(res))

    def hash_str(self, hint=True):
        res = [str(self.sql)]
        dummy = []
        for root_alias in self.root_nodes:
            res.append(self.__hint(root_alias, dummy))
        if hint:
            res.extend(dummy)
        return ' '.join(sorted(res))

    def __action_sequence(self, node, seq=None):
        if seq is None:
            seq = []
        if isinstance(node, int):
            left_alias = self.left_children[node]
            right_alias = self.right_children[node]
            self.__action_sequence(left_alias, seq)
            self.__action_sequence(right_alias, seq)
            seq.append((left_alias, right_alias))
        return seq

    def action_sequence(self):
        res = []
        for root_node in self.root_nodes:
            self.__action_sequence(root_node, res)
        return res

    def __action_depths(self, node, data):
        if isinstance(node, int):
            left_child, right_child = self.left_children[node], self.right_children[node]
            left_depth = self.__action_depths(left_child, data)
            right_depth = self.__action_depths(right_child, data)
            depth = max(left_depth, right_depth) + 1
            data[node] = depth
            return depth
        return 0

    def __action_sequence_by_layer(self, node, data : list):
        if isinstance(node, int):
            left_alias = self.join_on_left[node]
            right_alias = self.join_on_right[node]
            join_method = self._join_method.get(node, self.ALL_JOIN)
            left_child, right_child = self.left_children[node], self.right_children[node]
            left_depth = self.__action_sequence_by_layer(left_child, data=data)
            right_depth = self.__action_sequence_by_layer(right_child, data=data)
            depth = max(left_depth, right_depth) + 1
            while len(data) < depth:
                data.append([])
            data[depth - 1].append((left_alias, right_alias, join_method))
            return depth
        return 0

    def __legacy_action_sequence_by_layer(self):
        data = []
        for root_node in self.root_nodes:
            self.__action_sequence_by_layer(root_node, data)
        return data

    def action_sequence_by_layer(self):
        if self.total_branch_nodes == 0:
            return []
        node_depths = [None for i in range(self.total_branch_nodes)]
        for root_node in self.root_nodes:
            self.__action_depths(root_node, node_depths)
        for node in range(1, len(node_depths)):
            left_child, right_child = self.left_children[node], self.right_children[node]
            depth = max(node_depths[node], node_depths[node - 1])
            if isinstance(left_child, int):
                depth = max(depth, node_depths[left_child] + 1)
            if isinstance(right_child, int):
                depth = max(depth, node_depths[right_child] + 1)
            node_depths[node] = depth
        max_depth = node_depths[-1]
        res = [[] for i in range(max_depth)]
        for node, depth in enumerate(node_depths):
            left_alias = self.join_on_left[node]
            right_alias = self.join_on_right[node]
            join_method = self._join_method.get(node, self.ALL_JOIN)
            res[depth - 1].append((left_alias, right_alias, join_method))
        return res

    @classmethod
    def oracle_join_method(cls, type):
        if type == cls.NO_NEST_LOOP_JOIN:
            return "NO_USE_NL"
        elif type == cls.NO_MERGE_JOIN:
            return "NO_USE_MERGE"
        elif type == cls.NO_HASH_JOIN:
            return "NO_USE_HASH"
        elif type == cls.NEST_LOOP_JOIN:
            return "USE_NL"
        elif type == cls.MERGE_JOIN:
            return "USE_MERGE"
        elif type == cls.HASH_JOIN:
            return "USE_HASH"
        return None

    def __oracle_hint(self, node, hints : list):
        if isinstance(node, int):
            # branch node
            left_alias = self.left_children[node]
            right_alias = self.right_children[node]
            left = self.__oracle_hint(left_alias, hints)
            right = self.__oracle_hint(right_alias, hints)

            join_method = self.oracle_join_method(self._join_method.get(node, self.ALL_JOIN))
            if join_method is not None:
                if not isinstance(left_alias, int):
                    hints.append(f'{join_method}({left_alias})')
                if not isinstance(right_alias, int):
                    hints.append(f'{join_method}({right_alias})')

            return f'({left} {right})'
        # leaf
        return node

    def _oracle_hint_str(self, hints=None):
        if hints is None:
            hints = []
        return self.__oracle_hint(self.total_branch_nodes - 1, hints)

    def __oracle_leading(self, node=None):
        if node is None:
            node = self.total_branch_nodes - 1
        if isinstance(node, int):
            left_alias = self.left_children[node]
            right_alias = self.right_children[node]
            left = self.__oracle_leading(left_alias)
            right = self.__oracle_leading(right_alias)
            if isinstance(left_alias, int):
                if isinstance(right_alias, int):
                    # Oracle supports left-deep only
                    assert False, 'Oracle does not support bushy plans'
                else:
                    return f'{left} {right}'
            elif isinstance(right_alias, int):
                return f'{right} {left}'
            else:
                return f'{left} {right}'
        else:
            return node

    def oracle(self):
        hints = []
        self._oracle_hint_str(hints)
        leading = self.__oracle_leading()
        hints = f'/*+ LEADING({leading}) {" ".join(hints)} */'

        res = self.__to_sql(oracle=True)
        res = f'select {hints} {res}'
        if self.sql.limit_clause is not None:
            res = f'select * from ({res}) where {self.sql.limit_clause.oracle()}'
        return res

    def __str__(self):
        hints = []
        self._hint_str(hints)
        hints = f'/*+ {" ".join(hints)} */ '

        res = self.__to_sql()
        if False and self.completed:
            assert (res == ''.join((
                'select ',
                ',\n'.join((str(x) for x in self.sql.target_tables)),
                '\nfrom ',
                self.table_sql(self.total_branch_nodes - 1, parenthesis=False)[0],
                ';',
            )))
        return hints + res

    def cost(self):
        for pre in self.sql.pre_actions:
            database.result(pre)
        res = database.cost(str(self))
        for post in self.sql.post_actions:
            database.result(post)
        return res

    def relative_cost(self):
        return self.cost() / self.sql.cost()

    def latency(self):
        for pre in self.sql.pre_actions:
            database.result(pre)
        res = database.latency(str(self))
        for post in self.sql.post_actions:
            database.result(post)
        return res

    def relative_latency(self):
        return self.latency() / self.sql.latency()

    def reward(self, use_latency=False, relative=False, log=True):
        if relative:
            res = self.relative_latency() if use_latency else self.relative_cost()
        else:
            res = self.latency() if use_latency else self.cost()
        if log:
            return math.log(1 + res)
        return res

    @property
    def baseline(self):
        return self.sql.baseline

    def baseline_candidates(self):
        return Baseline.candidate_steps(self, database.config.bushy)

    def __update_mapping_attributes(self):
        assert self.completed
        if self.__mapping_attributes is None:
            base = Baseline(database.plan_latency(str(self))[0][0][0]['Plan'])
            self.__mapping_attributes = base.mapping_attrs(self)

    def actual_rows(self, node):
        self.__update_mapping_attributes()
        expect_rows, width, actual_rows, *_ = self.__mapping_attributes.get(node)
        return actual_rows

    def expect_rows(self, node):
        attr = self.__baseline_attributes.get(node, None)
        if attr is None:
            self.__attribute_update()
            attr = self.__baseline_attributes.get(node, None)
        assert attr is not None
        expect_rows, width, *_ = attr
        return expect_rows

    def __root_node_encoding(self, root_alias):
        return torch.zeros(database.config.feature_size, device=self.sql.device)

    def root_node_emb_update(self, tensors, node_indexes):
        for alias in sorted(self.root_nodes, key=str):
            node_index = node_indexes[f'.{alias}']
            self.root_node_embeddings[alias] = tensors[node_index]

    def root_node_emb_init(self, tensors, node_indexes):
        for alias in sorted(self.sql.aliases):
            node_index = node_indexes[f'~{alias}']
            self.root_node_embeddings[alias] = tensors[node_index]

    def root_node_emb_set(self, alias, value):
        self.root_node_embeddings[alias] = value

    def root_node_emb(self, alias):
        return self.root_node_embeddings.get(alias, torch.zeros(database.config.feature_size, device=self.sql.device))

    def root_node_emb_all(self):
        res = []
        for alias in sorted(self.root_nodes, key=str):
            res.append(self.root_node_embeddings[alias])
        return torch.stack(res, dim=0)

    def root_node_emb_all_hidden(self):
        res = []
        for alias in sorted(self.root_nodes, key=str):
            res.append(self.root_node_embeddings[alias].chunk(2, dim=-1)[0])
        return torch.stack(res, dim=0)

    def leaf_node_emb_all_hidden(self):
        res = []
        for alias in sorted(self.sql.aliases):
            res.append(self.root_node_embeddings[alias].chunk(2, dim=-1)[0])
        return torch.stack(res, dim=0)

    def node_emb_all_hidden(self):
        assert self.completed
        res = []
        for alias in range(self.total_branch_nodes):
            res.append(self.root_node_embeddings[alias].chunk(2, dim=-1)[0])
        return torch.stack(res, dim=0)

    def root_node_embedding_candidates(self, update_func):
        edges, candidates = self.candidates(bushy=database.config.bushy)
        candidates = list(candidates)
        res = []
        for left_alias, right_alias in candidates:
            left_emb, right_emb = self.root_node_emb(left_alias), self.root_node_emb(right_alias)
            emb = update_func(left_emb, right_emb)
            res.append(emb)
        res = torch.stack(res, dim=0)
        return res, candidates

    def next_plans(self, update_func):
        updates = zip(self.root_node_embedding_candidates(update_func))
        res = []
        for new_emb, candidate in updates:
            p = self.clone()
            p.join(*candidate)
            p.root_node_embeddings[p.total_branch_nodes - 1] = new_emb
            res.append(p)
        return res

    def to_hetero_graph_dgl(self, bushy=False):
        _dgl, data_dict, node_indexes, *_ = self.sql.to_hetero_graph_dgl()
        return _dgl, node_indexes

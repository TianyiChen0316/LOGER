import torch
import dgl
import dgl.nn.pytorch.conv as dglconv
import math

from . import GTConv

from core import database

class TableTransform(torch.nn.Module):
    def __init__(self, hidden_size=32, num_heads=16, aggr=None):
        super().__init__()
        if aggr is None:
            aggr = 'max'
        self.__aggr = aggr

        self.__table_feature_size = database.schema.max_columns
        self.__table_global_size = database.config.feature_extra_length
        self.__onehot_size = database.schema.size

        self.__feature_size = database.config.feature_size

        self.__hidden_size = hidden_size
        self.__num_heads = num_heads

        self.onehot_transform = torch.nn.Sequential(
            torch.nn.Linear(
                self.__onehot_size +
                self.__table_global_size +
                0,
                self.__feature_size,
                bias=False,
            ),
            #torch.nn.ReLU(),
        )

        self.onehot_embedding = torch.nn.Linear(self.__onehot_size, self.__hidden_size * self.__hidden_size, bias=False)
        self.schema_prepare = torch.nn.Linear(13, self.__hidden_size, bias=False)
        self.schema_embedding = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.__hidden_size, num_heads * 3),
        )

        self.table_edge_fc = torch.nn.Linear(num_heads, hidden_size, bias=False)
        self.table_filter_transform_left = torch.nn.Linear(num_heads, self.__hidden_size, bias=False)
        self.table_filter_transform_right = torch.nn.Linear(num_heads, self.__hidden_size, bias=False)

        self.table_filter_tail = torch.nn.Linear(self.__hidden_size * 3 + self.__feature_size, self.__feature_size, bias=False)

    def forward(self, g : dgl.DGLHeteroGraph):
        """
          LOGER supports incremental training on schema updates. The one-hot encoding vectors here are developed in the previous
          version. For convenience, we keep parts of them in the current version to implement learned embedding vectors and
          matrices by multiplication with weight matrices and reshaping. These one-hot vectors can be equally replaced with
          corresponding learned vectors and matrices as input.
        """
        table_filter = g.nodes['table'].data['filter']
        table_mask = g.nodes['table'].data['filter_mask']
        table_edge = g.nodes['table'].data['edge']
        table_global = g.nodes['table'].data['global']
        table_onehot = g.nodes['table'].data['onehot']
        table_others = g.nodes['table'].data['others']

        table_encoding = torch.cat([
            table_onehot,
        ], dim=-1)

        onehot_embedding : torch.Tensor = self.onehot_embedding(table_encoding).reshape(-1, self.__hidden_size, self.__hidden_size)

        table_others = self.schema_prepare(table_others)
        onehot_embedding = torch.bmm(table_others, onehot_embedding)
        onehot_embedding = self.schema_embedding(onehot_embedding)

        edge_embedding = onehot_embedding[..., 2 * self.__num_heads:3 * self.__num_heads]
        onehot_embedding = torch.cat([
            onehot_embedding[..., :self.__num_heads],
            onehot_embedding[..., self.__num_heads:2 * self.__num_heads],
        ], dim=-2)

        table_other_features = torch.cat([
            table_onehot,
            table_global,
        ], dim=-1)
        onehot_transform = self.onehot_transform(table_other_features)

        _table_mask = table_mask.to(torch.bool)

        table_filter = torch.cat([table_filter, 1 - table_filter], dim=-1)
        table_mask = torch.cat([table_mask, table_mask], dim=-1)

        table_filter = -torch.log(torch.clamp(1 - (table_filter * table_mask), 1e-9))

        table_filter = table_filter.unsqueeze(-1)
        _table_filter = onehot_embedding * table_filter
        table_filter_left, table_filter_right = _table_filter[..., :self.__table_feature_size + 1, :], _table_filter[..., self.__table_feature_size + 1:, :]

        table_edge_mask = table_edge.to(torch.bool)
        table_edge = edge_embedding * table_edge.unsqueeze(-1)

        if _table_mask.ndim >= 2:
            edges = []
            lefts = []
            rights = []
            for batch_index in range(_table_mask.shape[0]):
                _tfe = table_edge[batch_index, ...]
                _tfl = table_filter_left[batch_index, ...]
                _tfr = table_filter_right[batch_index, ...]

                _tem = table_edge_mask[batch_index, ...].unsqueeze(-1)
                _tm = _table_mask[batch_index, ...].unsqueeze(-1)

                _tfe = _tfe.masked_select(_tem.unsqueeze(-1)).view(-1, self.__num_heads)
                _tfl = _tfl.masked_select(_tm.unsqueeze(-1)).view(-1, self.__num_heads)
                _tfr = _tfr.masked_select(_tm.unsqueeze(-1)).view(-1, self.__num_heads)

                if _tfe.shape[-2] == 0:
                    _tfe = torch.zeros(*_tfe.shape[:-2], _tfe.shape[-1], device=_tfe.device)
                else:
                    if self.__aggr == 'mean':
                        _tfe = _tfe.mean(dim=-2, keepdim=False)
                    elif self.__aggr == 'sum':
                        _tfe = _tfe.sum(dim=-2, keepdim=False)
                    else:
                        _tfe = _tfe.max(dim=-2, keepdim=False).values

                if _tfl.shape[-2] == 0:
                    _tfl = torch.zeros(*_tfl.shape[:-2], _tfl.shape[-1], device=_tfl.device)
                else:
                    if self.__aggr == 'mean':
                        _tfl = _tfl.mean(dim=-2, keepdim=False)
                    elif self.__aggr == 'sum':
                        _tfl = _tfl.sum(dim=-2, keepdim=False)
                    else:
                        _tfl = _tfl.max(dim=-2, keepdim=False).values

                if _tfr.shape[-2] == 0:
                    _tfr = torch.zeros(*_tfr.shape[:-2], _tfr.shape[-1], device=_tfr.device)
                else:
                    if self.__aggr == 'mean':
                        _tfr = _tfr.mean(dim=-2, keepdim=False)
                    elif self.__aggr == 'sum':
                        _tfr = _tfr.sum(dim=-2, keepdim=False)
                    else:
                        _tfr = _tfr.max(dim=-2, keepdim=False).values

                edges.append(_tfe)
                lefts.append(_tfl)
                rights.append(_tfr)
            table_edge = torch.stack(edges, dim=0)
            table_filter_left = torch.stack(lefts, dim=0)
            table_filter_right = torch.stack(rights, dim=0)
        else:
            table_edge = table_edge.masked_select(table_edge_mask.unsqueeze(-1)).view(*table_edge.shape[:-2], -1, self.__num_heads)
            table_filter_left = table_filter_left.masked_select(_table_mask.unsqueeze(-1)).view(*table_filter_left.shape[:-2], -1, self.__num_heads)
            table_filter_right = table_filter_right.masked_select(_table_mask.unsqueeze(-1)).view(*table_filter_right.shape[:-2], -1, self.__num_heads)

            if table_edge.shape[-2] == 0:
                table_edge = torch.zeros(*table_edge.shape[:-2], table_edge.shape[-1], device=table_edge.device)
            else:
                if self.__aggr == 'mean':
                    table_edge = table_edge.mean(dim=-2, keepdim=False)
                elif self.__aggr == 'sum':
                    table_edge = table_edge.sum(dim=-2, keepdim=False)
                else:
                    table_edge = table_edge.max(dim=-2, keepdim=False).values

            if table_filter_left.shape[-2] == 0:
                table_filter_left = torch.zeros(*table_filter_left.shape[:-2], table_filter_left.shape[-1], device=table_filter_left.device)
            else:
                if self.__aggr == 'mean':
                    table_filter_left = table_filter_left.mean(dim=-2, keepdim=False)
                elif self.__aggr == 'sum':
                    table_filter_left = table_filter_left.sum(dim=-2, keepdim=False)
                else:
                    table_filter_left = table_filter_left.max(dim=-2, keepdim=False).values

            if table_filter_right.shape[-2] == 0:
                table_filter_right = torch.zeros(*table_filter_right.shape[:-2], table_filter_right.shape[-1], device=table_filter_right.device)
            else:
                if self.__aggr == 'mean':
                    table_filter_right = table_filter_right.mean(dim=-2, keepdim=False)
                elif self.__aggr == 'sum':
                    table_filter_right = table_filter_right.sum(dim=-2, keepdim=False)
                else:
                    table_filter_right = table_filter_right.max(dim=-2, keepdim=False).values

        edge_embedding = self.table_edge_fc(table_edge)
        table_filter_left : torch.Tensor = self.table_filter_transform_left(table_filter_left)
        table_filter_right : torch.Tensor = self.table_filter_transform_right(table_filter_right)

        res = torch.cat([table_filter_left, table_filter_right, edge_embedding, onehot_transform], dim=-1)
        res = self.table_filter_tail(res)
        return res


class Step1(torch.nn.Module):
    def __init__(self, num_table_layers=3):
        super().__init__()

        self.__feature_size = database.config.feature_size

        self.__table_feature_size = database.schema.max_columns * database.config.feature_length + database.config.feature_extra_length

        self.table_transform = TableTransform()

        self.num_table_layers = num_table_layers
        self.table_to_table = [GTConv(
            self.__feature_size,
            self.__feature_size,
            num_heads=8,
            dropout=0.1,
            layer_norm=True,
            residual=True,
            use_bias=False,
        ) for i in range(self.num_table_layers)]
        for i in range(self.num_table_layers):
            self.add_module(f'table_to_table_{i}', self.table_to_table[i])

    def forward(self, g):
        table_x = self.table_transform(g)

        table_res = table_x
        for i in range(self.num_table_layers):
            table_res = self.table_to_table[i](
                (dgl.edge_type_subgraph(g, [('table', 'to', 'table')])),
                table_res,
            )

        g.nodes['table'].data['res'] = table_res

        return g


class PredictTail(torch.nn.Module):
    def __init__(self, out_dim=1, aggr='mean'):
        super().__init__()
        self.__feature_size = database.config.feature_size

        if aggr == 'mean':
            self.aggr = lambda x: x.mean(dim=0, keepdim=False)
        elif aggr == 'max':
            self.aggr = lambda x: x.max(dim=0, keepdim=False).values
        else:
            # aggr == 'sum'
            self.aggr = lambda x: x.sum(dim=0, keepdim=False)

        # mean
        self.aggr = lambda x: x.mean(dim=0, keepdim=False)

        self.embedding_tail = torch.nn.Sequential(
            torch.nn.Linear(self.__feature_size, self.__feature_size),
            torch.nn.ReLU(),
        )

        database_num_tables = database.schema.size

        self.schema_size = database_num_tables

        self.join_matrix_embeddings = torch.nn.parameter.Parameter(
            torch.empty((self.__feature_size, database_num_tables * 4)), requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.join_matrix_embeddings, a=math.sqrt(5))

        self.extra_fc = torch.nn.Linear(self.__feature_size * 2, self.__feature_size)

        self.tail = torch.nn.Sequential(
            torch.nn.Linear(self.__feature_size * 3, self.__feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__feature_size, out_dim),
        )

        self.parent_fc = torch.nn.Sequential(
            torch.nn.Linear(self.__feature_size * 3, self.__feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__feature_size, self.__feature_size),
        )
        self.prev_tail = torch.nn.Sequential(
            torch.nn.Linear(self.__feature_size * 2, self.__feature_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__feature_size, 1),
        )

    def aggregate(self, hiddens):
        return self.aggr(hiddens)

    def prev_predict(self, g, extra, joins):
        if g.dim() == 1:
            g = g.unsqueeze(0)
        res = self.embedding_tail(g)

        matrix = (self.join_matrix_embeddings * joins.view(*joins.shape[:-1], 1, joins.shape[-1])).sum(dim=-1, keepdim=False) / extra
        if matrix.dim() == 1:
            matrix = matrix.unsqueeze(0).repeat(g.shape[0], 1)
        res = self.prev_tail(torch.cat([res, matrix], dim=-1))
        return res

    def forward(self, g, extra, parent_emb, left_emb, right_emb, joins):
        parent_emb = torch.cat([parent_emb, left_emb, right_emb], dim=-1)
        if parent_emb.dim() == 1:
            parent_emb = parent_emb.unsqueeze(0)
        parent_emb = self.parent_fc(parent_emb)

        res = self.embedding_tail(g)

        matrix = (self.join_matrix_embeddings * joins.view(*joins.shape[:-1], 1, joins.shape[-1])).sum(dim=-1, keepdim=False) / extra
        res = torch.cat([res, matrix], dim=-1)
        if res.dim() == 1:
            res = res.unsqueeze(0).repeat(parent_emb.shape[0], 1)
        res = self.tail(torch.cat([res, parent_emb], dim=-1))
        return res


class UseGeneratedPredict(torch.nn.Module):
    def __init__(self, num_table_layers=3):
        super().__init__()
        self.__feature_size = database.config.feature_size
        database_num_tables = database.schema.size
        database_num_columns = database.schema.total_columns
        self._use_generated_plan_predict = torch.nn.Sequential(
            torch.nn.Linear(self.__feature_size + database_num_tables * database_num_tables + database_num_columns, self.__feature_size, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.__feature_size, 2),
        )
        self.num_table_layers = num_table_layers
        self.table_to_table = [GTConv(
            self.__feature_size,
            self.__feature_size,
            num_heads=8,
            dropout=0.1,
            layer_norm=True,
            residual=True,
            use_bias=False,
        ) for i in range(self.num_table_layers)]
        for i in range(self.num_table_layers):
            self.add_module(f'table_to_table_{i}', self.table_to_table[i])
        self.pooling = dgl.nn.AvgPooling()

    def forward(self, g : dgl.DGLHeteroGraph, input):
        subgraph : dgl.DGLGraph = dgl.edge_type_subgraph(g, [('table', 'to', 'table')])
        subgraph.set_batch_num_nodes(g.batch_num_nodes('table'))
        table_res = g.nodes['table'].data['res']
        for i in range(self.num_table_layers):
            table_res = self.table_to_table[i](
                subgraph,
                table_res,
            )
        table_res = self.pooling(subgraph, table_res)
        res = self._use_generated_plan_predict(torch.cat([table_res, input], dim=-1))
        return res

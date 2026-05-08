import torch
import torch.nn.functional as F


class PrototypeRefinementModule(torch.nn.Module):
    def __init__(self, temperature=10.0, top_z=5, similarity='cosine'):
        super().__init__()
        self.t1 = temperature
        self.top_z = top_z
        assert similarity in ['cosine', 'euclidean']
        self.similarity = similarity

    def compute_similarity(self, x, proto):
        x = x.view(-1, x.size(-1))  # 展平为 [*, C]
        proto = proto.view(-1, proto.size(-1))  # 展平为 [*, C]
        if self.similarity == 'cosine':
            x_norm = F.normalize(x, dim=-1)
            proto_norm = F.normalize(proto, dim=-1)
            return torch.mm(x_norm, proto_norm.t())  # [M, N]
        else:
            return -torch.cdist(x, proto)  # [M, N]

    def pseudo_label_query(self, query_features, prototypes):
        sim = self.compute_similarity(query_features, prototypes)
        return F.softmax(sim * self.t1, dim=-1)  # [M, N]

    def compute_entropy(self, w):
        return -torch.sum(w * torch.log(w + 1e-8), dim=-1)  # [M]

    def refine_prototypes(self, prototypes, query_features):
        # 处理 3D prototypes + 4D query_features 的情况
        if prototypes.dim() == 3 and query_features.dim() == 4:
            B, Ntask_q, M, C = query_features.size()  # query的批次、任务数
            total_query_tasks = B * Ntask_q  # query总任务数（批次×任务）

            # 扩展prototypes以匹配query的总任务数
            Ntask_p, N, _ = prototypes.size()  # prototype的任务数
            if Ntask_p != total_query_tasks:
                # 若原型任务数与query总任务数不匹配，则重复原型以适配（适用于批次共享原型的场景）
                prototypes = prototypes.repeat_interleave(total_query_tasks // Ntask_p, dim=0)

            # 将query压缩为3D：[B*Ntask_q, M, C]
            query_features = query_features.view(total_query_tasks, M, C)

        # 确保两者都是3D
        assert prototypes.dim() == 3 and query_features.dim() == 3, "处理后必须为3D"
        total_tasks, N, C = prototypes.size()
        _, M, _ = query_features.size()
        assert total_tasks == query_features.size(0), "总任务数必须匹配"

        refined_prototypes = torch.zeros_like(prototypes)

        for task in range(total_tasks):
            proto_task = prototypes[task].view(N, C)  # [N, C]
            query_task = query_features[task].view(M, C)  # [M, C]

            w = self.pseudo_label_query(query_task, proto_task)  # [M, N]
            entropy = self.compute_entropy(w)  # [M]

            for n in range(N):
                w_n = w[:, n]
                conf_score = w_n / (entropy + 1e-8)
                top_k = min(self.top_z, M)
                top_idx = torch.topk(conf_score, top_k).indices

                selected_feat = query_task[top_idx]
                selected_w = w_n[top_idx].unsqueeze(-1)

                if selected_feat.size(0) > 0:
                    weighted_query = torch.sum(selected_w * selected_feat, dim=0)
                    refined_proto = (proto_task[n] + weighted_query) / (1 + top_k)
                else:
                    refined_proto = proto_task[n]

                refined_prototypes[task, n] = refined_proto

        return refined_prototypes

    def forward(self, prototypes, query_features):
        return self.refine_prototypes(prototypes, query_features)
import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfCorrelationModule(nn.Module):
    def __init__(self):
        super(SelfCorrelationModule, self).__init__()

    def forward(self, x):
        """
        x: (B, shot, way, C) - 输入特征
        返回: (B, shot, way, C) - 自相关增强后的特征
        """
        B, shot, way, C = x.shape

        # (1) 调整形状为 (B*way, shot, C) 进行类内自相关
        x = x.permute(0, 2, 1, 3).contiguous().view(B * way, shot, C)

        # (2) 生成注意力图 Attention Map
        x_flat = x  # (B*way, shot, C)
        #attention = F.softmax(torch.bmm(x_flat, x_flat.transpose(1, 2)), dim=-1)  # (B*way, shot, shot)
        attention = F.softmax(torch.bmm(x_flat, x_flat.transpose(1, 2)) / (C ** 0.5), dim=-1)
        # (3) 计算加权增强特征
        enhanced_features = torch.bmm(attention, x_flat)  # (B*way, shot, C)

        # (4) 恢复原始形状 (B, shot, way, C)
        #enhanced_features = enhanced_features.contiguous().view(B, way, shot, C).permute(0, 2, 1, 3)
        enhanced_features = enhanced_features.reshape(B, way, shot, C).permute(0, 2, 1, 3)  # 使用 reshape 替代 view

        return enhanced_features

class CrossCorrelationModule(nn.Module):
    def __init__(self, temperature=0.05):
        super(CrossCorrelationModule, self).__init__()
        self.temperature = temperature

    def forward(self, Fq, Fs):
        """
        Args:
            Fq: 查询集特征 (B, query, way, C)
            Fs: 支持集特征 (B, shot, way, C)
        Returns:
            cq: 由 query 加权生成的 support 特征 (B, shot, way, C)
            cs: 由 support 加权生成的 query 特征 (B, query, way, C)
        """
        B, query, way, C = Fq.shape
        _, shot, _, _ = Fs.shape

        # 1. 调整形状，便于计算相似度
        #Fq_flat = Fq.view(B, query * way, C)      # (B, query*way, C)
        Fq_flat = Fq.contiguous().view(B, query * way, C)
        Fs_flat = Fs.contiguous().view(B, shot * way, C)       # (B, shot*way, C)

        # 2. 计算 Cosine Similarity
        cos_sim = torch.bmm(Fq_flat, Fs_flat.transpose(1, 2))  # (B, query*way, shot*way)
        norm_q = torch.norm(Fq_flat, dim=-1, keepdim=True)  # (B, query*way, 1)
        norm_s = torch.norm(Fs_flat, dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, shot*way)
        cos_sim = cos_sim / (norm_q @ norm_s + 1e-8)

        # 3. 计算 Cross-Correlation Attention Map
        Mq = F.softmax(cos_sim / self.temperature, dim=-1)  # (B, query*way, shot*way)
        Ms = F.softmax(cos_sim.transpose(1, 2) / self.temperature, dim=-1)  # (B, shot*way, query*way)

        # 4. 使用 Attention Map 加权特征
        cq_flat = torch.bmm(Mq, Fs_flat)  # (B, query*way, C)
        cs_flat = torch.bmm(Ms, Fq_flat)  # (B, shot*way, C)

        # 5. 恢复原始形状
        cq = cq_flat.view(B, query, way, C)  # (B, query, way, C)
        cs = cs_flat.view(B, shot, way, C)   # (B, shot, way, C)

        return cq, cs
'''
#放原型网络之后
class CrossCorrelationModule(nn.Module):        
    def __init__(self, temperature=0.05):
        super(CrossCorrelationModule, self).__init__()
        self.temperature = temperature

    def forward(self, Fq, Fs):
        """
        Args:
            Fq: 查询集特征 (B, query, way, C)
            Fs: 支持集原型 (B, way, C)
        Returns:
            cq: 由 query 加权生成的 support 特征 (B, query, way, C)
            cs: 由 support 加权生成的 query 特征 (B, way, C)
        """
        B, query, way, C = Fq.shape
        # 确保支持集原型维度一致
        if Fs.ndim == 3:  # (B, way, C) -> (B, 1, way, C)
            Fs = Fs.unsqueeze(1)  # 适配维度

        _, _, _, _ = Fs.shape

        # 1. 调整形状
        Fq_flat = Fq.contiguous().view(B, query * way, C)  # (B, query * way, C)
        Fs_flat = Fs.contiguous().view(B, way, C)          # (B, way, C)

        # 2. 计算 Cosine Similarity
        cos_sim = torch.bmm(Fq_flat, Fs_flat.transpose(1, 2))  # (B, query*way, way)
        norm_q = torch.norm(Fq_flat, dim=-1, keepdim=True)     # (B, query*way, 1)
        norm_s = torch.norm(Fs_flat, dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, way)
        cos_sim = cos_sim / (norm_q @ norm_s + 1e-8)           # (B, query*way, way)

        # 3. 计算 Cross-Correlation Attention Map
        Mq = F.softmax(cos_sim / self.temperature, dim=-1)  # (B, query*way, way)
        Ms = F.softmax(cos_sim.transpose(1, 2) / self.temperature, dim=-1)  # (B, way, query*way)

        # 4. 使用 Attention Map 加权特征
        cq_flat = torch.bmm(Mq, Fs_flat)  # (B, query*way, C)
        cs_flat = torch.bmm(Ms, Fq_flat)  # (B, way, C)

        # 5. 恢复原始形状
        cq = cq_flat.view(B, query, way, C)  # (B, query, way, C)
        cs = cs_flat                         # (B, way, C)

        return cq, cs

class SelfCorrelationModule(nn.Module):
    def __init__(self):
        super(SelfCorrelationModule, self).__init__()

    def forward(self, Fq, Fs):
        """
        Args:
            Fq: 查询集特征 (B, query, way, C)
            Fs: 支持集特征 (B, way, C)
        Returns:
            enhanced_Fq: 自相关增强后的查询集特征 (B, query, way, C)
            enhanced_Fs: 自相关增强后的支持集特征 (B, way, C)
        """
        B, query, way, C = Fq.shape

        # (1) 处理查询集 Fq
        Fq_flat = Fq.view(B * query, way, C)  # (B * query, way, C)
        attention_q = F.softmax(torch.bmm(Fq_flat, Fq_flat.transpose(1, 2)), dim=-1)  # (B * query, way, way)
        enhanced_Fq = torch.bmm(attention_q, Fq_flat)  # (B * query, way, C)
        enhanced_Fq = enhanced_Fq.view(B, query, way, C)  # 恢复形状

        # (2) 处理支持集 Fs
        attention_s = F.softmax(torch.bmm(Fs, Fs.transpose(1, 2)), dim=-1)  # (B, way, way)
        enhanced_Fs = torch.bmm(attention_s, Fs)  # (B, way, C)

        return enhanced_Fq, enhanced_Fs
'''

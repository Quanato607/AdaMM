import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 仅保留与 MutualNet 相关的 3D 模块

class BasicConv3d(nn.Module):
    """3D 基础卷积: Conv3d -> BatchNorm3d -> ReLU"""
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm3d, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.stdv = 1. / math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm=nn.BatchNorm2d, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU()

        self.stdv = 1. / math.sqrt(in_channels)

    def reset_params(self):
        self.conv.weight.data.uniform_(-self.stdv, self.stdv)
        self.bn.weight.data.uniform_()
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    

import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNet3D(nn.Module):
    def __init__(self, node_num, dim, normalize_input=False):
        super(GraphNet3D, self).__init__()
        # 节点数量（簇的数量）
        self.node_num = node_num
        # 特征维度（描述子的维度）
        self.dim = dim
        # 是否对输入进行归一化
        self.normalize_input = normalize_input

        # 初始化可学习的簇中心（anchor）参数，shape: [node_num, dim]
        self.anchor = nn.Parameter(torch.rand(node_num, dim))
        # 初始化可学习的带宽参数 sigma，shape: [node_num, dim]
        self.sigma = nn.Parameter(torch.rand(node_num, dim))

    def init(self, initcache):
        """
        从给定的h5文件中加载预先计算的簇中心和描述子
        initcache: h5文件路径，包含 'centroids' 和 'descriptors'
        """
        if not os.path.exists(initcache):
            print(initcache + ' 不存在!!!\n')
        else:
            with h5py.File(initcache, mode='r') as h5:
                # 从文件中获取初始簇中心和描述子
                clsts = h5.get("centroids")[...]
                traindescs = h5.get("descriptors")[...]
                # 使用内置方法加载到模型参数
                self.init_params(clsts, traindescs)
                # 释放内存
                del clsts, traindescs

    def init_params(self, clsts, traindescs=None):
        """
        将加载的簇中心 clsts 转换为模型可训练参数
        """
        # 将 anchor 参数替换为从文件加载的中心
        self.anchor = nn.Parameter(torch.from_numpy(clsts))

    def gen_soft_assign(self, x, sigma):
        """
        计算每个像素/特征向量到各簇中心的软分配
        x: 输入特征，shape [B, C, H, W, D]
        sigma: 带宽参数，shape [node_num, dim]
        返回 soft_assign: shape [B, node_num, H*W]
        """
        B, C, H, W, D = x.size()
        N = H * W * D
        # 初始化 soft_assign 张量
        soft_assign = torch.zeros([B, self.node_num, N], device=x.device, dtype=x.dtype)
        # 将特征重塑为 [B, N, C]
        flattened = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        # 对每个簇中心计算负二范数，以便后续softmax
        for node_id in range(self.node_num):
            # 计算残差并按 sigma 缩放
            residual = (flattened - self.anchor[node_id]).div(sigma[node_id])
            # 计算 L2 范数平方，并取负半
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        # 在簇维度上softmax，得到概率分配
        soft_assign = F.softmax(soft_assign, dim=1)
        return soft_assign

    def forward(self, x):
        """
        前向传播：将输入特征图映射到 GraphNet 节点表示
        返回:
            nodes: 聚合后的节点特征，shape [B, C, node_num]
            soft_assign: 软分配矩阵，shape [B, node_num, H*W]
        """
        B, C, H, W, D = x.size()
        # 可选地对输入描述子进行 L2 归一化
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # 对 sigma 进行 sigmoid，使其为正
        sigma = torch.sigmoid(self.sigma)
        # 计算软分配
        soft_assign = self.gen_soft_assign(x, sigma)

        eps = 1e-9
        # 初始化节点特征聚合结果
        nodes = torch.zeros([B, self.node_num, C], device=x.device, dtype=x.dtype)
        flattened = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        # 对每个簇执行加权残差聚合
        for node_id in range(self.node_num):
            # 计算残差并按 sigma 缩放
            residual = (flattened - self.anchor[node_id]).div(sigma[node_id])
            # 加权求和并除以分配概率总和
            weight = soft_assign[:, node_id, :].unsqueeze(2)
            sum_weight = soft_assign[:, node_id, :].sum(dim=1).unsqueeze(1) + eps
            nodes[:, node_id, :] = (residual * weight).sum(dim=1) / sum_weight

        # 对每个节点的特征进行 L2 归一化
        nodes = F.normalize(nodes, p=2, dim=2)
        # 将节点特征展平，并全局 L2 归一化
        nodes = nodes.view(B, -1).contiguous()
        nodes = F.normalize(nodes, p=2, dim=1)
        # 恢复形状 [B, C, node_num]
        return nodes.view(B, C, self.node_num).contiguous(), soft_assign


class GraphConvNet(nn.Module):
    '''
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    '''

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
        support = torch.matmul(x_t, self.weight)  # b x k x c

        adj = torch.softmax(adj, dim=2)
        output = (torch.matmul(adj, support)).permute(0, 2, 1).contiguous()  # b x c x k

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class CascadeGCNet(nn.Module):
    def __init__(self, dim, loop):
        super(CascadeGCNet, self).__init__()
        self.gcn1 = GraphConvNet(dim, dim)
        self.gcn2 = GraphConvNet(dim, dim)
        self.gcn3 = GraphConvNet(dim, dim)
        self.gcns = [self.gcn1, self.gcn2, self.gcn3]
        assert (loop == 1 or loop == 2 or loop == 3)
        self.gcns = self.gcns[0:loop]
        self.relu = nn.ReLU()

    def forward(self, x):
        for gcn in self.gcns:
            x_t = x.permute(0, 2, 1).contiguous()  # b x k x c
            x = gcn(x, adj=torch.matmul(x_t, x))  # b x c x k
        x = self.relu(x)
        return x
    
class MutualModule1(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm3d, dropout=0.1):
        super(MutualModule1, self).__init__()
        self.dim = dim

        self.gcn = CascadeGCNet(dim, loop=3)

        self.pred0 = nn.Conv2d(self.dim, 3, kernel_size=1)  # predicted contour is used for contour-region mutual sub-module

        self.pred1_ = nn.Conv2d(self.dim, 3, kernel_size=1)  # region prediction

        # conv region feature afger reproj
        self.conv0 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))

        # self.ecg = ECGraphNet(self.dim, BatchNorm, dropout)

    def forward(self, region_x, region_graph, assign, contour_x):
        b, c, h, w, d = contour_x.shape

        contour = self.pred0(contour_x)

        region_graph = self.gcn(region_graph)
        n_region_x = region_graph.bmm(assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))

        region_x = region_x + n_region_x  # raw-feature with residual

        region_x = region_x + contour_x
        region_x = self.conv1(region_x)


        region = self.pred1_(region_x)

        return  contour, region
    


class AdaptiveGraphEnhanceModule(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm3d, dim=64, out_dim=64, num_clusters=8, dropout=0.1):
        super(AdaptiveGraphEnhanceModule, self).__init__()

        self.dim = dim

        self.contour_proj0 = GraphNet3D(node_num=num_clusters, dim=self.dim, normalize_input=False)
        self.region_proj0 = GraphNet3D(node_num=num_clusters, dim=self.dim, normalize_input=False)

        self.contour_conv = nn.Sequential(BasicConv2d(self.dim, self.dim, nn.BatchNorm2d, kernel_size=1, padding=0))
        self.contour_conv[0].reset_params()

        self.region_conv1 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv1[0].reset_params()

        self.region_conv2 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.region_conv2[0].reset_params()
        
        self.gcn = GraphConvNet(dim, dim)
        
        self.e2r = MutualModule1(self.dim, BatchNorm, dropout)
        
        self.mdf = ModifiedMutualModule(self.dim,num_heads = 1, out_dim=out_dim)

    def forward(self, contour_x, region_x):

        region_graph, region_assign = self.region_proj0(region_x)
        contour_graph, contour_assign = self.contour_proj0(contour_x)

        contour_graph = self.contour_conv(contour_graph.unsqueeze(3)).squeeze(3)

        region_graph_T = region_graph.permute(0, 2, 1)  # (B, D, N)
        E = region_graph_T.bmm(contour_graph)         # (B, N, M)
        
        region_graph = self.gcn(region_graph,E)
        contour_graph = self.gcn(contour_graph,E)

        region = self.mdf(region_graph, contour_graph, region_assign, contour_assign,region_x,contour_x)

        return   region

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_layers = nn.ModuleList([
            nn.Linear(self.in_dim, self.out_dim ) 
        ])
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.output_proj = nn.Linear(out_dim, out_dim)  # 合并多头注意力

    def forward(self, x, adj):
        # 假设输入 x 的形状为 [batch, nodes, in_dim]，adj 的形状为 [batch, nodes, nodes]
        x = x.permute(0, 2, 1)  # 转置后形状为 [batch, nodes, in_dim]
        batch_size, num_nodes, in_dim = x.shape
        attn_heads = []
        for attn_layer in self.attn_layers:
            x_flat = x.reshape(-1, self.in_dim)
            q = k = attn_layer(x_flat)  # 形状：[batch, nodes, out_dim // num_heads]
            q = q.view(batch_size, num_nodes, -1)  # 恢复形状 [batch, nodes, out_dim // num_heads]
            k = k.view(batch_size, num_nodes, -1)
            scores = torch.bmm(q, k.transpose(1, 2))  # 点积注意力：[batch, nodes, nodes]
            scores = scores / torch.sqrt(torch.tensor(scores.size(-1), dtype=torch.float32))
            scores = F.softmax(scores, dim=-1)
            attn_heads.append(torch.bmm(scores, q))  # 形状：[batch, nodes, out_dim // num_heads]


        # 拼接多头输出并投影
        x = torch.cat(attn_heads, dim=-1)  # 形状：[batch, nodes, out_dim]
 
        x = self.output_proj(x)
        # 转置回 [batch, out_dim, nodes]
        return x.permute(0, 2, 1)
    
    
class ModifiedMutualModule(nn.Module):
    def __init__(self, dim, BatchNorm=nn.BatchNorm3d,num_heads=4, dropout=0.1, out_dim=3):
        super(ModifiedMutualModule, self).__init__()
        self.dim = dim
        
        self.gat = GATLayer(dim, dim, num_heads=num_heads, dropout=dropout)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, padding=0), nn.BatchNorm2d(dim), nn.ReLU())
        self.conv0 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.conv1 = nn.Sequential(BasicConv3d(self.dim, self.dim, BatchNorm, kernel_size=1, padding=0))
        self.pred = nn.Conv3d(self.dim, out_dim, kernel_size=1)
        self.gcn = CascadeGCNet(dim, loop=3)

    def compute_similarity(self, contour_graph, region_graph):
        # 余弦相似性计算
        contour_graph = contour_graph.permute(0, 2, 1)  # 转换为 [batch, nodes, in_dim]
        region_graph = region_graph.permute(0, 2, 1)
        contour_norm = F.normalize(contour_graph, dim=-1)
        region_norm = F.normalize(region_graph, dim=-1)
        similarity = torch.bmm(contour_norm, region_norm.transpose(1, 2))
        return similarity

 
    def generate_adjacency(self, similarity, threshold=0.8):
        # 基于相似性添加跨图边
        adj = (similarity > threshold).float()
        adj = (adj + adj.transpose(1, 2)) / 2  # 确保对称性
        
        return adj   
    def generate_unified_adjacency(self, adj, cross_similarity, threshold=0.8):
        # adj: [B, N, N], 原始图的邻接矩阵
        # cross_similarity: [B, N, N], 跨图节点相似性矩阵

        # 创建 zero 矩阵存放 unified_adj
        B, N, _ = adj.shape
        unified_adj = torch.zeros(B, 2 * N, 2 * N, device=adj.device)

        # 填充邻接矩阵
        unified_adj[:, :N, :N] = adj  # contour_graph 的自关系
        unified_adj[:, N:, N:] = adj  # region_graph 的自关系

        # 计算跨图关系
        cross_adj = (cross_similarity > threshold).float()

        # 填充跨图的关系
        unified_adj[:, :N, N:] = cross_adj
        unified_adj[:, N:, :N] = cross_adj.transpose(1, 2)

        # 确保对称性
        unified_adj = (unified_adj + unified_adj.transpose(1, 2)) / 2
        return unified_adj
        
    
    def forward(self, region_graph, contour_graph, region_assign, contour_assign,region_x, contour_x):

        # 计算相似性矩阵
        similarity = self.compute_similarity(contour_graph, region_graph)
        adj = self.generate_adjacency(similarity)

        # 使用GAT计算聚合特征
        contour_graph = self.gat(contour_graph, adj)  # GAT邻接矩阵，用动态构造方式生成
        region_graph = self.gat(region_graph, adj)
        
        # 更新图
        unified_graph = torch.cat([contour_graph, region_graph], dim=-1)
        unified_adj = self.generate_unified_adjacency(adj, similarity)
        unified_graph = self.gat(unified_graph, unified_adj)
 
        region_part = unified_graph[:, :, 8:]
        weights = torch.softmax(torch.cat((contour_assign, region_assign), dim=1), dim=1)
        contour_weight = weights[:, :8, :]  # 对应 contour 的权重
        region_weight = weights[:, 8:, :]  # 对应 region 的权重
        fused_assign = contour_weight * contour_assign + region_weight * region_assign
        # 特征回投    
        n_region_x = region_part.bmm(fused_assign)
        n_region_x = self.conv0(n_region_x.view(region_x.size()))
        
        region_x = region_x + n_region_x  # raw-feature with residual
        region_x = region_x + contour_x
        region_x = self.conv1(region_x)

        region = self.pred(region_x)
        
        return region


def count_parameters(model: nn.Module):
    """返回模型的总参数量和可训练参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_mutualnet3d_forward():
    """
    测试 3D 版 MutualNet 的前向和反向传播，验证输出批大小和体素维度一致，并打印参数量。
    """
    # 参数配置
    batch_size = 2
    dim = 64
    D, H, W = 40, 48, 32 # 体数据的深度、高和宽

    # 随机生成 3D 输入特征
    contour_x = torch.randn(batch_size, dim, D, H, W, requires_grad=True)
    region_x = torch.randn(batch_size, dim, D, H, W, requires_grad=True)

    # 实例化并测试模型
    model = MutualNet(BatchNorm=nn.BatchNorm3d, dim=dim, num_clusters=8, dropout=0.1)
    model.train()

    # 前向传播
    region_out = model(contour_x, region_x)

    # 打印形状信息
    print(f"输入 region_x 形状: {region_x.shape}")
    print(f"输出 region_out 形状: {region_out.shape}")

    # 校验批大小和体素维度
    assert region_out.shape[0] == batch_size, \
        f"输出 batch size {region_out.shape[0]} 应与输入 {batch_size} 一致"
    assert region_out.shape[2:] == (D, H, W), \
        f"输出体素维度 {region_out.shape[2:]} 应与输入 {(D, H, W)} 一致"

    # 打印输出通道数
    C_out = region_out.shape[1]
    print(f"输出通道数: {C_out}")

    # 反向传播测试
    loss = region_out.abs().mean()
    loss.backward()
    assert contour_x.grad is not None, "contour_x 未计算梯度"
    assert region_x.grad is not None,  "region_x 未计算梯度"
    print("梯度计算正常，测试通过！")

    # 打印模型参数量
    total, trainable = count_parameters(model)
    print(f"模型总参数量: {total} ({total/1e6:.2f}M)")  
    print(f"可训练参数量: {trainable} ({trainable/1e6:.2f}M)")


if __name__ == "__main__":
    test_mutualnet3d_forward()
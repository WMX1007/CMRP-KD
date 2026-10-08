import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

# ==========================================
# 1. 定义 DIST 相关的辅助函数 (Pearson & Relations)
# ==========================================


def cosine_similarity(a, b, eps=1e-8):
    # 计算余弦相似度
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    # Pearson = 去中心化后的 Cosine
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    # 类间关系：基于行（样本）计算 Pearson
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    # 类内关系：基于列（类别）计算 Pearson (注意这里的转置)
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, tau=1.0, beta=1.0, gamma=1.0):
        """
        Args:
            tau: temperature
            beta: inter-class (row Pearson) 权重
            gamma: intra-class (col Pearson) 权重
        消融用法：
            beta=1, gamma=0 → 只用 L_inter
            beta=0, gamma=1 → 只用 L_intra
            beta=1, gamma=1 → 完整 CMRD
        """
        super(DIST, self).__init__()
        self.tau = tau
        self.beta = beta
        self.gamma = gamma

    def forward(self, z_s, z_t):
        # z_s, z_t 是 Raw Logits (未经过 Softmax 和 Temperature 除法)

        # 1. 内部进行温度缩放和 Softmax
        y_s = (z_s / self.tau).softmax(dim=1)
        y_t = (z_t / self.tau).softmax(dim=1)

        # 2. 计算 Inter / Intra；权重为 0 时直接置零，避免无用计算与梯度
        inter_loss = self.tau**2 * inter_class_relation(y_s, y_t) if self.beta > 0 else z_s.new_zeros(())
        intra_loss = self.tau**2 * intra_class_relation(y_s, y_t) if self.gamma > 0 else z_s.new_zeros(())

        # 3. 加权求和
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return kd_loss

# ==========================================
# 2. 修改后的 KDCriterion
# ==========================================


class KDCriterion:
    def __init__(self, **kwargs) -> None:
        args = SimpleNamespace(**kwargs)
        self.args = args
        self.criterion_aligned_img_kd = args.img_criterion
        self.kd_weight = args.kd_weight
        self.temperature = args.temperature
        # inter/intra 权重，用于消融；默认 1.0/1.0 等价于完整 CMRD
        self.beta = float(getattr(args, "beta", 1.0))
        self.gamma = float(getattr(args, "gamma", 1.0))
        # 初始化 DIST Loss 模块
        self.dist_loss_fn = DIST(tau=self.temperature, beta=self.beta, gamma=self.gamma)
        logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(100.0), requires_grad=False)
        self.logit_scale = logit_scale.exp()

    def __call__(self, inputs):
        hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp = inputs

        # 1. Image Loss (保持原样)
        img_loss = self.criterion_aligned_img_kd(hidden_features, aligned_img)

        # 2. 计算 Raw Logits (关键修改！！！)
        # [原代码]: logits = scale * sim / self.temperature
        # [新代码]: logits = scale * sim
        student_nlp_logits_raw = self.logit_scale * hidden_features @ aligned_nlp.T
        teacher_nlp_logits_raw = self.logit_scale * clip_img_features @ clip_nlp_features.T

        # ==========================================
        # 🟢 [DEBUG] 插入检查代码
        # ==========================================
        '''if np.random.rand() < 0.05:  # 防止刷屏，只打印 5% 的 step
            print("\n" + "="*30)
            print(f"--- Step Check ---")

            # 1. 检查 Logit Scale (CLIP 的关键参数)
            print(f"[Scale] Logit Scale: {self.logit_scale.item():.4f}")

            # 2. 检查 Logits 的数值范围 (非常重要！)
            # 如果 Max > 100，说明分布极度尖锐；如果 Max < 10，说明分布太平滑
            s_min, s_max, s_mean = student_nlp_logits_raw.min().item(), student_nlp_logits_raw.max().item(), student_nlp_logits_raw.mean().item()
            t_min, t_max, t_mean = teacher_nlp_logits_raw.min().item(), teacher_nlp_logits_raw.max().item(), teacher_nlp_logits_raw.mean().item()

            print(f"[Logits] Student: Min={s_min:.2f}, Max={s_max:.2f}, Mean={s_mean:.2f}")
            print(f"[Logits] Teacher: Min={t_min:.2f}, Max={t_max:.2f}, Mean={t_mean:.2f}")

            # 3. 检查 Softmax 后的最大置信度 (Confidence)
            # 这能告诉你 Temperature 是否设置得当
            with torch.no_grad():
                # 模拟 DIST 内部的计算
                s_prob = (student_nlp_logits_raw /self.temperature).softmax(dim=1)
                t_prob = (teacher_nlp_logits_raw /self.temperature).softmax(dim=1)

                s_conf = s_prob.max(dim=1)[0].mean().item()  # 平均最高置信度
                t_conf = t_prob.max(dim=1)[0].mean().item()

            print(f"[Conf]  Student Avg Max Prob (T={self.temperature}): {s_conf:.4f}")
            print(f"[Conf]  Teacher Avg Max Prob (T={self.temperature}): {t_conf:.4f}")

            # 4. 检查 Loss 的原始量级 (在乘权重之前)
            loss_debug = self.dist_loss_fn(
                student_nlp_logits_raw, teacher_nlp_logits_raw)
            print(f"[Loss]  Raw DIST Loss: {loss_debug.item():.4f}")
            print("="*30 + "\n")'''
        # ==========================================


        # 3. 计算 DIST Loss (Inter + Intra)
        kd_loss = self.dist_loss_fn(
            student_nlp_logits_raw, teacher_nlp_logits_raw)

        # 4. 损失缩放 (保留原逻辑)
        #kd_loss = kd_loss * self.args.class_num / 2
        kd_loss = kd_loss * self.kd_weight

        return img_loss, kd_loss

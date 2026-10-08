import torch
from torch import nn
import torchvision.models as models

import open_clip

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)

class TeacherStudent(nn.Module):
    def __init__(self, teacher, student, data_attributes, use_teacher=True):
        super(TeacherStudent, self).__init__()
        self.teacher, self.align, self.frozen_nlp_features = None, None, None
        self.data_attributes = data_attributes
        self.student = StudentNet(student, data_attributes.class_num, use_teacher)
        if use_teacher:
            # 获取设备信息，兼容不同类型的模型
            try:
                # 尝试从 resnet 获取设备（原有的 CNN 模型）
                device = next(self.student.model.resnet.parameters()).device
            except AttributeError:
                # 如果没有 resnet 属性，从其他地方获取设备（Swin 等模型）
                device = next(self.student.model.parameters()).device
            
            self.teacher = TeacherNet(teacher)
            self.align = AlignNet(self.teacher.last_features_dim, self.student.num_features)
            self.frozen_nlp_features = self.get_frozen_nlp_features(data_attributes)
    
    def get_frozen_nlp_features(self, attributes):
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        text_tokens = self.teacher.tokenizer([prompt_tmpl.format(word) for word in classes_list])
        nlp_features = self.teacher.encode_text(text_tokens).detach()
        return feature_norm(nlp_features)
    
    def forward(self, x):
        if self.teacher:
            clip_img_features = self.teacher(x)
            frozen_nlp_features = self.frozen_nlp_features.to(clip_img_features.device)
            aligned_img, aligned_nlp = self.align(clip_img_features, frozen_nlp_features)
            hidden_features, out = self.student(x)
            return hidden_features, out, clip_img_features, frozen_nlp_features, aligned_img, aligned_nlp
        return self.student(x)



class TeacherNet(nn.Module):
    def __init__(self, teacher):
        super(TeacherNet, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(teacher.arch, pretrained=teacher.pretrained)
        self.model.requires_grad_(False)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(teacher.arch)
        self.last_features_dim = self.model.transformer.resblocks[-1].mlp.c_proj.out_features

    def encode_image(self, x):
        return self.model.encode_image(x)

    def encode_text(self, x):
        return self.model.encode_text(x)

    def forward(self, x):
        with torch.no_grad():
            clip_img_features = self.encode_image(x).detach()
        clip_img_features = feature_norm(clip_img_features)
        return clip_img_features
    

class AlignNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(AlignNet, self).__init__()

        self.align_img_layer = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.ReLU(), 
            nn.Linear(out_features, out_features)
            )
        self.align_nlp_layer = nn.Sequential(
            nn.Linear(in_features, out_features), 
            nn.ReLU(), 
            nn.Linear(out_features, out_features)
        )
    
    def forward(self, x, clip_nlp_features):
        align_img = self.align_img_layer(x)
        align_nlp = self.align_nlp_layer(clip_nlp_features)
        return feature_norm(align_img), feature_norm(align_nlp)



class StudentNet(nn.Module):
    def __init__(self, student, class_num, use_teacher=True):
        super(StudentNet, self).__init__()
        self.use_teacher = use_teacher
        self.num_features = None
        if self.use_teacher:
            # 检查是否是 Swin Transformer
            if 'swin' in student.arch.lower():
                # 检查是否是 OpenCLIP 支持的 Swin 模型
                if student.arch in ['swin_base_patch4_window7_224']:
                    # 使用 OpenCLIP 加载 Swin 模型
                    origin_model, _, _ = open_clip.create_model_and_transforms(student.arch, pretrained='openai')
                    self.model = ModifiedSwin(origin_model, class_num)
                    self.num_features = self.model.num_features
                else:
                    # 使用 timm 加载其他 Swin 模型（如 swin_tiny）
                    import timm
                    # 将自定义名称映射到 timm 的标准名称
                    timm_name_map = {
                        'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224'
                    }
                    timm_name = timm_name_map.get(student.arch, student.arch)
                    origin_model = timm.create_model(timm_name, pretrained=True)
                    self.model = ModifiedTimmSwin(origin_model, class_num)
                    self.num_features = self.model.num_features
            else:
                # 原有的 torchvision 模型加载方式
                origin_model = models.__dict__[student.arch](pretrained=True)
                self.model  = ModifiedResNet(origin_model, class_num)
                self.num_features = self.model.num_features
        else:
            if 'swin' in student.arch.lower():
                # 使用 OpenCLIP 加载 Swin 模型
                self.model, _, _ = open_clip.create_model_and_transforms(student.arch, pretrained='openai')
                # 修改最后的分类层
                try:
                    num_features = self.model.head.in_features
                    self.model.head = nn.Linear(num_features, class_num)
                except:
                    # 如果没有 head 属性，尝试其他可能的属性
                    num_features = self.model.transformer.resblocks[-1].mlp.c_proj.out_features
                    self.model.head = nn.Linear(num_features, class_num)
            else:
                # 原有的 torchvision 模型加载方式
                self.model = models.__dict__[student.arch](pretrained=True)
                try:
                    num_features = self.model.fc.in_features
                    self.model.fc = nn.Linear(num_features, class_num)
                except:
                    num_features = self.model.classifier[1].in_features
                    self.model.classifier[1] = nn.Linear(num_features, class_num)

    def forward(self, x):
        if self.use_teacher:
            hidden_features, out = self.model(x)
            return feature_norm(hidden_features), out
        out = self.model(x)
        return out


class ModifiedResNet(torch.nn.Module):
    def __init__(self, origin_model, classnum):
        super(ModifiedResNet, self).__init__()
        self.resnet = origin_model
        
        try:
            num_features = origin_model.fc.in_features
            self.resnet.fc = nn.Identity()
        except:
            num_features = origin_model.classifier[1].in_features
            self.resnet.classifier  = nn.Identity()           
        self.linear_cls = nn.Linear(num_features, classnum)
        self.num_features = num_features

    def forward(self, x):
        hidden_features = self.resnet(x)
        out = self.linear_cls(hidden_features)

        return hidden_features, out


class ModifiedSwin(torch.nn.Module):
    def __init__(self, origin_model, classnum):
        super(ModifiedSwin, self).__init__()
        self.swin = origin_model
        
        # 获取 Swin 模型的特征维度
        # 对于 OpenCLIP 的 Swin 模型，通常是 visual.head 或 transformer 的输出维度
        try:
            # 尝试获取 visual encoder 的输出维度
            if hasattr(origin_model, 'visual'):
                if hasattr(origin_model.visual, 'head'):
                    num_features = origin_model.visual.head.in_features
                    origin_model.visual.head = nn.Identity()
                else:
                    # 如果没有 head，使用 transformer 的输出维度
                    num_features = origin_model.visual.trunk.head.in_features
                    origin_model.visual.trunk.head = nn.Identity()
            else:
                # 直接是 vision transformer
                num_features = origin_model.head.in_features
                origin_model.head = nn.Identity()
        except:
            # 备用方案：使用常见的 Swin Base 维度
            num_features = 1024  # Swin Base 的默认输出维度
            
        self.linear_cls = nn.Linear(num_features, classnum)
        self.num_features = num_features

    def forward(self, x):
        # 获取 Swin 的特征
        if hasattr(self.swin, 'visual'):
            hidden_features = self.swin.visual(x)
        else:
            hidden_features = self.swin(x)
        
        out = self.linear_cls(hidden_features)
        return hidden_features, out


class ModifiedTimmSwin(torch.nn.Module):
    def __init__(self, origin_model, classnum):
        super(ModifiedTimmSwin, self).__init__()
        self.swin = origin_model
        
        # 获取 timm Swin 模型的特征维度
        try:
            if hasattr(origin_model, 'head'):
                num_features = origin_model.head.in_features
                origin_model.head = nn.Identity()
            elif hasattr(origin_model, 'classifier'):
                num_features = origin_model.classifier.in_features
                origin_model.classifier = nn.Identity()
            else:
                # 备用方案
                num_features = 768  # Swin Tiny 的默认输出维度
        except:
            num_features = 768  # Swin Tiny 的默认输出维度
            
        # 添加全局平均池化层，将 4D 特征图转换为 2D 特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_cls = nn.Linear(num_features, classnum)
        self.num_features = num_features

    def forward(self, x):
        hidden_features = self.swin(x)
        
        # 检查特征维度并处理
        if len(hidden_features.shape) == 4:
            # 如果是 4D 特征图 [B, H, W, C] 或 [B, C, H, W]
            if hidden_features.shape[-1] == self.num_features:
                # [B, H, W, C] 格式，转换为 [B, C, H, W]
                hidden_features = hidden_features.permute(0, 3, 1, 2)
            # 应用全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
            hidden_features = self.global_pool(hidden_features)
            # 展平为 [B, C]
            hidden_features = hidden_features.flatten(1)
        elif len(hidden_features.shape) == 3:
            # 如果是 3D 特征 [B, N, C]，取平均
            hidden_features = hidden_features.mean(dim=1)
        
        out = self.linear_cls(hidden_features)
        return hidden_features, out





if __name__ == "__main__":
    _ = TeacherStudent()

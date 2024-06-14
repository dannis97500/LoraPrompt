import torch
import torch.nn as nn
import copy

from models.clip_lora2.prompt_learner import cfgc, load_clip_to_cpu, TextEncoder, PromptLearner
from utils.class_names import core50_classnames, domainnet_classnames, cddb_classnames
from peft import LoraConfig, get_peft_model

class SliNetLora(nn.Module):

    def __init__(self, args):
        super(SliNetLora, self).__init__()
        self.cfg = cfgc()
        clip_model = load_clip_to_cpu(self.cfg)
        self.clip_model = clip_model

        lora_config = LoraConfig(
            r=8,                # Rank of the LoRA decomposition
            lora_alpha=16,      # Scaling factor for LoRA
            target_modules=["out_proj"],  # Target the attention modules
            lora_dropout=0.1,   # Dropout rate for LoRA
        )
        
        self.image_encoder = clip_model.visual
        # self.image_encoder = get_peft_model(copy.deepcopy(self.image_encoder), lora_config)
        self.lora_image_model_pool = nn.ModuleList([
            get_peft_model(copy.deepcopy(self.image_encoder), lora_config)
            for i in range(args["total_sessions"])
        ])
        
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.class_num = 1
        if args["dataset"] == "cddb":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(cddb_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 2
        elif args["dataset"] == "domainnet":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(domainnet_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 345
        elif args["dataset"] == "core50":
            self.classifier_pool = nn.ModuleList([
                PromptLearner(self.cfg, list(core50_classnames.values()), self.clip_model)
                for i in range(args["total_sessions"])
            ])
            self.class_num = 50
        else:
            raise ValueError('Unknown datasets: {}.'.format(args["dataset"]))

        self.prompt_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], args["prompt_length"], bias=False)
            for i in range(args["total_sessions"])
        ])


        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.output_dim

    def extract_vector(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features

    def forward(self, image):
        logits = []
        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.lora_image_model_pool[self.numtask-1](image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.classifier_pool[self.numtask-1]
        tokenized_prompts = prompts.tokenized_prompts
        text_features = self.text_encoder(prompts(), tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits.append(logit_scale * image_features @ text_features.t())
        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features
        }

    def interface(self, image, selection):
        # instance_batch = torch.stack([i.weight for i in self.prompt_pool], 0)[selection, :, :]
        results = []
        # 遍历数据和索引
        for data, idx in zip(image.type(self.dtype), selection):
            data = data.unsqueeze(0)
            # 根据索引从模型池中获取模型
            output = self.lora_image_model_pool[idx](data)
            # 将输出结果添加到结果列表中
            results.append(output)
        image_features = torch.stack(results).squeeze()
        
        # image_features = self.lora_image_model_pool[selection](image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = []
        for prompt in self.classifier_pool:
            tokenized_prompts = prompt.tokenized_prompts
            text_features = self.text_encoder(prompt(), tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits.append(logit_scale * image_features @ text_features.t())
        logits = torch.cat(logits,1)
        selectedlogit = []
        for idx, ii in enumerate(selection):
            selectedlogit.append(logits[idx][self.class_num*ii:self.class_num*ii+self.class_num])
        selectedlogit = torch.stack(selectedlogit)
        return selectedlogit


    def update_fc(self, nb_classes):
        self.numtask +=1

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

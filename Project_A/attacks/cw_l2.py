import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
import time
import json

class CWL2Attack:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        targeted: bool = True,
        confidence: float = 0.0,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        binary_search_steps: int = 9,
        initial_const: float = 0.001,
        abort_early: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0
    ):
        self.model = model
        self.device = device
        self.targeted = targeted
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.abort_early = abort_early
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def _margin_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = logits.size(0)
        target_logits = logits[torch.arange(batch_size), target]
        
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), target] = False
        other_logits = logits[mask].reshape(batch_size, -1)
        max_other_logits = other_logits.max(dim=1)[0]
        
        loss = torch.max(max_other_logits - target_logits, 
                       -torch.tensor(self.confidence, device=self.device))
        return loss
    
    def _tanh_transform(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (torch.tanh(w) + 1)
    
    def attack_batch(self, images: torch.Tensor, labels: torch.Tensor, 
                    target_classes: Optional[torch.Tensor] = None) -> Tuple[List[Dict], Optional[torch.Tensor]]:
        self.model.eval()
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if target_classes is None:
            target_classes = torch.randint(0, 10, (images.size(0),), device=self.device)
            mask = target_classes == labels
            target_classes[mask] = (target_classes[mask] + 1) % 10
        else:
            target_classes = target_classes.to(self.device)
        
        results = []
        adv_images = []
        for i in range(images.size(0)):
            result, adv_img = self._attack_single(images[i:i+1], labels[i:i+1], target_classes[i:i+1], i)
            results.append(result)
            if adv_img is not None:  # 只添加成功的对抗样本
                adv_images.append(adv_img)
        
        # 返回结果和对抗样本（如果有）
        if adv_images:
            return results, torch.cat(adv_images, dim=0)
        else:
            return results, None

    # def attack_batch(self, images: torch.Tensor, labels: torch.Tensor, 
    #                 target_classes: Optional[torch.Tensor] = None) -> List[Dict]:
    #     self.model.eval()
    #     images = images.to(self.device)
    #     labels = labels.to(self.device)
        
    #     if target_classes is None:
    #         target_classes = torch.randint(0, 10, (images.size(0),), device=self.device)
    #         mask = target_classes == labels
    #         target_classes[mask] = (target_classes[mask] + 1) % 10
    #     else:
    #         target_classes = target_classes.to(self.device)
        
    #     results = []
    #     adv_images = []
    #     for i in range(images.size(0)):
    #         result, adv_img = self._attack_single(images[i:i+1], labels[i:i+1], target_classes[i:i+1], i)
    #         results.append(result)
    #         adv_images.append(adv_img)
        
    #     return results, torch.cat(adv_images) if adv_images else None
    
    def _attack_single(self, image: torch.Tensor, true_label: torch.Tensor, 
                  target_class: torch.Tensor, idx: int) -> Tuple[Dict, Optional[torch.Tensor]]:
        import time
        start_time = time.time()
        TIMEOUT = 300  # 5分钟超时
        
        best_l2 = float('inf')
        best_adv = None
        found_adv = False
        best_pred = None  # 新增：保存最佳预测
        
        c = self.initial_const
        lower_bound = 0
        upper_bound = 1e10
        
        for search_step in range(self.binary_search_steps):
            w = torch.arctanh(2 * image - 1).detach().requires_grad_(True)
            optimizer = torch.optim.Adam([w], lr=self.learning_rate)
            
            for iteration in range(self.max_iterations):
                if time.time() - start_time > TIMEOUT:
                    print(f"  ⚠️ Sample {idx} timeout after {TIMEOUT}s")
                    return {
                        'image_idx': idx,
                        'true_label': true_label.item(),
                        'target_label': target_class.item(),
                        'pred_label': None,
                        'success': False,
                        'l2_distance': float('inf'),
                        'best_c': c,
                        'time_seconds': time.time() - start_time,
                        'confidence': self.confidence
                    }, None               

                optimizer.zero_grad()
                x_adv = self._tanh_transform(w, image)
                
                logits = self.model(x_adv)
                margin_loss = self._margin_loss(logits, target_class)
                l2_dist = torch.sum((x_adv - image) ** 2)
                
                loss = l2_dist + c * margin_loss
                loss.backward()
                optimizer.step()
                
                if margin_loss.item() <= 0 and l2_dist.item() < best_l2:
                    best_l2 = l2_dist.item()
                    best_adv = x_adv.detach().clone()
                    best_pred = logits.argmax(dim=1).item()  # 保存预测
                    found_adv = True
                      

            if found_adv:
                upper_bound = min(upper_bound, c)
                c = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, c)
                if upper_bound < 1e10:
                    c = (lower_bound + upper_bound) / 2
                else:
                    c = c * 10
        
        if best_adv is not None:
            # 使用保存的best_pred
            return {
                'image_idx': idx,
                'true_label': true_label.item(),
                'target_label': target_class.item(),
                'pred_label': best_pred,  # 使用保存的预测
                'success': best_pred == target_class.item(),
                'l2_distance': best_l2 ** 0.5,
                'best_c': c,
                'time_seconds': time.time() - start_time,
                'confidence': self.confidence
            }, best_adv
        else:
            return {
                'image_idx': idx,
                'true_label': true_label.item(),
                'target_label': target_class.item(),
                'pred_label': None,
                'success': False,
                'l2_distance': float('inf'),
                'best_c': c,
                'time_seconds': time.time() - start_time,
                'confidence': self.confidence
            }, None
        
        

    # def _attack_single(self, image: torch.Tensor, true_label: torch.Tensor, 
    #                   target_class: torch.Tensor, idx: int) -> Dict:
    #     start_time = time.time()
        
    #     best_l2 = float('inf')
    #     best_adv = None
    #     found_adv = False
        
    #     c = self.initial_const
    #     lower_bound = 0
    #     upper_bound = 1e10
        
    #     for search_step in range(self.binary_search_steps):
    #         w = torch.arctanh(2 * image - 1).detach().requires_grad_(True)
    #         optimizer = torch.optim.Adam([w], lr=self.learning_rate)
            
    #         for iteration in range(self.max_iterations):
    #             optimizer.zero_grad()
    #             x_adv = self._tanh_transform(w, image)
                
    #             logits = self.model(x_adv)
    #             margin_loss = self._margin_loss(logits, target_class)
    #             l2_dist = torch.sum((x_adv - image) ** 2)
                
    #             loss = l2_dist + c * margin_loss
    #             loss.backward()
    #             optimizer.step()
                
    #             if margin_loss.item() <= 0 and l2_dist.item() < best_l2:
    #                 best_l2 = l2_dist.item()
    #                 best_adv = x_adv.detach().clone()
    #                 found_adv = True
            
    #         if found_adv:
    #             upper_bound = min(upper_bound, c)
    #             c = (lower_bound + upper_bound) / 2
    #         else:
    #             lower_bound = max(lower_bound, c)
    #             if upper_bound < 1e10:
    #                 c = (lower_bound + upper_bound) / 2
    #             else:
    #                 c = c * 10
        
        # if best_adv is not None:
        #     with torch.no_grad():
        #         pred = self.model(best_adv).argmax(dim=1).item()
            
        #     return {
        #         'image_idx': idx,
        #         'true_label': true_label.item(),
        #         'target_label': target_class.item(),
        #         'pred_label': pred,
        #         'success': pred == target_class.item(),
        #         'l2_distance': best_l2 ** 0.5,
        #         'best_c': c,
        #         'time_seconds': time.time() - start_time,
        #         'confidence': self.confidence
        #     }
        # else:
        #     return {
        #         'image_idx': idx,
        #         'true_label': true_label.item(),
        #         'target_label': target_class.item(),
        #         'pred_label': None,
        #         'success': False,
        #         'l2_distance': float('inf'),
        #         'best_c': c,
        #         'time_seconds': time.time() - start_time,
        #         'confidence': self.confidence
        #     }
            # if best_adv is not None:
            #     return {
            #         'image_idx': idx,
            #         'true_label': true_label.item(),
            #         'target_label': target_class.item(),
            #         'pred_label': pred,
            #         'success': True,
            #         'l2_distance': best_l2 ** 0.5,
            #         'best_c': c,
            #         'time_seconds': time.time() - start_time,
            #         'confidence': self.confidence
            #     }, best_adv
            # else:
            #     return {
            #         'image_idx': idx,
            #         'true_label': true_label.item(),
            #         'target_label': target_class.item(),
            #         'pred_label': None,
            #         'success': False,
            #         'l2_distance': float('inf'),
            #         'best_c': c,
            #         'time_seconds': time.time() - start_time,
            #         'confidence': self.confidence
            #     }, None


def save_attack_results(results: List[Dict], output_path: str):
    import csv
    csv_path = output_path.replace('.json', '.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_idx', 'true_label', 'target_label', 
                                              'pred_label', 'success', 'l2_distance', 
                                              'best_c', 'time_seconds', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {csv_path} and {output_path}")
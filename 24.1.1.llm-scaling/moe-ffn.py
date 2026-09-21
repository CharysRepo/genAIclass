import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))


class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        return topk_weights, topk_indices


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        weights, indices = self.router(x)

        combined_output = torch.zeros_like(x)        
        for expert_idx in range(self.num_experts):
            token_mask, topk_rank = torch.where(indices == expert_idx)
            if token_mask.numel() == 0:
                continue
            expert_inputs = x[token_mask]
            expert_outputs = self.experts[expert_idx](expert_inputs)
            gating_weights = weights[token_mask, topk_rank].unsqueeze(-1)
            combined_output[token_mask] += gating_weights * expert_outputs
            
        return combined_output


class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w_2(self.act(self.w_1(x)))


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on target device: {device.type.upper()}\n")

    # Hyperparameters for comparison
    seq_len = 2048 
    d_model = 512
    d_ff = 2048
    num_experts = 8
    top_k = 2           

    # Create networks
    moe_net = MoELayer(d_model, d_ff, num_experts, top_k).to(device)
    std_net = StandardFFN(d_model, d_ff).to(device)

    # Calculate total parameters
    moe_total_params = count_parameters(moe_net)
    std_total_params = count_parameters(std_net)

    # Calculate active parameters per token
    single_expert_params = count_parameters(moe_net.experts[0]) # Count single expert module parameters
    router_params = count_parameters(moe_net.router)
    moe_active_params = router_params + (single_expert_params * top_k)
    std_active_params = std_total_params

    dummy_input = torch.randn(seq_len, d_model, device=device)

    # benchmark moe net
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()    
    for _ in range(200):
        moe_output = moe_net(dummy_input)        
    if device.type == "cuda": 
        torch.cuda.synchronize() 
    moe_time = (time.perf_counter() - start_time) / 200

    # benchmark standard net
    if device.type == "cuda": 
        torch.cuda.synchronize()
    start_time = time.perf_counter()    
    for _ in range(200):
        std_output = std_net(dummy_input)        
    if device.type == "cuda": 
        torch.cuda.synchronize()
    std_time = (time.perf_counter() - start_time) / 200

    print(f"{'Metric':<30} | {'Standard FFN':<15} | {'Mixture of Experts (MoE)':<25}")
    print("-" * 80)
    print(f"{'Total Parameters':<30} | {std_total_params:<15,} | {moe_total_params:<25,}")
    print(f"{'Active Params (per token)':<30} | {std_active_params:<15,} | {moe_active_params:<25,}")
    print(f"{'Avg Execution Time (seconds)':<30} | {std_time:<15.6f} | {moe_time:<25.6f}")
    print("-" * 80)
    
    if device.type == "cuda" and moe_time > std_time:
        print("\nObservation Note:")
        print("You might notice the custom native PyTorch MoE layer runs slower than the Standard FFN on GPU.")
        print("Reason: Sequential python loops (`for expert_idx in...`) cause kernel launch bottlenecks on GPU.")
        print("Solution: Production stacks use specialized CUDA operators (like MegaBlocks or vLLM custom kernels)")
        print("          to execute all experts concurrently in a single fused GPU call.")

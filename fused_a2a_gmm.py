import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl
from torchao.prototype.moe_training.kernels.triton_utils import (
    blockwise_barrier,
    sync_threads,
)
import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem

class FusedAllToAllGMM(torch.autograd.Function):
    # A symmetric memory buffer for exchanging input rows/tokens during forward
    input_sym_mem_buf = None

    # A symmetric memory for exchanging split sizes during both forward and backward
    input_splits_sym_mem_buf = None

    # A symmetric memory buffer holding the grad_output during backward
    grad_out_sym_mem_buf = None

    # Maximum output length (need to be set before use of FusedAllToAllGMM)
    max_output_rows_per_rank = None

    # A preallocated buffer for holding the grad_input, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_buf = None

    # A preallocated buffer for holding the grad_input splits, that can be reused without cudaMalloc/cudaFree each iteration
    grad_input_splits_buf = None

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        input_data: torch.Tensor,
        input_splits: torch.Tensor,
        max_output_rows_per_rank: int,
        expert_weights: torch.Tensor,
        group: dist.ProcessGroup = dist.group.WORLD,
    ):
        """
        Args:
            input: input float8_e4m3fn tensor with data for all ranks concatenated.
            input_splits: input splits of shape (group.world_size,)
            max_output_rows_per_rank: maximum output rows/tokens per rank.
            group: process group to scope the collective.
        """
        # Enable symm mem for the group if not already enabled
        symm_mem.set_backend("NVSHMEM")
        nvshmem_lib = nvshmem.enable_triton()
        if not symm_mem.is_symm_mem_enabled_for_group(group):
            symm_mem.enable_symm_mem_for_group(group)

        FusedAllToAllGMM.max_output_rows_per_rank = max_output_rows_per_rank


        # Initialize sym mem buffer for float8 e4m3 input data (one time only)
        if FusedAllToAllGMM.input_sym_mem_buf is None:
            FusedAllToAllGMM.input_sym_mem_buf = symm_mem.empty(
                FusedAllToAllGMM.max_output_rows_per_rank,
                *input_data.shape[1:],
                dtype=input_data.dtype,
                device=input_data.device,
            )

        # Initialize input splits buffer (one time only)
        if FusedAllToAllGMM.input_splits_sym_mem_buf is None:
            FusedAllToAllGMM.input_splits_sym_mem_buf = symm_mem.empty(
                *input_splits.shape,
                dtype=input_splits.dtype,
                device=input_splits.device,
            )

        # Copy quantized data, and output splits to symm mem buffers
        FusedAllToAllGMM.input_sym_mem_buf.narrow(
            0, 0, input_data.shape[0]
        ).copy_(input_data)

        # Copy input splits to symm mem buffer
        FusedAllToAllGMM.input_splits_sym_mem_buf.copy_(input_splits)

        # Allocate buffers for output data, and splits.
        output = input_data.new_empty(
            FusedAllToAllGMM.max_output_rows_per_rank, *input_data.shape[1:]
        )
        output_splits = torch.empty_like(input_splits)

        # Shuffle input to output
        triton_fused_a2a_gmm(
            FusedAllToAllGMM.input_sym_mem_buf,
            FusedAllToAllGMM.input_splits_sym_mem_buf,
            output,
            output_splits,
            expert_weights,
            group=group,
        )

        # Saving for backward: output splits in forward is the input splits in backward
        ctx.group = group
        ctx.input_shape = input_data.shape
        ctx.max_output_rows_per_rank = max_output_rows_per_rank
        ctx.save_for_backward(output_splits, expert_weights)
        return output, output_splits

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output, grad_splits):
        """
        Backward is implemented as a shuffle of the output's gradients to the input.
        Args:
            `grad_output`: output's gradients passed from the downstream.
            `grad_splits`: unused.
        """
        # In backward, _all_to_all_v input is `grad_output`, and output is `grad_input`.
        grad_output_splits, expert_weights = ctx.saved_tensors

        # Initialize grad_output sym mem buffer (one time only)
        if FusedAllToAllGMM.grad_out_sym_mem_buf is None:
            FusedAllToAllGMM.grad_out_sym_mem_buf = symm_mem.empty(
                FusedAllToAllGMM.max_output_rows_per_rank,
                *grad_output.shape[1:],
                dtype=torch.float8_e4m3fn,
                device=grad_output.device,
            )


        # Copy in float8 grad out data to a symm mem buffer
        FusedAllToAllGMM.grad_out_sym_mem_buf.narrow(
            0, 0, grad_output.shape[0]
        ).copy_(grad_output)


        # Copy in splits to symm mem buffer
        FusedAllToAllGMM.input_splits_sym_mem_buf.copy_(grad_output_splits)

        # Allocate buffers for grad_input data, and splits if necessary
        if FusedAllToAllGMM.grad_input_buf is None:
            FusedAllToAllGMM.grad_input_buf = grad_output.new_empty(
                ctx.max_output_rows_per_rank,
                *ctx.input_shape[1:],
            )

        if FusedAllToAllGMM.grad_input_splits_buf is None:
            FusedAllToAllGMM.grad_input_splits_buf = torch.empty_like(
                grad_output_splits
            )

        # TODO: Shuffle gradients back to the input, backward GMMs

        return FusedAllToAllGMM.grad_input_buf, None, None, None


# Alias
fused_a2a_gmm = FusedAllToAllGMM.apply


# Triton launcher function
def _triton_fused_a2a_gmm(
    input: torch.Tensor,
    input_splits: torch.Tensor,
    output: torch.Tensor,
    output_splits: torch.Tensor,
    expert_weights: torch.Tensor,
    group: dist.ProcessGroup = dist.group.WORLD,
    BLOCKS_PER_REMOTE_RANK: int = 32,
    BLOCK_SIZE: int = 16384,
):
    assert input.dim() == 2, f"{input.shape}"
    assert output.dim() == 2, f"{output.shape}"
    assert output.shape[1] == input.shape[1]

    # Prepare symmetric memory managed buffers for input, input_splits
    # - `input` shape (tokens, dim) -> to a sym mem managed buffer of shape (num_ranks, tokens, dim)
    # - `input_splits` shape (num_ranks,) -> to a sym mem managed buffer of shape (num_ranks, num_ranks)`
    input_hdl = symm_mem.rendezvous(input, group=group)
    input_splits_hdl = symm_mem.rendezvous(input_splits, group=group)

    input_ptrs = input_hdl.buffer_ptrs_dev
    input_splits_ptrs = input_splits_hdl.buffer_ptrs_dev
    signal_pad_ptrs = input_hdl.signal_pad_ptrs_dev
    dim = output.shape[1]
    num_blocks = input_hdl.world_size * BLOCKS_PER_REMOTE_RANK

    _triton_fused_a2a_gmm_kernel(num_blocks, 1, 1)](
        input_ptrs,
        input_splits_ptrs,
        output,
        output_splits,
        signal_pad_ptrs,
        dim=dim,
        rank=input_hdl.rank,
        world_size=input_hdl.world_size,
        BLOCKS_PER_REMOTE_RANK=BLOCKS_PER_REMOTE_RANK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
        extern_libs=nvshmem_lib,
    )

    return output


@triton.jit
def _triton_fused_a2a_gmm_kernel(
    input_ptrs,
    input_splits_ptr,
    output_ptr,
    output_splits_ptr,
    signal_pad_ptrs,
    dim: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    BLOCKS_PER_REMOTE_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    sync_threads()

    remote_rank = tl.program_id(0) // BLOCKS_PER_REMOTE_RANK
    block_offset = tl.program_id(0) % BLOCKS_PER_REMOTE_RANK

    # 1. Get input row to read from the given remote rank (to get data coming to this local rank),
    #    and how many rows we're reading.
    # 2. Get the output row offset to write that data to.
    input_row_offset, output_row_offset, num_rows_to_read = _exchange_row_offsets(
        input_splits_ptr,
        rank,
        remote_rank,
        world_size,
    )

    # One thread block per rank will update output_splits
    if block_offset == 0:
        tl.store(output_splits_ptr + remote_rank, num_rows_to_read)

    # Update input and output pointers to point to the specific row we're reading/writing.
    # 1. `input` is symmetric memory managed buffer of shape [num_ranks, tokens, dim].
    #   We increment the ptr by `+remote_rank` along the 0th dim to get to the remote rank ptr,
    #   then increment that ptr by `input_row_offset * dim (stride)` to get the
    #   start offset for this rank's data on that remote rank.
    # 2. `output` is a regular local tensor, we can stride into it as usual.
    input_ptr = (
        tl.load(input_ptrs.to(tl.pointer_type(tl.uint64)) + remote_rank).to(
            tl.pointer_type(tl.bfloat16)
        )
        + input_row_offset * dim
    )
    output_ptr = output_ptr + output_row_offset * dim

    # Copy target region of remote rank input data to our local output buffer.
    total_input_elems_to_read = num_rows_to_read * dim
    num_input_blocks = tl.cdiv(total_input_elems_to_read, BLOCK_SIZE)
    for block_idx in tl.range(num_input_blocks):
        offs = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_input_elems_to_read
        data = tl.load(input_ptr + offs, mask=mask, other=0.0)
        tl.store(output_ptr + offs, data, mask=mask)

    sync_threads()
    blockwise_barrier(signal_pad_ptrs, None, rank, world_size, sem="relaxed")
    return


@triton.jit
def _exchange_row_offsets(
    split_sizes_ptrs,
    local_rank: tl.constexpr,
    remote_rank: tl.constexpr,
    world_size: tl.constexpr,
):
    """
    Returns:
    - `input_offset_for_remote_rank`:
    - `output_offset_for_remote_rank`:
    - `num_rows`:
    """
    # split_sizes_ptr points to 2d tensor of stacked input split size vectors (one per rank). Example:
    # rank 0 = [30, 10, 10, 20]
    # rank 1 = [20, 20, 10, 20]
    split_sizes_ptrs = split_sizes_ptrs.to(tl.pointer_type(tl.uint64))

    # Get pointer to remote rank's input_split_sizes tensor.
    remote_rank_input_splits_ptr = tl.load(split_sizes_ptrs + remote_rank).to(
        tl.pointer_type(tl.int64)
    )

    # num_rows_to_read is the specific number of tokens to read from remote_rank.
    num_rows_to_read = tl.load(remote_rank_input_splits_ptr + local_rank)

    # Calculate starting offset in symm mem buf to read data from remote_rank for this local_rank.
    #
    # Do this by computing prefix sum of remote split offsets prev ranks.
    # Ex. remote_rank split sizes = [10, 20, 30]
    # For local rank 1, masked load = [10, 0, 0]
    # Starting offset = sum([10, 0, 0]) = 10
    offsets = tl.arange(0, world_size)
    remote_split_sizes_prefix = tl.load(
        remote_rank_input_splits_ptr + offsets, mask=offsets < local_rank, other=0
    )
    input_offset_for_remote_rank = tl.sum(remote_split_sizes_prefix)

    # Calculate offset in local output buffer to start writing data to, for data coming from the remote_rank to this local_rank.
    #
    # We add `offsets` arange to get a set of pointers to the start of each row (rank) in the split_sizes matrix.
    # Then, we add the local rank to each pointer, incrementing it colwise to reach the value for this local rank.
    # Each ptrs now all point to how many tokens/rows that device has for local rank.
    #
    # torch equivalent: split_sizes_matrix[:, rank]
    ptr_to_each_rank_split_sizes = tl.load(split_sizes_ptrs + offsets).to(
        tl.pointer_type(tl.int64)
    )
    output_split_sizes_ptrs = ptr_to_each_rank_split_sizes + local_rank
    output_split_sizes = tl.load(
        output_split_sizes_ptrs, mask=offsets < remote_rank, other=0
    )
    output_offset_for_remote_rank = tl.sum(output_split_sizes)

    return input_offset_for_remote_rank, output_offset_for_remote_rank, num_rows_to_read

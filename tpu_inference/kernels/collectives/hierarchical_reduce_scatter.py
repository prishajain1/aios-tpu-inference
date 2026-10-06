# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hierarchical Recursive Doubling & All-to-All Reduce-Scatter Implementation."""

import functools
import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental import shard_map
from jax.experimental.pallas import tpu as pltpu


def _next_multiple_of(val, multiple):
    return ((val + multiple - 1) // multiple) * multiple


def _reduce_epilogue_kernel(
    x_ref,
    o_ref,
    *,
    num_devices: int = 8,
    activation: Callable[[jax.Array], jax.Array] | None = None,
):
    """Pairwise tree reduction in float32 inside VMEM with optional activation epilogue."""
    x = x_ref[...]
    if num_devices == 8:
        s01 = x[0].astype(jnp.float32) + x[1].astype(jnp.float32)
        s23 = x[2].astype(jnp.float32) + x[3].astype(jnp.float32)
        s0123 = s01 + s23

        s45 = x[4].astype(jnp.float32) + x[5].astype(jnp.float32)
        s67 = x[6].astype(jnp.float32) + x[7].astype(jnp.float32)
        s4567 = s45 + s67

        acc = s0123 + s4567
    else:
        acc = x[0].astype(jnp.float32)
        for i in range(1, num_devices):
            acc = acc + x[i].astype(jnp.float32)

    if activation is not None:
        acc = activation(acc)
    o_ref[...] = acc.astype(o_ref.dtype)


def _reduce_residual_epilogue_kernel(
    x_ref,
    res_ref,
    o_ref,
    *,
    num_devices: int = 8,
    activation: Callable[[jax.Array], jax.Array] | None = None,
):
    """Pairwise tree reduction in float32 fused with residual addition and activation in VMEM."""
    x = x_ref[...]
    if num_devices == 8:
        s01 = x[0].astype(jnp.float32) + x[1].astype(jnp.float32)
        s23 = x[2].astype(jnp.float32) + x[3].astype(jnp.float32)
        s0123 = s01 + s23

        s45 = x[4].astype(jnp.float32) + x[5].astype(jnp.float32)
        s67 = x[6].astype(jnp.float32) + x[7].astype(jnp.float32)
        s4567 = s45 + s67

        acc = s0123 + s4567
    else:
        acc = x[0].astype(jnp.float32)
        for i in range(1, num_devices):
            acc = acc + x[i].astype(jnp.float32)

    acc = acc + res_ref[...].astype(jnp.float32)
    if activation is not None:
        acc = activation(acc)
    o_ref[...] = acc.astype(o_ref.dtype)


def all_to_all_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    *,
    axis_name: str | tuple[str, ...] = 'x',
    bm: int = 256,
    bd: int = 2048,
    residual: jax.Array | None = None,
    activation: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Bandwidth-optimal reduce-scatter via native-dtype all_to_all + VMEM fp32 tree reduction.

    Exchanges shards in local_x.dtype (e.g. bfloat16, halving ICI wire traffic
    relative to float32 pre-cast) and accumulates in float32 in VMEM with optional
    fused residual addition and activation epilogue.
    """
    seq_len_dim, hidden_size_dim = local_x.shape
    final_chunk_size = seq_len_dim // num_devices

    x_chunks = local_x.reshape((num_devices, final_chunk_size, hidden_size_dim))
    received = jax.lax.all_to_all(
        x_chunks, axis_name, split_axis=0, concat_axis=0
    )

    bm_eff = min(bm, final_chunk_size)
    while final_chunk_size % bm_eff != 0 and bm_eff > 16:
        bm_eff //= 2
    bd_eff = min(bd, hidden_size_dim)
    while hidden_size_dim % bd_eff != 0 and bd_eff > 128:
        bd_eff //= 2

    grid = (final_chunk_size // bm_eff, hidden_size_dim // bd_eff)
    out_shape = jax.ShapeDtypeStruct(
        (final_chunk_size, hidden_size_dim), local_x.dtype
    )

    if residual is not None:
        in_specs = [
            pl.BlockSpec(
                (num_devices, bm_eff, bd_eff), lambda i, j: (0, i, j)
            ),
            pl.BlockSpec((bm_eff, bd_eff), lambda i, j: (i, j)),
        ]
        return pl.pallas_call(
            functools.partial(
                _reduce_residual_epilogue_kernel,
                num_devices=num_devices,
                activation=activation,
            ),
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=pl.BlockSpec((bm_eff, bd_eff), lambda i, j: (i, j)),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=('parallel', 'parallel'),
                vmem_limit_bytes=64 * 1024 * 1024,
            ),
        )(received, residual)

    in_specs = [
        pl.BlockSpec((num_devices, bm_eff, bd_eff), lambda i, j: (0, i, j))
    ]
    return pl.pallas_call(
        functools.partial(
            _reduce_epilogue_kernel,
            num_devices=num_devices,
            activation=activation,
        ),
        out_shape=out_shape,
        grid=grid,
        in_specs=in_specs,
        out_specs=pl.BlockSpec((bm_eff, bd_eff), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=('parallel', 'parallel'),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
    )(received)


# Asynchronously accumulates received data into a running sum using VMEM buffers.
def _accumulate(
    recv_hbm,
    run_hbm,
    out_hbm,
    vmem_recv,
    vmem_run,
    sync_sems,
    vmem_idx=0,
    activation: Callable[[jax.Array], jax.Array] | None = None,
):
    load_recv_sem, load_run_sem, store_run_sem = sync_sems

    load_recv_op = pltpu.make_async_copy(recv_hbm, vmem_recv.at[vmem_idx],
                                         load_recv_sem)
    load_run_op = pltpu.make_async_copy(run_hbm, vmem_run.at[vmem_idx],
                                        load_run_sem)
    load_recv_op.start()
    load_run_op.start()
    load_recv_op.wait()
    load_run_op.wait()

    acc = (
        vmem_run[vmem_idx, ...].astype(jnp.float32)
        + vmem_recv[vmem_idx, ...].astype(jnp.float32)
    )
    if activation is not None:
        acc = activation(acc)
    vmem_run[vmem_idx, ...] = acc.astype(vmem_run.dtype)

    if out_hbm is not None:
        store_run_op = pltpu.make_async_copy(vmem_run.at[vmem_idx], out_hbm,
                                             store_run_sem)
    else:
        store_run_op = pltpu.make_async_copy(vmem_run.at[vmem_idx], run_hbm,
                                             store_run_sem)

    store_run_op.start()
    return store_run_op


def _get_hypercube_chunk_idx(loop_idx, future_dims, prev_dims, my_chip_id,
                             target_dim, dim_val):
    base = 0
    for d in prev_dims:
        bit = (my_chip_id >> d) & 1
        base = base | (bit << d)
    for bit_pos, d in enumerate(future_dims):
        bit = (loop_idx >> bit_pos) & 1
        base = base | (bit << d)
    base = base | (dim_val << target_dim)
    return base


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    vmem_recv_ref,
    vmem_run_ref,
    final_copy_sem,
    phase1_sync_sems,
    load_recv_sem,
    load_run_sem,
    store_run_sem,
    *phase2_sync_sems_args,
    num_chips: int,
    num_hypercube_dims: int,
    num_micro_batches: int,
    hidden_size_dim: int,
    final_chunk_size: int,
    full_chunk_size: int,
    mb_size: int,
    vmem_pipeline_depth: int = 4,
    axis_name: str = 'x',
    activation: Callable[[jax.Array], jax.Array] | None = None,
):
    sync_sems = (load_recv_sem, load_run_sem, store_run_sem)

    if phase2_sync_sems_args:
        sync_sems_c2c = phase2_sync_sems_args[0]
    else:
        sync_sems_c2c = None

    cur_id = jax.lax.axis_index(axis_name)
    cur_chip_id = cur_id // 2
    cur_chiplet_bit = cur_id % 2
    is_even = cur_chiplet_bit == 0
    twin_id = jax.lax.select(is_even, cur_id + 1, cur_id - 1)

    # Phase 1: Intra-chip Reduce-Scatter (Fast Link)
    with jax.named_scope('phase1_intra_chip'):
        phase1_mb_size = _next_multiple_of(
            hidden_size_dim // num_micro_batches, 128)

        phase1_ops = []
        for micro_batch_idx in range(num_micro_batches):
            mb_start = micro_batch_idx * phase1_mb_size
            mb_size_actual = min(phase1_mb_size, hidden_size_dim - mb_start)
            if mb_size_actual <= 0:
                phase1_ops.append(None)
                continue

            mb_ops = []
            for pair_idx in range(num_chips):
                c_neigh = pair_idx * 2 + (1 - cur_chiplet_bit)

                op = pltpu.make_async_remote_copy(
                    src_ref=input_ref.at[
                        pl.ds(c_neigh * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    dst_ref=recv_buf_ref.at[
                        pl.ds(c_neigh * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    send_sem=phase1_sync_sems.at[pair_idx, micro_batch_idx],
                    recv_sem=phase1_sync_sems.at[pair_idx, micro_batch_idx],
                    device_id=twin_id,
                    device_id_type=pltpu.DeviceIdType.LOGICAL,
                )
                op.start()
                mb_ops.append(op)
            phase1_ops.append(mb_ops)

        # Wait and accumulate sequentially
        for micro_batch_idx in range(num_micro_batches):
            mb_start = micro_batch_idx * phase1_mb_size
            mb_size_actual = min(phase1_mb_size, hidden_size_dim - mb_start)
            if mb_size_actual <= 0:
                break

            store_ops = []
            for pair_idx in range(num_chips):
                c_me = pair_idx * 2 + cur_chiplet_bit

                phase1_ops[micro_batch_idx][pair_idx].wait()

                s_op = _accumulate(
                    recv_hbm=recv_buf_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    run_hbm=input_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    out_hbm=running_sum_ref.at[
                        pl.ds(c_me * final_chunk_size, final_chunk_size),
                        pl.ds(mb_start, mb_size_actual),
                    ],
                    vmem_recv=vmem_recv_ref.at[:,
                                               pl.ds(0, final_chunk_size),
                                               pl.ds(0, mb_size_actual)],
                    vmem_run=vmem_run_ref.at[:,
                                             pl.ds(0, final_chunk_size),
                                             pl.ds(0, mb_size_actual)],
                    sync_sems=sync_sems,
                    vmem_idx=pair_idx % vmem_pipeline_depth,
                )
                store_ops.append(s_op)
            for s_op in store_ops:
                s_op.wait()

    # Phase 2: Inter-chip Reduce-Scatter (3D Concurrent with Micro-batching)
    for phase_step in range(num_hypercube_dims):
        is_last_step = phase_step == num_hypercube_dims - 1
        with jax.named_scope(f'phase2_step_{phase_step}'):
            all_mb_ops = []
            for micro_batch_idx in range(num_micro_batches):
                mb_ops = []

                num_ops_in_step = 2**(num_hypercube_dims - 1 - phase_step)
                for op_idx in range(num_ops_in_step):

                    for hypercube_dim_idx in range(num_hypercube_dims):
                        dim = (hypercube_dim_idx +
                               phase_step) % num_hypercube_dims
                        neighbor_chip_id = cur_chip_id ^ (1 << dim)
                        my_dim_bit = (cur_chip_id >> dim) & 1
                        neigh_dim_bit = 1 - my_dim_bit

                        prev_dims = [
                            (hypercube_dim_idx + j) % num_hypercube_dims
                            for j in range(phase_step)
                        ]
                        future_dims = [
                            (hypercube_dim_idx + j) % num_hypercube_dims
                            for j in range(phase_step + 1, num_hypercube_dims)
                        ]

                        chunk_start = hypercube_dim_idx * full_chunk_size
                        chunk_end = min(chunk_start + full_chunk_size,
                                        hidden_size_dim)

                        mb_start_idx = min(
                            chunk_start + (micro_batch_idx * mb_size),
                            chunk_end)
                        mb_end_idx = min(mb_start_idx + mb_size, chunk_end)
                        k_size = mb_end_idx - mb_start_idx

                        my_base_chunk_idx = _get_hypercube_chunk_idx(
                            op_idx,
                            future_dims,
                            prev_dims,
                            cur_chip_id,
                            target_dim=dim,
                            dim_val=my_dim_bit,
                        )
                        neighbor_base_chunk_idx = _get_hypercube_chunk_idx(
                            op_idx,
                            future_dims,
                            prev_dims,
                            cur_chip_id,
                            target_dim=dim,
                            dim_val=neigh_dim_bit,
                        )

                        my_chunk_idx = my_base_chunk_idx * 2 + cur_chiplet_bit
                        neighbor_chunk_idx = neighbor_base_chunk_idx * 2 + cur_chiplet_bit

                        if k_size > 0:
                            op = pltpu.make_async_remote_copy(
                                src_ref=running_sum_ref.at[
                                    pl.ds(
                                        neighbor_chunk_idx * final_chunk_size,
                                        final_chunk_size,
                                    ),
                                    pl.ds(mb_start_idx, k_size),
                                ],
                                dst_ref=recv_buf_ref.at[
                                    pl.ds(
                                        neighbor_chunk_idx * final_chunk_size,
                                        final_chunk_size,
                                    ),
                                    pl.ds(mb_start_idx, k_size),
                                ],
                                send_sem=sync_sems_c2c.at[  # pytype: disable=attribute-error
                                    phase_step, micro_batch_idx,
                                    hypercube_dim_idx, op_idx],
                                recv_sem=sync_sems_c2c.at[  # pytype: disable=attribute-error
                                    phase_step, micro_batch_idx,
                                    hypercube_dim_idx, op_idx],
                                device_id=neighbor_chip_id * 2 +
                                cur_chiplet_bit,
                                device_id_type=pltpu.DeviceIdType.LOGICAL,
                            )
                            op.start()
                            mb_ops.append(
                                (op, my_chunk_idx, mb_start_idx, k_size))

                all_mb_ops.append(mb_ops)

            for micro_batch_idx in range(num_micro_batches):
                store_ops = []
                for op_id, (op, my_chunk_idx, c_start,
                            k_sz) in enumerate(all_mb_ops[micro_batch_idx]):
                    op.wait()
                    # On the final hypercube step, write directly to output_ref
                    # and apply optional activation epilogue in VMEM, avoiding
                    # the extra HBM-to-HBM final_copy_op round-trip.
                    out_target = (
                        output_ref.at[pl.ds(0, final_chunk_size), pl.ds(c_start, k_sz)]
                        if is_last_step
                        else None
                    )
                    s_op = _accumulate(
                        recv_hbm=recv_buf_ref.at[
                            pl.ds(my_chunk_idx *
                                  final_chunk_size, final_chunk_size),
                            pl.ds(c_start, k_sz),
                        ],
                        run_hbm=running_sum_ref.at[
                            pl.ds(my_chunk_idx *
                                  final_chunk_size, final_chunk_size),
                            pl.ds(c_start, k_sz),
                        ],
                        out_hbm=out_target,
                        vmem_recv=vmem_recv_ref.at[:,
                                                   pl.ds(0, final_chunk_size),
                                                   pl.ds(0, k_sz)],
                        vmem_run=vmem_run_ref.at[:,
                                                 pl.ds(0, final_chunk_size),
                                                 pl.ds(0, k_sz)],
                        sync_sems=sync_sems,
                        vmem_idx=op_id % vmem_pipeline_depth,
                        activation=activation if is_last_step else None,
                    )
                    store_ops.append(s_op)
                for s_op in store_ops:
                    s_op.wait()

    if num_hypercube_dims == 0:
        final_copy_op = pltpu.make_async_copy(
            src_ref=running_sum_ref.at[
                pl.ds(cur_id * final_chunk_size, final_chunk_size), :],
            dst_ref=output_ref,
            sem=final_copy_sem,
        )
        final_copy_op.start()
        final_copy_op.wait()


def hierarchical_reduce_scatter_local(
    local_x: jax.Array,
    num_devices: int,
    num_micro_batches: int = 2,
    axis_name: str | tuple[str, ...] = 'x',
    *,
    max_vmem_scratch_bytes: int = 32 * 1024 * 1024,
    use_all_to_all: bool = False,
    residual: jax.Array | None = None,
    activation: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    if use_all_to_all or residual is not None:
        return all_to_all_reduce_scatter_local(
            local_x,
            num_devices=num_devices,
            axis_name=axis_name,
            residual=residual,
            activation=activation,
        )

    num_chips = num_devices // 2
    num_hypercube_dims = int(math.log2(num_chips)) if num_chips > 0 else 0

    seq_len_dim = local_x.shape[0]
    hidden_size_dim = local_x.shape[1]
    final_chunk_size = seq_len_dim // num_devices

    # Auto-scale num_micro_batches (and vmem_pipeline_depth if needed) so the
    # two VMEM scratch buffers (2 * depth * final_chunk_size * max_vmem_mb_size)
    # never exceed VMEM capacity on large shapes (e.g. Hard: m=32768, d=4096).
    dtype_bytes = jax.dtypes.canonicalize_dtype(local_x.dtype).itemsize
    vmem_pipeline_depth = 4
    max_mbs = max(1, hidden_size_dim // 128)
    while num_micro_batches < max_mbs:
        cand_vmem_mb_size = _next_multiple_of(
            hidden_size_dim // num_micro_batches, 128
        )
        total_vmem_bytes = (
            2
            * vmem_pipeline_depth
            * final_chunk_size
            * cand_vmem_mb_size
            * dtype_bytes
        )
        if total_vmem_bytes <= max_vmem_scratch_bytes:
            break
        num_micro_batches *= 2

    max_vmem_mb_size = _next_multiple_of(
        hidden_size_dim // num_micro_batches, 128
    )
    while (
        vmem_pipeline_depth > 1
        and 2
        * vmem_pipeline_depth
        * final_chunk_size
        * max_vmem_mb_size
        * dtype_bytes
        > max_vmem_scratch_bytes
    ):
        vmem_pipeline_depth //= 2

    chunk_size_raw = (hidden_size_dim // num_hypercube_dims
                      if num_hypercube_dims > 0 else hidden_size_dim)
    full_chunk_size = _next_multiple_of(chunk_size_raw, 128)
    mb_size_raw = max(1, full_chunk_size // num_micro_batches)
    mb_size = _next_multiple_of(mb_size_raw, 128)

    out_shape = jax.ShapeDtypeStruct((final_chunk_size, hidden_size_dim),
                                     local_x.dtype)
    running_sum_shape = jax.ShapeDtypeStruct((seq_len_dim, hidden_size_dim),
                                             local_x.dtype)
    recv_buf_shape = jax.ShapeDtypeStruct((seq_len_dim, hidden_size_dim),
                                          local_x.dtype)

    scratch_shapes = [
        pltpu.VMEM(
            (vmem_pipeline_depth, final_chunk_size, max_vmem_mb_size),
            local_x.dtype,
        ),
        pltpu.VMEM(
            (vmem_pipeline_depth, final_chunk_size, max_vmem_mb_size),
            local_x.dtype,
        ),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA,
    ]

    if num_hypercube_dims > 0:
        scratch_shapes.append(
            pltpu.SemaphoreType.DMA((
                num_hypercube_dims,
                num_micro_batches,
                num_hypercube_dims,
                2**(num_hypercube_dims - 1),
            )))

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=(
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ),
        scratch_shapes=tuple(scratch_shapes),
        grid=(1, ),
    )

    kernel_fn = jax.tree_util.Partial(
        hier_rs_kernel,
        num_chips=num_chips,
        num_hypercube_dims=num_hypercube_dims,
        num_micro_batches=num_micro_batches,
        hidden_size_dim=hidden_size_dim,
        final_chunk_size=final_chunk_size,
        full_chunk_size=full_chunk_size,
        mb_size=mb_size,
        vmem_pipeline_depth=vmem_pipeline_depth,
        axis_name=axis_name,
        activation=activation,
    )

    hier_rs = pl.pallas_call(
        kernel_fn,
        out_shape=(out_shape, running_sum_shape, recv_buf_shape),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=96 * 1024 * 1024,
        ),
    )

    final_out, _, _ = hier_rs(local_x)
    return final_out


def hierarchical_reduce_scatter(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    in_specs: jax.sharding.PartitionSpec = jax.sharding.PartitionSpec(
        'x', None),
    num_micro_batches: int = 2,
    use_all_to_all: bool = False,
    residual: jax.Array | None = None,
    activation: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    num_devices = mesh.devices.size

    def inner(local_x):
        return hierarchical_reduce_scatter_local(
            local_x,
            num_devices,
            num_micro_batches,
            use_all_to_all=use_all_to_all,
            residual=residual,
            activation=activation,
        )

    return shard_map.shard_map(
        inner,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=in_specs,
        check_rep=False,
    )(x)

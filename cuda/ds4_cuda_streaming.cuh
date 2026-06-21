extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming_mode = enabled ? 1 : 0;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    if (!g_ssd_streaming_mode) {
        cuda_stream_selected_cache_release();
        cuda_stream_expert_cache_release_all();
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_budget_override = experts;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    cuda_stream_selected_cache_invalidate();
    cuda_stream_expert_cache_release_all();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared()) return 0;
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache.count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_expert_cache_release_all();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared() ||
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes) == 0) {
        return 0;
    }
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids || n_selected == 0 ||
        n_selected > n_total_expert ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed selected")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_selected);
    if (!cache) return 1;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming seed selected expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)selected_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    return 1;
}

static int cuda_stream_selected_cache_begin_compact_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            strict_failure,
        int            allow_global_cache) {
    cuda_stream_selected_cache_invalidate();
    cuda_model_load_progress_finish();

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !compact_ids || !slot_ids ||
        n_total_expert == 0 ||
        compact_count == 0 || compact_count > n_total_expert ||
        slot_count == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: CUDA streaming selected expert size overflow\n");
        return 0;
    }

    const uint64_t full_gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t compact_gate_bytes = (uint64_t)compact_count * gate_expert_bytes;
    const uint64_t compact_down_bytes = (uint64_t)compact_count * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: CUDA streaming selected expert range outside model map\n");
        return 0;
    }

    if (!allow_global_cache) {
        cuda_stream_expert_cache_release_all();
    }

    if (!cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.gate_ptr,
                                           &g_stream_selected_cache.gate_capacity,
                                           compact_gate_bytes,
                                           "selected gate experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.up_ptr,
                                           &g_stream_selected_cache.up_capacity,
                                           compact_gate_bytes,
                                           "selected up experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache.down_ptr,
                                           &g_stream_selected_cache.down_capacity,
                                           compact_down_bytes,
                                           "selected down experts") ||
        !cuda_stream_selected_ensure_i32(&g_stream_selected_cache.slot_selected_ptr,
                                         &g_stream_selected_cache.slot_selected_capacity,
                                         slot_count,
                                         "selected expert slots")) {
        return strict_failure ? 0 : 1;
    }

    if (allow_global_cache) {
        cuda_stream_expert_cache_note_size(gate_expert_bytes,
                                           down_expert_bytes);
    }
    const uint32_t configured_cache_budget =
        cuda_stream_expert_cache_configured_budget();
    const int use_global_cache =
        allow_global_cache &&
        configured_cache_budget != 0;
    cuda_stream_expert_cache *expert_cache = use_global_cache ?
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         configured_cache_budget) :
        NULL;
    int expert_cache_disabled = expert_cache == NULL;
    const uint32_t cache_count_before =
        expert_cache && expert_cache->valid ? expert_cache->count : 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t direct_loads = 0;

    for (uint32_t i = 0; i < compact_count; i++) {
        if (compact_ids[i] < 0 || (uint32_t)compact_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    compact_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }

        const uint64_t expert = (uint64_t)(uint32_t)compact_ids[i];
        const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
        int copied_from_global_cache = 0;

        if (!expert_cache_disabled) {
            int cache_slot =
                cuda_stream_expert_cache_find(expert_cache,
                                              model_map,
                                              model_size,
                                              layer,
                                              n_total_expert,
                                              (uint32_t)expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes);
            if (cache_slot >= 0) {
                cache_hits++;
                expert_cache->slots[(uint32_t)cache_slot].age =
                    ++expert_cache->tick;
            } else {
                cache_misses++;
                const uint32_t load_slot =
                    cuda_stream_expert_cache_lru_slot(expert_cache);
                const int append = !expert_cache->slots[load_slot].valid;
                if (cuda_stream_expert_cache_load_slot(expert_cache,
                                                       model_map,
                                                       model_size,
                                                       load_slot,
                                                       layer,
                                                       n_total_expert,
                                                       (uint32_t)expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes)) {
                    if (append && expert_cache->count < expert_cache->capacity) {
                        expert_cache->count++;
                    }
                    cache_slot = (int)load_slot;
                } else {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                    cache_slot = -1;
                }
            }

            if (cache_slot >= 0) {
                copied_from_global_cache =
                    cuda_stream_expert_cache_copy_to_compact(
                            expert_cache,
                            (uint32_t)cache_slot,
                            i,
                            g_stream_selected_cache.gate_ptr,
                            g_stream_selected_cache.up_ptr,
                            g_stream_selected_cache.down_ptr);
                if (!copied_from_global_cache) {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                }
            }
        }

        if (!copied_from_global_cache) {
            const uint64_t gate_src = gate_offset + expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + expert * down_expert_bytes;
            direct_loads++;
            if (!cuda_model_copy_to_device_streamed(g_stream_selected_cache.gate_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    gate_src,
                                                    gate_expert_bytes,
                                                    "selected moe_gate") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache.up_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    up_src,
                                                    gate_expert_bytes,
                                                    "selected moe_up") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache.down_ptr + down_dst,
                                                    model_map,
                                                    model_size,
                                                    down_src,
                                                    down_expert_bytes,
                                                    "selected moe_down")) {
                cuda_stream_selected_cache_invalidate();
                return strict_failure ? 0 : 1;
            }
        }
    }

    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache.slot_selected_ptr,
                            slot_ids,
                            (size_t)slot_count * sizeof(slot_ids[0]),
                            cudaMemcpyHostToDevice),
                 "streaming selected slot upload")) {
        cuda_stream_selected_cache_invalidate();
        return strict_failure ? 0 : 1;
    }

    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = slot_count;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = compact_count;
    g_stream_selected_cache.gate_offset = gate_offset;
    g_stream_selected_cache.up_offset = up_offset;
    g_stream_selected_cache.down_offset = down_offset;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.valid = 1;

    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming selected layer=%u slots=%u compact=%u global_budget=%u before=%u after=%u hits=%u misses=%u direct=%u gate/up %.2f MiB down %.2f MiB\n",
                layer,
                slot_count,
                compact_count,
                expert_cache && expert_cache->valid ? expert_cache->capacity : 0,
                cache_count_before,
                expert_cache && expert_cache->valid ? expert_cache->count : 0,
                cache_hits,
                cache_misses,
                direct_loads,
                (double)compact_gate_bytes / 1048576.0,
                (double)compact_down_bytes / 1048576.0);
    }
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected_ids || n_selected == 0) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids(n_selected);
    compact_ids.reserve(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            n_selected,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            0,
            1);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table ||
        !selected_ids ||
        table->n_total_expert == 0 ||
        n_selected == 0 ||
        n_tokens == 0 ||
        (uint64_t)n_tokens > UINT32_MAX / (uint64_t)n_selected) {
        return 0;
    }
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    const uint32_t slot_count = n_tokens * n_selected;
    std::vector<int32_t> slot_ids(slot_count);
    compact_ids.reserve(slot_count < n_total_expert ? slot_count : n_total_expert);

    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming batch selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }

    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            slot_count,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            1,
            0);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !expert_ids || n_experts == 0 ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed hotlist")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_experts);
    if (!cache || cache->capacity == 0) return 1;

    const uint32_t layer_seed_cap =
        n_experts < cache->capacity ? n_experts : cache->capacity;
    std::vector<uint32_t> chosen;
    try {
        chosen.reserve(layer_seed_cap);
    } catch (...) {
        return 1;
    }

    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming hotlist seed expert id %d is outside 0..%u at layer %u\n",
                    expert,
                    n_total_expert,
                    layer);
            return 0;
        }
        const uint32_t priority =
            expert_priorities ? expert_priorities[i] : (n_experts - i);
        uint32_t pos = 0;
        while (pos < chosen.size()) {
            const uint32_t other = chosen[pos];
            const uint32_t other_priority =
                expert_priorities ? expert_priorities[other] :
                                    (n_experts - other);
            if (priority > other_priority) break;
            pos++;
        }
        if (chosen.size() < layer_seed_cap) {
            chosen.insert(chosen.begin() + pos, i);
        } else if (pos < chosen.size()) {
            chosen.insert(chosen.begin() + pos, i);
            chosen.pop_back();
        }
    }

    const uint32_t n = (uint32_t)chosen.size();
    for (uint32_t ri = 0; ri < n; ri++) {
        const uint32_t i = chosen[n - 1u - ri];
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)expert_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "ds4: CUDA streaming hotlist seeded layer=%u requested=%u cached=%u cap=%u\n",
                layer,
                n_experts,
                n,
                cache->capacity);
    }
    return 1;
}
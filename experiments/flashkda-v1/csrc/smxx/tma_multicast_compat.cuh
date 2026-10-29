#pragma once

#include <cute/atom/copy_traits_sm90_tma.hpp>

// CUTLASS deliberately traps SM90_TMA_LOAD_MULTICAST on SM120 because the
// legacy CTA-group-1 multicast instruction can be substantially slower there.
// PTX still supports the instruction on sm_90+, so keep the standard CUTLASS
// implementation on Hopper and issue the same PTX directly on SM120. This lets
// the experimental cluster path run on consumer Blackwell for correctness and
// performance characterization without modifying the CUTLASS submodule.
namespace cute {

struct FLASH_KDA_TMA_LOAD_MULTICAST {
    CUTE_HOST_DEVICE static void copy(
        void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
        uint64_t cache_hint, void* smem_ptr, int32_t const& crd0
    ) {
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
        uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t mbar = cast_smem_ptr_to_uint(mbar_ptr);
        uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%4}], [%2], %3, %5;"
            :
            : "r"(smem), "l"(desc), "r"(mbar), "h"(multicast_mask),
              "r"(crd0), "l"(cache_hint)
            : "memory");
#else
        SM90_TMA_LOAD_MULTICAST_1D::copy(
            desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr, crd0);
#endif
    }

    CUTE_HOST_DEVICE static void copy(
        void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
        uint64_t cache_hint, void* smem_ptr,
        int32_t const& crd0, int32_t const& crd1
    ) {
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
        uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t mbar = cast_smem_ptr_to_uint(mbar_ptr);
        uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%4, %5}], [%2], %3, %6;"
            :
            : "r"(smem), "l"(desc), "r"(mbar), "h"(multicast_mask),
              "r"(crd0), "r"(crd1), "l"(cache_hint)
            : "memory");
#else
        SM90_TMA_LOAD_MULTICAST_2D::copy(
            desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr,
            crd0, crd1);
#endif
    }

    CUTE_HOST_DEVICE static void copy(
        void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
        uint64_t cache_hint, void* smem_ptr,
        int32_t const& crd0, int32_t const& crd1, int32_t const& crd2
    ) {
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
        uint64_t desc = reinterpret_cast<uint64_t>(desc_ptr);
        uint32_t mbar = cast_smem_ptr_to_uint(mbar_ptr);
        uint32_t smem = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint"
            " [%0], [%1, {%4, %5, %6}], [%2], %3, %7;"
            :
            : "r"(smem), "l"(desc), "r"(mbar), "h"(multicast_mask),
              "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
            : "memory");
#else
        SM90_TMA_LOAD_MULTICAST_3D::copy(
            desc_ptr, mbar_ptr, multicast_mask, cache_hint, smem_ptr,
            crd0, crd1, crd2);
#endif
    }

    using PREFETCH = typename SM90_TMA_LOAD::PREFETCH;
};

struct FLASH_KDA_TMA_LOAD_MULTICAST_OP : FLASH_KDA_TMA_LOAD_MULTICAST {};

template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<FLASH_KDA_TMA_LOAD_MULTICAST, NumBitsPerTMA, AuxParams_>
{
    using ThrID = Layout<_1>;
    using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
    using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
    using RefLayout = SrcLayout;

    TmaDescriptor tma_desc_;
    using AuxParams = AuxParams_;
    AuxParams aux_params_;

    CUTE_HOST_DEVICE constexpr
    TmaDescriptor const* get_tma_descriptor() const {
        return &tma_desc_;
    }

    CUTE_HOST_DEVICE constexpr
    Copy_Traits<FLASH_KDA_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA> with(
        uint64_t& tma_load_mbar,
        uint16_t const& multicast_mask,
        TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL
    ) const {
        return {&this->tma_desc_, &tma_load_mbar, multicast_mask,
                static_cast<uint64_t>(cache_hint)};
    }

    CUTE_HOST_DEVICE constexpr
    Copy_Traits<FLASH_KDA_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA> with(
        TmaDescriptor const* new_tma_desc,
        uint64_t& tma_load_mbar,
        uint16_t const& multicast_mask,
        TMA::CacheHintSm90 const& cache_hint = TMA::CacheHintSm90::EVICT_NORMAL
    ) const {
        return {new_tma_desc, &tma_load_mbar, multicast_mask,
                static_cast<uint64_t>(cache_hint)};
    }

    template <class GShape>
    CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
        static_assert(
            is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
        return make_coord_tensor(make_layout(g_shape, aux_params_.g_stride_));
    }

    template <class TS, class SLayout, class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void copy_unpack(
        Copy_Traits const&, Tensor<TS, SLayout> const&, Tensor<TD, DLayout>&) = delete;
};

template <class NumBitsPerTMA>
struct Copy_Traits<FLASH_KDA_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA>
    : TMA_LOAD_Unpack<FLASH_KDA_TMA_LOAD_MULTICAST_OP, NumBitsPerTMA> {
    using ThrID = Layout<_1>;
    using SrcLayout = Layout<Shape<_1, NumBitsPerTMA>>;
    using DstLayout = Layout<Shape<_1, NumBitsPerTMA>>;
    using RefLayout = SrcLayout;

    tuple<TmaDescriptor const*, uint64_t*, uint16_t, uint64_t> const opargs_;

    CUTE_HOST_DEVICE
    Copy_Traits(TmaDescriptor const* desc, uint64_t* mbar,
                uint16_t mask, uint64_t hint)
        : opargs_(desc, mbar, mask, hint) {}

    CUTE_HOST_DEVICE constexpr
    TmaDescriptor const* get_tma_descriptor() const {
        return get<0>(opargs_);
    }
};

} // namespace cute

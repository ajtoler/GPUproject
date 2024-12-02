#ifndef workdiv_h
#define workdiv_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "config.hpp"

// If first argument not a multiple of the second argument, round it up to the next multiple
inline constexpr Idx round(Idx value, Idx divisor) { return (value + divisor - 1) / divisor * divisor; }

// Return integer division of first argument by the second argument, rounded up to the next integer
inline constexpr Idx divide(Idx value, Idx divisor) { return (value + divisor - 1) / divisor; }

// Describes if accelerator expects threads-per-block and elements-per-thread to be swapped
template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
struct requires_single_thread_per_block : public std::true_type {};

// ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <typename TDim>
struct requires_single_thread_per_block<alpaka::AccGpuCudaRt<TDim, Idx>> : public std::false_type {};
#endif

// ALPAKA_ACC_GPU_HIP_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
template <typename TDim>
struct requires_single_thread_per_block<alpaka::AccGpuHipRt<TDim, Idx>> : public std::false_type {};
#endif 

// If accelerator expects the threads-per-block and elements-per-thread to be swapped
template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
inline constexpr bool requires_single_thread_per_block_v = requires_single_thread_per_block<TAcc>::value;

// Create accelerator-dependent work division for 1D kernels
template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
inline WorkDiv<Dim1D> make_workdiv(Idx blocks, Idx elements) {
  if constexpr (not requires_single_thread_per_block_v<TAcc>) {
    // On GPU backend, each thread is looking at single element:
    //   - number of threads per block is "elements";
    //   - number of elements per thread always 1.
    return WorkDiv<Dim1D>(blocks, elements, Idx{1});
  } else {
    // On CPU backend, run serially with a single thread per block:
    //   - number of elements per thread is "elements".
    //   - number of threads per block always 1;
    return WorkDiv<Dim1D>(blocks, Idx{1}, elements);
  }
}

// Create accelerator-dependent workdiv for ND kernels
template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc>>>
inline WorkDiv<alpaka::Dim<TAcc>> make_workdiv(const Vec<alpaka::Dim<TAcc>>& blocks,
                                               const Vec<alpaka::Dim<TAcc>>& elements) {
  using Dim = alpaka::Dim<TAcc>;
  if constexpr (not requires_single_thread_per_block_v<TAcc>) {
    // On GPU backend, each thread is looking at single element:
    //   - number of threads per block is "elements";
    //   - number of elements per thread always 1.
    return WorkDiv<Dim>(blocks, elements, Vec<Dim>::ones());
  } else {
    // On CPU backend, run serially with a single thread per block:
    //   - number of elements per thread is "elements".
    //   - number of threads per block always 1;
    return WorkDiv<Dim>(blocks, Vec<Dim>::ones(), elements);
  }
}

template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and alpaka::Dim<TAcc>::value == 1>>
class elements_with_stride {
public:
  ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc)
      : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
        first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
        stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
        extent_{stride_} {}

  ALPAKA_FN_ACC inline elements_with_stride(TAcc const& acc, Idx extent)
      : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]},
        first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
        stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u] * elements_},
        extent_{extent} {}

  class iterator {
    friend class elements_with_stride;

    ALPAKA_FN_ACC inline iterator(Idx elements, Idx stride, Idx extent, Idx first)
        : elements_{elements},
          stride_{stride},
          extent_{extent},
          first_{std::min(first, extent)},
          index_{first_},
          last_{std::min(first + elements, extent)} {}

  public:
    ALPAKA_FN_ACC inline Idx operator*() const { return index_; }

    // pre-increment iterator
    ALPAKA_FN_ACC inline iterator& operator++() {
      if constexpr (requires_single_thread_per_block_v<TAcc>) {
        // increment index along elements processed by current thread
        ++index_;
        if (index_ < last_)
          return *this;
      }

      // increment thread index with grid stride
      first_ += stride_;
      index_ = first_;
      last_ = std::min(first_ + elements_, extent_);
      if (index_ < extent_)
        return *this;

      // iterator reached or passed end of extent, clamp to extent
      first_ = extent_;
      index_ = extent_;
      last_ = extent_;
      return *this;
    }

    // post-increment iterator
    ALPAKA_FN_ACC inline iterator operator++(int) {
      iterator old = *this;
      ++(*this);
      return old;
    }

    ALPAKA_FN_ACC inline bool operator==(iterator const& other) const {
      return (index_ == other.index_) and (first_ == other.first_);
    }

    ALPAKA_FN_ACC inline bool operator!=(iterator const& other) const { return not(*this == other); }

  private:
    // non-const to support iterator copy and assignment
    Idx elements_;
    Idx stride_;
    Idx extent_;
    // modified by pre/post-increment operator
    Idx first_;
    Idx index_;
    Idx last_;
  };

  ALPAKA_FN_ACC inline iterator begin() const { return iterator(elements_, stride_, extent_, first_); }

  ALPAKA_FN_ACC inline iterator end() const { return iterator(elements_, stride_, extent_, extent_); }

private:
  const Idx elements_;
  const Idx first_;
  const Idx stride_;
  const Idx extent_;
};

template <typename TAcc, typename = std::enable_if_t<alpaka::isAccelerator<TAcc> and (alpaka::Dim<TAcc>::value > 0)>>
class elements_with_stride_nd {
public:
  using Dim = alpaka::Dim<TAcc>;
  using Vec = alpaka::Vec<Dim, Idx>;

  ALPAKA_FN_ACC inline elements_with_stride_nd(TAcc const& acc)
      : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
        first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
        stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
        extent_{stride_} {}

  ALPAKA_FN_ACC inline elements_with_stride_nd(TAcc const& acc, Vec extent)
      : elements_{alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)},
        first_{alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc) * elements_},
        stride_{alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc) * elements_},
        extent_{extent} {}

  class iterator {
    friend class elements_with_stride_nd;

  public:
    ALPAKA_FN_ACC inline Vec operator*() const { return index_; }

    // pre-increment iterator
    ALPAKA_FN_ACC constexpr inline iterator operator++() {
      increment();
      return *this;
    }

    // post-increment iterator
    ALPAKA_FN_ACC constexpr inline iterator operator++(int) {
      iterator old = *this;
      increment();
      return old;
    }

    ALPAKA_FN_ACC constexpr inline bool operator==(iterator const& other) const { return (index_ == other.index_); }

    ALPAKA_FN_ACC constexpr inline bool operator!=(iterator const& other) const { return not(*this == other); }

  private:
    // private, explicit constructor
    ALPAKA_FN_ACC inline iterator(elements_with_stride_nd const* loop, Vec first)
        : loop_{loop},
          thread_{alpaka::elementwise_min(first, loop->extent_)},
          range_{alpaka::elementwise_min(first + loop->elements_, loop->extent_)},
          index_{thread_} {}

    template <size_t I>
    ALPAKA_FN_ACC inline constexpr bool nth_elements_loop() {
      bool overflow = false;
      ++index_[I];
      if (index_[I] >= range_[I]) {
        index_[I] = thread_[I];
        overflow = true;
      }
      return overflow;
    }

    template <size_t N>
    ALPAKA_FN_ACC inline constexpr bool do_elements_loops() {
      if constexpr (N == 0) {
        // overflow
        return true;
      } else {
        if (not nth_elements_loop<N - 1>()) {
          return false;
        } else {
          return do_elements_loops<N - 1>();
        }
      }
    }

    template <size_t I>
    ALPAKA_FN_ACC inline constexpr bool nth_strided_loop() {
      bool overflow = false;
      thread_[I] += loop_->stride_[I];
      if (thread_[I] >= loop_->extent_[I]) {
        thread_[I] = loop_->first_[I];
        overflow = true;
      }
      index_[I] = thread_[I];
      range_[I] = std::min(thread_[I] + loop_->elements_[I], loop_->extent_[I]);
      return overflow;
    }

    template <size_t N>
    ALPAKA_FN_ACC inline constexpr bool do_strided_loops() {
      if constexpr (N == 0) {
        // overflow
        return true;
      } else {
        if (not nth_strided_loop<N - 1>()) {
          return false;
        } else {
          return do_strided_loops<N - 1>();
        }
      }
    }

    // increment iterator
    ALPAKA_FN_ACC inline constexpr void increment() {
      if constexpr (requires_single_thread_per_block_v<TAcc>) {
        // linear ND loops over elements associated with thread;
        if (not do_elements_loops<Dim::value>()) {
          return;
        }
      }

      // strided ND loop over threads in kernel launch grid;
      if (not do_strided_loops<Dim::value>()) {
        return;
      }

      // iterator has reached/passed end of extent, clamp to extent
      thread_ = loop_->extent_;
      range_ = loop_->extent_;
      index_ = loop_->extent_;
    }

    // const pointer to elements_with_stride_nd that iterator refers to
    const elements_with_stride_nd* loop_;

    // modified by pre/post-increment operator
    Vec thread_;  // first element processed by thread
    Vec range_;   // last element processed by thread
    Vec index_;   // current element processed by thread
  };

  ALPAKA_FN_ACC inline iterator begin() const { return iterator{this, first_}; }

  ALPAKA_FN_ACC inline iterator end() const { return iterator{this, extent_}; }

private:
  const Vec elements_;
  const Vec first_;
  const Vec stride_;
  const Vec extent_;
};

#endif  // HeterogeneousCore_AlpakaInterface_interface_workdiv_h
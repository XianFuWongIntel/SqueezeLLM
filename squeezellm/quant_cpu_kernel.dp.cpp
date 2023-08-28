#include <dpct/dpct.hpp>
#include <torch/all.h>
#include <torch/python.h>

#include <torch/extension.h>
// #include <ipex.h>

// half-tensor
// #include <c10/cuda/CUDAStream.h>
// #include <ATen/cuda/CUDATensorMethods.cuh>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2);

template <typename scalar_t>
void SPMV_ATOMIC(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows
,
    const sycl::nd_item<3> &item_ct1);

template <typename scalar_t>
void SPMV_ATOMIC_BATCHED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1);

void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec);

void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int topX,
    int full_width,
    int batch,
    int vec_height,
    int matwidth
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec);

void vecquant3matmul_nuq_perchannel_cpu(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:38: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:39: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}

// 4-bit matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_cpu(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:40: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:41: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks * threads, threads),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height,
              width, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}

// 3-bit batched matvec kernel (LUT-based)
void vecquant3matmul_nuq_perchannel_batched_cpu(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:42: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:43: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant3MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height, width,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}

// 4-bit batched matvec kernel (LUT-based)
void vecquant4matmul_nuq_perchannel_batched_cpu(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:44: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:45: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat_data_ptr_int_ct1 = mat.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant4MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height, width,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}

//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_cpu(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  /*
  DPCT1038:5: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      /*
      DPCT1049:6: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  /*
  DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:46: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:47: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_cpu(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  /*
  DPCT1038:8: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      /*
      DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  /*
  DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:48: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:49: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();
}


//NUQ + Sparse
void vecquant3matmul_spmv_nuq_perchannel_batched_cpu(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat3,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:50: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:51: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks3 * threads3, threads3),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant3MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height3, width3,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:11: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      /*
      DPCT1049:12: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));
}

//NUQ + Sparse
void vecquant4matmul_spmv_nuq_perchannel_batched_cpu(
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor mat,
  torch::Tensor vec,
  torch::Tensor mul,
  int num_rows,
  torch::Tensor mat4,
  torch::Tensor lookup_table
) {

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:13: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto cgf = [&](sycl::handler &cgh) {
    /*
    DPCT1101:52: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:53: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks4 * threads4, threads4),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant4MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height4, width4,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  };

  dpct::get_default_queue().submit(cgf);
  dpct::get_default_queue().wait();

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:14: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      /*
      DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));
}


//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_cpu(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat3,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:54: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:55: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks3 * threads3, threads3),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant3MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height3,
              width3, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:18: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      /*
      DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  //dense matvec kernel here!
  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:56: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    auto full_row_indices_data_ptr_int_ct2 = full_row_indices.data_ptr<int>();
    auto mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernel(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, item_ct1, blockvec_acc_ct1.get_pointer());
        });
  });
}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_cpu(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat4,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:20: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:57: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:58: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks4 * threads4, threads4),
        [=](sycl::nd_item<3> item_ct1) {
          VecQuant4MatMulKernelNUQPerChannel(
              vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
              mul_data_ptr_float_ct2, lookup_table_data_ptr_float_ct3, height4,
              width4, item_ct1, blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
        });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:22: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic", ([&] {
      /*
      DPCT1049:23: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC(rows_data_int_ct0, cols_data_int_ct1,
                                     mat_data_scalar_t_ct2,
                                     vec_data_scalar_t_ct3,
                                     mul_data_scalar_t_ct4, num_rows, item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  //dense matvec kernel here!
  /*
  DPCT1049:21: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:59: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    auto full_row_indices_data_ptr_int_ct2 = full_row_indices.data_ptr<int>();
    auto mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernel(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, item_ct1, blockvec_acc_ct1.get_pointer());
        });
  });
}

//NUQ + hybrid sparse kernel
void vecquant3matmul_spmv_hybrid_nuq_perchannel_batched_cpu(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat3,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // dense kernel
  int height3 = mat3.size(0);
  int width3 = mat3.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks3(1, (width3 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height3 + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3);
  sycl::range<3> threads3(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:24: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:60: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:61: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(8, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat3_data_ptr_int_ct1 = mat3.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks3 * threads3, threads3),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant3MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat3_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height3, width3,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:26: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      /*
      DPCT1049:27: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  /*
  DPCT1049:25: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:62: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    auto full_row_indices_data_ptr_int_ct2 = full_row_indices.data_ptr<int>();
    auto mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernelBatched(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, batch, vec_height, matwidth, item_ct1,
              blockvec_acc_ct1.get_pointer());
        });
  });
}


//NUQ + hybrid sparse kernel
void vecquant4matmul_spmv_hybrid_nuq_perchannel_batched_cpu(
    torch::Tensor rows, torch::Tensor cols, torch::Tensor mat,
    torch::Tensor vec, torch::Tensor full_rows, torch::Tensor full_row_indices,
    torch::Tensor mul, int num_rows, torch::Tensor mat4,
    torch::Tensor lookup_table) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  // dense kernel
  int height4 = mat4.size(0);
  int width4 = mat4.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks4(1, (width4 + BLOCKWIDTH - 1) / BLOCKWIDTH,
                         (height4 + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads4(1, 1, BLOCKWIDTH);

  /*
  DPCT1049:28: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:63: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);
    /*
    DPCT1101:64: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 2> deq2_acc_ct1(
        sycl::range<2>(16, 128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto mat4_data_ptr_int_ct1 = mat4.data_ptr<int>();
    auto mul_data_ptr_float_ct2 = mul.data_ptr<float>();
    auto lookup_table_data_ptr_float_ct3 = lookup_table.data_ptr<float>();

    cgh.parallel_for(sycl::nd_range<3>(blocks4 * threads4, threads4),
                     [=](sycl::nd_item<3> item_ct1) {
                       VecQuant4MatMulKernelNUQPerChannelBatched(
                           vec_data_ptr_float_ct0, mat4_data_ptr_int_ct1,
                           mul_data_ptr_float_ct2,
                           lookup_table_data_ptr_float_ct3, height4, width4,
                           batch, vec_height, item_ct1,
                           blockvec_acc_ct1.get_pointer(), deq2_acc_ct1);
                     });
  });

  //spmv kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_rows + BLOCKWIDTH - 1) / BLOCKWIDTH;

  /*
  DPCT1038:30: When the kernel function name is used as a macro argument, the
  migration result may be incorrect. You need to verify the definition of the
  macro.
  */
  AT_DISPATCH_FLOATING_TYPES(mat.type(), "spmv_atomic_batched", ([&] {
      /*
      DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    auto cgf = [&](sycl::handler &cgh) {
      auto rows_data_int_ct0 = rows.data<int>();
      auto cols_data_int_ct1 = cols.data<int>();
      auto mat_data_scalar_t_ct2 = mat.data<scalar_t>();
      auto vec_data_scalar_t_ct3 = vec.data<scalar_t>();
      auto mul_data_scalar_t_ct4 = mul.data<scalar_t>();

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         SPMV_ATOMIC_BATCHED(
                             rows_data_int_ct0, cols_data_int_ct1,
                             mat_data_scalar_t_ct2, vec_data_scalar_t_ct3,
                             mul_data_scalar_t_ct4, num_rows, batch, vec_height,
                             item_ct1);
                       });
    };

    dpct::get_default_queue().submit(cgf);
    dpct::get_default_queue().wait();
                             }));

  // handle topk indices
  int height = full_rows.size(0);
  int width = full_rows.size(1);
  sycl::range<3> blocks_topX(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                             (height + BLOCKWIDTH - 1) / BLOCKWIDTH);
  sycl::range<3> threads_topX(1, 1, BLOCKWIDTH);

  int matwidth = mul.size(1);

  //dense matvec kernel here!
  /*
  DPCT1049:29: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    /*
    DPCT1101:65: 'BLOCKWIDTH' expression was replaced with a value. Modify the
    code to use the original expression, provided in comments, if it is
    correct.
    */
    sycl::local_accessor<float, 1> blockvec_acc_ct1(
        sycl::range<1>(128 /*BLOCKWIDTH*/), cgh);

    auto vec_data_ptr_float_ct0 = vec.data_ptr<float>();
    auto full_rows_data_ptr_float_ct1 = full_rows.data_ptr<float>();
    auto full_row_indices_data_ptr_int_ct2 = full_row_indices.data_ptr<int>();
    auto mul_data_ptr_float_ct3 = mul.data_ptr<float>();

    cgh.parallel_for(
        sycl::nd_range<3>(blocks_topX * threads_topX, threads_topX),
        [=](sycl::nd_item<3> item_ct1) {
          DenseMatVecKernelBatched(
              vec_data_ptr_float_ct0, full_rows_data_ptr_float_ct1,
              full_row_indices_data_ptr_int_ct2, mul_data_ptr_float_ct3, height,
              width, batch, vec_height, matwidth, item_ct1,
              blockvec_acc_ct1.get_pointer());
        });
  });
}


void VecQuant3MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] =
      vec[(row / BLOCKHEIGHT3) * BLOCKWIDTH + item_ct1.get_local_id(2)];

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 8;
  for (int val = 0; val < 8; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i = width * row + col;
  int k = 0;

  float res = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  /*
  DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp2 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp2 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp2 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp2 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp2 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp2 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp2 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp2 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp2 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp2 >>  27) & 0x7][off] * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
    k += 11;
    res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
    res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
    res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
    res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
    res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
    res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
    res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
    res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
    res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
    res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];
    i += width;
    k += 10;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[col],
                                                                     res);
}

//4-bit per-channel
void VecQuant4MatMulKernelNUQPerChannel(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] =
      vec[(row / BLOCKHEIGHT4) * BLOCKWIDTH + item_ct1.get_local_id(2)];

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  /*
  DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  float res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    res += deq2[(tmp >>  0) & 0xf][off] * blockvec[k + 0];
    res += deq2[(tmp >>  4) & 0xf][off] * blockvec[k + 1];
    res += deq2[(tmp >>  8) & 0xf][off] * blockvec[k + 2];
    res += deq2[(tmp >>  12) & 0xf][off] * blockvec[k + 3];
    res += deq2[(tmp >>  16) & 0xf][off] * blockvec[k + 4];
    res += deq2[(tmp >>  20) & 0xf][off] * blockvec[k + 5];
    res += deq2[(tmp >>  24) & 0xf][off] * blockvec[k + 6];
    res += deq2[(tmp >>  28) & 0xf][off] * blockvec[k + 7];

    i += width;
    k += 8;
  }

  dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(&mul[col],
                                                                     res);
}


//batched version (3-bit)
void VecQuant3MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT3 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 8;
  for (int val = 0; val < 8; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    //initialize vars
    i = width * row + col;
    res = 0;
    k = 0;

    /*
    DPCT1065:34: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + (row / BLOCKHEIGHT3) * BLOCKWIDTH +
            item_ct1.get_local_id(2)];
    /*
    DPCT1065:35: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    while (k < BLOCKWIDTH) {
      tmp1 = as_unsigned(mat[i]);

      res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];

      i += width;
      tmp2 = as_unsigned(mat[i]);
      tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
      tmp2 >>= 1;
      res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp2 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp2 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp2 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp2 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp2 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp2 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp2 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp2 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp2 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp2 >>  27) & 0x7][off] * blockvec[k + 9];

      i += width;
      tmp1 = as_unsigned(mat[i]);
      tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
      tmp1 >>= 2;
      res += deq2[(tmp >>  0) & 0x7][off] * blockvec[k + 10];
      k += 11;
      res += deq2[(tmp1 >>  0) & 0x7][off] * blockvec[k + 0];
      res += deq2[(tmp1 >>  3) & 0x7][off] * blockvec[k + 1];
      res += deq2[(tmp1 >>  6) & 0x7][off] * blockvec[k + 2];
      res += deq2[(tmp1 >>  9) & 0x7][off] * blockvec[k + 3];
      res += deq2[(tmp1 >>  12) & 0x7][off] * blockvec[k + 4];
      res += deq2[(tmp1 >>  15) & 0x7][off] * blockvec[k + 5];
      res += deq2[(tmp1 >>  18) & 0x7][off] * blockvec[k + 6];
      res += deq2[(tmp1 >>  21) & 0x7][off] * blockvec[k + 7];
      res += deq2[(tmp1 >>  24) & 0x7][off] * blockvec[k + 8];
      res += deq2[(tmp1 >>  27) & 0x7][off] * blockvec[k + 9];
      i += width;
      k += 10;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[b * width + col], res);
  }
}

//batched version (4-bit)
void VecQuant4MatMulKernelNUQPerChannelBatched(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int batch,
    int vec_height
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec,
    sycl::local_accessor<float, 2> deq2) {

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  //Modified dequant block

  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + (val & 0xf);
    deq2[val][off] = lookup_table[lut_index];
  }

  int i;
  float res;
  int k;
  unsigned int tmp;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res = 0;
    k = 0;

    /*
    DPCT1065:36: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + (row / BLOCKHEIGHT4) * BLOCKWIDTH +
            item_ct1.get_local_id(2)];
    /*
    DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    while (k < BLOCKWIDTH) {
      tmp = as_unsigned(mat[i]);

      res += deq2[(tmp >>  0) & 0xf][off] * blockvec[k + 0];
      res += deq2[(tmp >>  4) & 0xf][off] * blockvec[k + 1];
      res += deq2[(tmp >>  8) & 0xf][off] * blockvec[k + 2];
      res += deq2[(tmp >>  12) & 0xf][off] * blockvec[k + 3];
      res += deq2[(tmp >>  16) & 0xf][off] * blockvec[k + 4];
      res += deq2[(tmp >>  20) & 0xf][off] * blockvec[k + 5];
      res += deq2[(tmp >>  24) & 0xf][off] * blockvec[k + 6];
      res += deq2[(tmp >>  28) & 0xf][off] * blockvec[k + 7];

      i += width;
      k += 8;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[b * width + col], res);
  }
}

template <typename scalar_t>
void SPMV_ATOMIC(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows
,
  const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);
    if (row < num_rows) {
        float dot = 0;
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int i = start_elem; i < end_elem; i++) {
            dot += mat[i] * vec[cols[i]];
        }
        dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
            &mul[row], dot);
    }
}

template <typename scalar_t>
void SPMV_ATOMIC_BATCHED(
  const       int* __restrict__ rows,
  const       int* __restrict__ cols,
  const  scalar_t* __restrict__ mat,
  const  scalar_t* __restrict__ vec,
         scalar_t* __restrict__ mul,
  const  int num_rows,
  int batch,
  int vec_height
,
  const sycl::nd_item<3> &item_ct1) {
    int row = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
              item_ct1.get_local_id(2);

    if (row < num_rows) {
        int start_elem = rows[row];
        int end_elem = rows[row+1];
        for (int b = 0; b < batch; ++b){
            float dot = 0;
            for (int i = start_elem; i < end_elem; i++) {
                dot += mat[i] * vec[b * vec_height + cols[i]];
                // dot += mat[i] * vec[cols[i] * batch + b];
                // dot += mat[i] * vec[cols[i]];
            }
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &mul[b * num_rows + row], dot);
            // atomicAdd(&mul[row * batch + b], dot);
            // atomicAdd(&mul[row], dot);
        }
    }
}

// Dense kernel for only a subset of rows
void DenseMatVecKernel(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec) {

  int row = BLOCKWIDTH * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  blockvec[item_ct1.get_local_id(2)] = vec[row + item_ct1.get_local_id(2)];

  item_ct1.barrier(sycl::access::fence_space::local_space);

  int i = width * row + col;
  int k = 0;
  float res = 0;

  if (item_ct1.get_local_id(2) < width) {
    while (k < BLOCKWIDTH) {
      res += full_rows[i] * blockvec[k];
      k += 1;
      i += width;
    }

    int col_idx = full_row_indices[col];
    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        &mul[col_idx], res);
  }
}


// Dense kernel for only a subset of rows
void DenseMatVecKernelBatched(
    const  float* __restrict__ vec,
    const  float* __restrict__ full_rows,
    const  int* __restrict__ full_row_indices,
           float* __restrict__ mul,
    int height,
    int width,
    int batch,
    int vec_height,
    int matwidth
,
    const sycl::nd_item<3> &item_ct1,
    float *blockvec) {

  int row = BLOCKWIDTH * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);

  for (int b = 0; b < batch; ++b){
    int i = width * row + col;
    int k = 0;
    float res = 0;

    item_ct1.barrier(sycl::access::fence_space::local_space);
    blockvec[item_ct1.get_local_id(2)] =
        vec[b * vec_height + row + item_ct1.get_local_id(2)];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (item_ct1.get_local_id(2) < width) {
      while (k < BLOCKWIDTH) {
        res += full_rows[i] * blockvec[k];
        k += 1;
        i += width;
      }

      int col_idx = full_row_indices[col];
      dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
          &mul[b * matwidth + col_idx], res);
    }
  }
}
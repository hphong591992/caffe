#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_mil_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxMILLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* aux_prob_data, const Dtype* label, const Dtype* selector_indicator, Dtype* loss,
          const Dtype lambda, const Dtype margin,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int similarity_value = static_cast<int>(selector_indicator[n * dim + label_value * spatial_dim + s]) == (n * dim + label_value * spatial_dim + s);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      const Dtype sample_loss = max(prob_data[n * dim + label_value * spatial_dim + s],
                                    Dtype(FLT_MIN));
      loss[index] = -log(sample_loss);
      const Dtype aux_sample_loss = max(aux_prob_data[n * dim + label_value * spatial_dim + s],
                                        Dtype(FLT_MIN));
      if (similarity_value == 0) {
        loss[index] -= (-lambda) * max(margin - (log(sample_loss) - log(aux_sample_loss)), 0.0);
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithMILLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  aux_softmax_layer_->Forward(aux_softmax_bottom_vec_, aux_softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* aux_prob_data = aux_prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* selector_indicator = bottom[3]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  const Dtype margin = this->layer_param_.softmax_with_max_margin_loss_param().margin();
  const Dtype lambda = this->layer_param_.softmax_with_max_margin_loss_param().lambda();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxMILLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, aux_prob_data, label, selector_indicator, loss_data, lambda, margin,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (normalize_) {
    Dtype count;
    caffe_gpu_asum(nthreads, counts, &count);
    loss /= count;
  } else {
    loss /= outer_num_;
  }
  top[0]->mutable_cpu_data()[0] = loss;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxMILLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, const Dtype* selector_indicator, Dtype* bottom_diff, Dtype* aux_bottom_diff, const Dtype lambda, const Dtype margin,
          const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    const int similarity_value = static_cast<int>(selector_indicator[n * dim + label_value * spatial_dim + s]) == (n * dim + label_value * spatial_dim + s);

    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
        aux_bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      if (similarity_value == 0) {
        const Dtype sample_prob = bottom_diff[n * dim + label_value * spatial_dim + s];
        const Dtype aux_sample_prob = aux_bottom_diff[n * dim + label_value * spatial_dim + s];
        if ((log(sample_prob) - log(aux_sample_prob)) < margin) {
          for (int c = 0; c < channels; ++c) {
            bottom_diff[n * dim + c * spatial_dim + s] *= (Dtype(1.0)+lambda);
            aux_bottom_diff[n * dim + c * spatial_dim + s] *= (-lambda);
          }
          bottom_diff[n * dim + label_value * spatial_dim + s] -= (Dtype(1.0)+lambda);
          aux_bottom_diff[n * dim + label_value * spatial_dim + s] -= (-lambda);
        } else {
          bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
          for (int c = 0; c < channels; ++c) {
            aux_bottom_diff[n * dim + c * spatial_dim + s] = 0;
          }
        }
      } else {
        bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        for (int c = 0; c < channels; ++c) {
          aux_bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
      }
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithMILLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    Dtype* aux_bottom_diff = bottom[2]->mutable_gpu_diff();
    const Dtype* aux_prob_data = aux_prob_.gpu_data();
    caffe_gpu_memcpy(aux_prob_.count() * sizeof(Dtype), aux_prob_data, aux_bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const Dtype* selector_indicator = bottom[3]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    const Dtype margin = this->layer_param_.softmax_with_max_margin_loss_param().margin();
    const Dtype lambda = this->layer_param_.softmax_with_max_margin_loss_param().lambda();
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxMILLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, selector_indicator, bottom_diff, aux_bottom_diff, lambda, margin,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      Dtype count;
      caffe_gpu_asum(nthreads, counts, &count);
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
      caffe_gpu_scal(aux_prob_.count(), loss_weight / count, aux_bottom_diff);
    } else {
      caffe_gpu_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
      caffe_gpu_scal(aux_prob_.count(), loss_weight / outer_num_, aux_bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithMILLossLayer);

}  // namespace caffe

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_max_margin_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithMaxMarginLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  aux_softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);

  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  aux_softmax_bottom_vec_.clear();
  aux_softmax_bottom_vec_.push_back(bottom[2]);
  aux_softmax_top_vec_.clear();
  aux_softmax_top_vec_.push_back(&aux_prob_);
  aux_softmax_layer_->SetUp(aux_softmax_bottom_vec_, aux_softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithMaxMarginLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  aux_softmax_layer_->Reshape(aux_softmax_bottom_vec_, aux_softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";

  //TODO: Repeat size checks for bottom[2]

  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithMaxMarginLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  aux_softmax_layer_->Forward(aux_softmax_bottom_vec_, aux_softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* aux_prob_data = aux_prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* similarity_value = bottom[3]->cpu_data();
  const Dtype margin = this->layer_param_.softmax_with_max_margin_loss_param().margin();
  const Dtype lambda = this->layer_param_.softmax_with_max_margin_loss_param().lambda();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      const Dtype sample_loss = std::max(prob_data[i * dim + label_value * inner_num_ + j],
                                         Dtype(FLT_MIN));
      loss -= log(sample_loss);

      const int sim = static_cast<int>(similarity_value[i*inner_num_ + j]);
      //TODO: Add checks for sim value
      const Dtype aux_sample_loss = std::max(aux_prob_data[i * dim + label_value * inner_num_ + j],
                                             Dtype(FLT_MIN));
      if (sim == 0) {
        loss -= (-lambda) * std::max(margin - (log(sample_loss) - log(aux_sample_loss)), 0.0);
      }
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithMaxMarginLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  //TODO: condition on propagate_down[2]
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    Dtype* aux_bottom_diff = bottom[2]->mutable_cpu_diff();
    const Dtype* aux_prob_data = aux_prob_.cpu_data();
    caffe_copy(aux_prob_.count(), aux_prob_data, aux_bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* similarity_value = bottom[3]->cpu_data();
    const Dtype margin = this->layer_param_.softmax_with_max_margin_loss_param().margin();
    const Dtype lambda = this->layer_param_.softmax_with_max_margin_loss_param().lambda();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const int sim = static_cast<int>(similarity_value[i*inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
            aux_bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          if (sim == 0) {
            Dtype sample_prob = bottom_diff[i * dim + label_value * inner_num_ + j];
            Dtype aux_sample_prob = aux_bottom_diff[i * dim + label_value * inner_num_ + j];
            if ((log(sample_prob) - log(aux_sample_prob)) < margin) {
              for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
                bottom_diff[i * dim + c * inner_num_ + j] *= (Dtype(1.0)+lambda);
                aux_bottom_diff[i * dim + c * inner_num_ + j] *= -lambda;
              }
              bottom_diff[i * dim + label_value * inner_num_ + j] -= (Dtype(1.0)+lambda);
              aux_bottom_diff[i * dim + label_value * inner_num_ + j] -= -lambda;
            } else {
              bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
              for (int c = 0; c < bottom[2]->shape(softmax_axis_); ++c) {
                aux_bottom_diff[i * dim + c * inner_num_ + j] = 0;
              }
            }
            sample_prob = bottom_diff[i * dim + label_value * inner_num_ + j];
            aux_sample_prob = aux_bottom_diff[i * dim + label_value * inner_num_ + j];
          } else {
            bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
            for (int c = 0; c < bottom[2]->shape(softmax_axis_); ++c) {
              aux_bottom_diff[i * dim + c * inner_num_ + j] = 0;
            }
          }
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
      caffe_scal(aux_prob_.count(), loss_weight / count, aux_bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
      caffe_scal(aux_prob_.count(), loss_weight / outer_num_, aux_bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithMaxMarginLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithMaxMarginLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithMaxMarginLoss);

}  // namespace caffe

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ext/stdio_filebuf.h>
#include <omp.h>


REGISTER_OP("CsvIter")
    .Input("data_file: string")
    .Output("labels: float32")
    .Output("signs: int64")
    .Attr("input_schema: list(string)")
    .Attr("feas: list(string)")
    .Attr("label: string = 'label' ")
    .Attr("batch_size: int = 10000")
    .Attr("buff_size: int = 4000000")
    .SetIsStateful()
;

const int BUFF_LINE_SIZE = 1024*1024;

using namespace tensorflow;
using std::cout;
using std::endl;

class CsvIterOp: public OpKernel {
 public:

  explicit CsvIterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("buff_size",  &buff_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_schema", &input_schema_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feas", &feas_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("label", &label_));
    label_idx_ = std::find(input_schema_.begin(), input_schema_.end(), label_) - input_schema_.begin();
    fea_idxs_.clear();
    for (auto fea : feas_) {
      fea_idxs_.push_back(std::find(input_schema_.begin(), input_schema_.end(), fea) - input_schema_.begin());
    }
    buff_head_ = 0;
    label_buff_.clear();
    buff_size_ = std::max(1, int(buff_size_/batch_size_))*batch_size_;
    n_valid_buff_ = 0;
    label_buff_.resize(buff_size_);
    sign_buff_.resize(fea_idxs_.size());
    for (int i = 0; i < fea_idxs_.size(); ++i) {
      sign_buff_[i].resize(buff_size_);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    //cout << "entering Compute " << endl; 
    if (n_valid_buff_ - buff_head_ <= 0) {
      fill_buff(ctx);
    }
    const int n_data = std::min(batch_size_, int32(n_valid_buff_ - buff_head_));
    const int n_fea = feas_.size();
    Tensor *label_tensor, *sign_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("labels", TensorShape({n_data}), &label_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output("signs",  TensorShape({n_data, n_fea}), &sign_tensor));
    auto label_data = label_tensor->flat<float>();
    auto sign_data = sign_tensor->tensor<int64, 2>();
    for (int i = 0; i < n_data; ++i) {
      label_data(i) = label_buff_[i+buff_head_];
    }
    for (int fid = 0; fid < n_fea ; ++fid) {
      for (int i = 0; i < n_data; ++i) {
        sign_data(i, fid) = sign_buff_[fid][i+buff_head_];
      }
    }
    buff_head_ += n_data;
  }

 private :
  void fill_buff(OpKernelContext* ctx) {
    const Tensor* data_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("data_file", &data_file_tensor));
    auto data_file = data_file_tensor->scalar<string>()();
    buff_head_ = 0;

    std::vector<string> data_lines;
    {
      mutex_lock l(mu_);
      if (data_file != current_data_file_) {
        //cout << "data_file[" << data_file << "] crt[" << current_data_file_ << "] fp[" << fp_ << "]" << endl;
        if (fp_ != NULL) {
          fclose(fp_);
          fp_ = NULL;
        }
        fp_ = fopen(data_file.c_str(), "r");
        OP_REQUIRES(ctx, fp_ != NULL, errors::InvalidArgument("Fails to open data file: ", data_file))
        current_data_file_ = data_file;
      }
      //cout << "handling new batch of " << current_data_file_ << endl; 
      string data_line;
      while (data_lines.size() < buff_size_) {
        if (NULL == fgets(buff_, BUFF_LINE_SIZE-1, fp_)) {
          break;  
        }
        data_lines.push_back(buff_);
      }
    }

    const int n_data = data_lines.size();
    const int n_fea = feas_.size();
    n_valid_buff_ = n_data;
    //num_threads(2)
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < n_data; ++i) {
      ParseLine(data_lines, i, label_buff_, sign_buff_);
    }
  }

  int32 batch_size_;
  int32 buff_size_ = 0;

  mutex mu_;
  std::string current_data_file_ = "";
  FILE *fp_ = NULL;
  char buff_[BUFF_LINE_SIZE];
  std::vector<std::string> input_schema_;
  std::vector<std::string> feas_;
  std::string label_;
  int label_idx_;
  std::vector<int> fea_idxs_;

  std::vector<float> label_buff_;
  std::vector<std::vector<int64>> sign_buff_;
  int buff_head_ = 0;
  int n_valid_buff_ = 0;

  template<typename T1, typename T2>
  void ParseLine(const std::vector<string>& lines, int lineid,
                 T1& label_data, T2& sign_data) {

    const char* p = lines[lineid].c_str();
    int offset;
    std::vector<uint64> row(input_schema_.size());
    for (size_t i = 0; i < input_schema_.size(); ++i) {
      sscanf(p, "%llu%n", &row[i], &offset);
      p += offset;
    }
    for (int fid : fea_idxs_) {
      //sign_data(lineid, fid) = row[fid] % 10;
      sign_data[fid][lineid] = row[fid] % 10;
    }
    label_data[lineid] = (row[label_idx_]>0);
  }
};

REGISTER_KERNEL_BUILDER(Name("CsvIter").Device(DEVICE_CPU), CsvIterOp);

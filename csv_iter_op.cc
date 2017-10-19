#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/hash/hash.h"
#include <ctime>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ext/stdio_filebuf.h>


REGISTER_OP("CsvIter")
    .Input("data_file: string")
    .Output("labels: float32")
    .Output("signs: int64")
    .Attr("input_schema: list(string)")
    .Attr("feas: list(string)")
    .Attr("label: string = 'label' ")
    .Attr("batch_size: int = 10000")
;

const int BUFF_LINE_SIZE = 1024*1024;

using namespace tensorflow;
using std::cout;
using std::endl;

class CsvIterOp: public OpKernel {
 public:

  explicit CsvIterOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("input_schema", &input_schema_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feas", &feas_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("label", &label_));
    label_idx_ = std::find(input_schema_.begin(), input_schema_.end(), label_) - input_schema_.begin();
    fea_idxs_.clear();
    for (auto fea : feas_) {
      fea_idxs_.push_back(std::find(input_schema_.begin(), input_schema_.end(), fea) - input_schema_.begin());
    }
  }

  void Compute(OpKernelContext* ctx) override {
    cout << "entering Compute " << endl; 
    const Tensor* data_file_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("data_file", &data_file_tensor));
    auto data_file = data_file_tensor->scalar<string>()();

    std::vector<string> data_lines;
    {
      mutex_lock l(mu_);
      if (data_file != current_data_file_) {
        cout << "data_file[" << data_file << "] crt[" << current_data_file_ << "] fp[" << fp_ << "]" << endl;
        if (fp_ != NULL) {
          fclose(fp_);
          fp_ = NULL;
        }
        fp_ = fopen(data_file.c_str(), "r");
        OP_REQUIRES(ctx, fp_ != NULL, errors::InvalidArgument("Fails to open data file: ", data_file))
        current_data_file_ = data_file;
      }
      cout << "handling new batch of " << current_data_file_ << endl; 
      string data_line;
      while (data_lines.size() < batch_size_) {
        if (NULL == fgets(buff_, BUFF_LINE_SIZE-1, fp_)) {
          break;  
        }
        data_lines.push_back(buff_);
      }
    }

    int64 n_data = data_lines.size();
    int64 n_fea = feas_.size();

    Tensor *label_tensor, *sign_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("labels", TensorShape({n_data}), &label_tensor));
    OP_REQUIRES_OK(ctx, ctx->allocate_output("signs",  TensorShape({n_data, n_fea}), &sign_tensor));
    auto label_data = label_tensor->flat<float>();
    auto sign_data = sign_tensor->tensor<int64, 2>();

    for (size_t i = 0; i < data_lines.size(); ++i) {
      ParseLine(ctx, data_lines, i, label_data, sign_data);
    }
  }

 private :
  int32 batch_size_;

  mutex mu_;
  std::string current_data_file_ = "";
  FILE *fp_ = NULL;
  char buff_[BUFF_LINE_SIZE];
  std::vector<std::string> input_schema_;
  std::vector<std::string> feas_;
  std::string label_;
  int label_idx_;
  std::vector<int> fea_idxs_;

  template<typename T1, typename T2>
  void ParseLine(OpKernelContext* ctx, const std::vector<string>& lines, int lineid,
                 T1& label_data, T2& sign_data) {

    const char* p = lines[lineid].c_str();
    int offset;
    std::vector<uint64> row(input_schema_.size());
    for (size_t i = 0; i < input_schema_.size(); ++i) {
      sscanf(p, "%llu%n", &row[i], &offset);
      p += offset;
    }
    for (size_t i : fea_idxs_) {
       sign_data(lineid, i) = row[i] % 10;
    }
    label_data(lineid) = (row[label_idx_]>0);
  }
};

REGISTER_KERNEL_BUILDER(Name("CsvIter").Device(DEVICE_CPU), CsvIterOp);

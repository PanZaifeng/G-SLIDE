#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "GPUTimer.h"
#include "Network.h"
#include "jsoncpp/json/json.h"
#include "utils.h"

std::vector<int> jarr_to_vec(const Json::Value &jarr) {
  const size_t size = jarr.size();
  std::vector<int> res(size);
  for (int i = 0; i < size; ++i) {
    res[i] = jarr[i].asInt();
  }

  return res;
}

// Return: real batch size
int get_batch_data(std::ifstream &ist, std::vector<int> &h_cmprs_input_nodes,
                   std::vector<float> &h_cmprs_input_vals,
                   std::vector<int> &h_cmprs_input_offsets,
                   std::vector<int> &h_cmprs_labels,
                   std::vector<int> &h_cmprs_label_offsets,
                   const int batch_size) {
  h_cmprs_input_nodes.clear();
  h_cmprs_input_vals.clear();
  h_cmprs_input_offsets.clear();
  h_cmprs_labels.clear();
  h_cmprs_label_offsets.clear();

  h_cmprs_input_offsets.push_back(0);
  h_cmprs_label_offsets.push_back(0);

  for (int b = 0; b < batch_size; ++b) {
    int label;
    if (ist >> label) {
      h_cmprs_labels.push_back(label);
    } else {
      return b;
    }

    while (ist.get() == ',') {
      ist >> label;
      h_cmprs_labels.push_back(label);
    }
    h_cmprs_label_offsets.push_back(h_cmprs_labels.size());

    do {
      int node;
      ist >> node;
      assert(ist.get() == ':');

      float val;
      ist >> val;
      h_cmprs_input_nodes.push_back(node);
      h_cmprs_input_vals.push_back(val);
    } while (ist.get() == ' ');
    h_cmprs_input_offsets.push_back(h_cmprs_input_nodes.size());
  }

  return batch_size;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s config.json\n", argv[0]);
    exit(1);
  }

  Json::Reader reader;
  Json::Value root;
  std::ifstream config_ist(argv[1]);
  if (!reader.parse(config_ist, root)) {
    printf("Parse %s failed!\n", argv[1]);
    exit(1);
  }

  const std::vector<int> node_num_per_layer =
      jarr_to_vec(root["node_num_per_layer"]);
  const std::vector<int> node_capacity_per_layer =
      jarr_to_vec(root["node_capacity_per_layer"]);
  const int input_size = root["input_size"].asInt();
  const int max_batch_size = root["max_batch_size"].asInt();
  const int input_capacity = root["input_capacity"].asInt();
  const int label_capacity = root["label_capacity"].asInt();
  const int K = root["K"].asInt(), L = root["L"].asInt();
  const int bin_size = root["bin_size"].asInt();
  const int bucket_num_per_tbl = root["bucket_num_per_tbl"].asInt();
  const int bucket_capacity = root["bucket_capacity"].asInt();
  const int threshold = root["threshold"].asInt();
  const int tbl_num_per_tile = root["tbl_num_per_tile"].asInt();
  const int tbl_num_per_thread = root["tbl_num_per_thread"].asInt();
  const int linked_bucket_num_per_tbl =
      root["linked_bucket_num_per_tbl"].asInt();
  const int linked_pool_size = root["linked_pool_size"].asInt();

  Network network(node_num_per_layer, node_capacity_per_layer, input_size,
                  max_batch_size, input_capacity, label_capacity, K, L,
                  bin_size, bucket_num_per_tbl, bucket_capacity, threshold,
                  tbl_num_per_tile, tbl_num_per_thread,
                  linked_bucket_num_per_tbl, linked_pool_size);

  const std::vector<int> max_act_nums = jarr_to_vec(root["max_act_nums"]);
  const int max_label_num = root["max_label_num"].asInt();
  const float lr = root["lr"].asFloat();
  const float BETA1 = root["BETA1"].asFloat();
  const float BETA2 = root["BETA2"].asFloat();
  const int rebuild_period = root["rebuild_period"].asInt();
  const int thread_num = root["thread_num"].asInt();
  const int epoch_num = root["epoch_num"].asInt();

  const std::string train_fname = root["train_fname"].asString();
  const std::string test_fname = root["test_fname"].asString();

  GPUTimer timer;
  float tot_time = 0;

  for (int e = 0; e < epoch_num; e++) {
    printf("------------------- Epoch %d ---------------------\n", e);
    std::ifstream train_ist(train_fname);
    std::ifstream test_ist(test_fname);
    if (!train_ist || !test_ist) {
      std::cerr << "Cannot open dataset file!" << std::endl;
      exit(-1);
    }

    std::string header;
    std::getline(train_ist, header);  // skip header
    std::getline(test_ist, header);   // skip header

    int batch_size;
    int cnt = 0;
    do {
      std::vector<int> h_cmprs_input_nodes;
      std::vector<float> h_cmprs_input_vals;
      std::vector<int> h_cmprs_input_offsets;
      std::vector<int> h_cmprs_labels;
      std::vector<int> h_cmprs_label_offsets;
      batch_size =
          get_batch_data(train_ist, h_cmprs_input_nodes, h_cmprs_input_vals,
                         h_cmprs_input_offsets, h_cmprs_labels,
                         h_cmprs_label_offsets, max_batch_size);

      const float tmplr =
          lr * sqrt((1 - pow(BETA2, cnt + 1))) / (1 - pow(BETA1, cnt + 1));
      const bool rebuild = cnt % 5 == 4;

      timer.start();

      network.train(h_cmprs_input_nodes, h_cmprs_input_vals,
                    h_cmprs_input_offsets, h_cmprs_labels,
                    h_cmprs_label_offsets, max_act_nums, batch_size, tmplr,
                    max_label_num, thread_num, rebuild);

      tot_time += timer.record("[BATCH " + std::to_string(cnt) + "] ");

      cnt++;
      // if (cnt > 10) break;

    } while (batch_size == max_batch_size);
    network.rebuild();

    printf("Current elapsed time %f ms\n", tot_time);

    // eval
    int correct_cnt = 0, test_cnt = 0;
    do {
      std::vector<int> h_cmprs_input_nodes;
      std::vector<float> h_cmprs_input_vals;
      std::vector<int> h_cmprs_input_offsets;
      std::vector<int> h_cmprs_labels;
      std::vector<int> h_cmprs_label_offsets;
      batch_size =
          get_batch_data(test_ist, h_cmprs_input_nodes, h_cmprs_input_vals,
                         h_cmprs_input_offsets, h_cmprs_labels,
                         h_cmprs_label_offsets, max_batch_size);

      timer.start();

      correct_cnt += network.eval(
          h_cmprs_input_nodes, h_cmprs_input_vals, h_cmprs_input_offsets,
          h_cmprs_labels, h_cmprs_label_offsets, batch_size, thread_num);

      // timer.record("Infer time ");

      test_cnt += batch_size;
      // if (test_cnt >= 512) break;

    } while (batch_size == max_batch_size);

    printf("Test %d records, %d correct; accuracy: %f\n", test_cnt, correct_cnt,
           ((float)correct_cnt) / test_cnt);
  }
}

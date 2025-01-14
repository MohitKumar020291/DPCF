#include <torch/torch.h>
#include <vector>
#include <time.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <sys/time.h>
#include <filesystem>
#include <fstream>
#include <cmath>

//ASCII only
std::string allowed_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,;'";
int n_letters = allowed_characters.length();

torch::Device device = torch::kCPU;

//-----------------------------------------------//
void initialize_device() {
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device\n";
    } else {
        std::cout << "Using CPU device\n";
    }
}
//-----------------------------------------------//

//-----------------------------------------------//
int letterToIndex(char letter) {
    auto pos = std::find(allowed_characters.begin(), allowed_characters.end(), letter);
    if (pos == allowed_characters.end()) {
        return -1; // Return an invalid index for undefined characters.
    }
    return pos - allowed_characters.begin();
}
//-----------------------------------------------//

//-----------------------------------------------//
torch::Tensor nameToTensor(std::string &name) {
    torch::Tensor tensor = torch::zeros({(long)name.length(), 1, n_letters}, device);
    for (int i=0; i<name.length(); i++) {
        // tensor[i][0][letterToIndex(name[i])] = 1;
        tensor.index_put_({i, 0, letterToIndex(name[i])}, 1);
    }
    return tensor;
}
//-----------------------------------------------//

//-----------------------------------------------//
void list_txt_files(const std::string &data_dir, std::vector<std::string> &txt_files) {
    namespace fs = std::filesystem; //we will use fs instead of std::filesystem in list_txt_files
    for (const auto &entry : fs::directory_iterator(data_dir)) { 
        if (entry.is_regular_file() && entry.path().extension() == ".txt") { 
            std::string filename = entry.path().filename().string();
            txt_files.push_back(filename); 
        } 
    }
}
//-----------------------------------------------//

//-----------------------------------------------//
std::pair<std::string, int> label_from_output(torch::Tensor output, const std::vector<std::string> &labels_set) {
    // Ensure the output tensor has only one row (batch size = 1).
    if (output.size(0) != 1) {
        throw std::runtime_error("Output tensor must have a batch size of 1 for this function.");
    }

    // Get the index of the maximum value for the single row.
    auto index_tensor = output.argmax(1); // Shape: [1]
    int index = index_tensor.item<int>(); // Convert the scalar tensor to an integer.
    
    // Ensure the index is within the labels set bounds.
    if (index < 0 || index >= static_cast<int>(labels_set.size())) {
        throw std::runtime_error("Index out of bounds for labels_set.");
    }

    std::string label = labels_set[index];
    return std::make_pair(label, index);
}

//-----------------------------------------------//

//-----------------------------------------------//
std::pair<torch::Tensor, torch::Tensor> split_train_test(NamedDataset alldata, float train_size_) {
    auto dataset_size = alldata.size(0);
    auto train_size = static_cast<int64_t>(dataset_size * train_size_);
    auto test_size = dataset_size - train_size;

    torch::manual_seed(42);
    auto dataset_indices = torch::randperm(dataset_size, torch::TensorOptions().device(device));

    auto train_indices = dataset_indices.narrow(0, 0, train_size);
    auto test_indices = dataset_indices.narrow(0, train_size, test_size);

    auto train_data = alldata.index({train_indices});
    auto test_data = alldata.index({test_indices});

    return std::make_pair(train_data, test_data);
}
//-----------------------------------------------//

//-----------------------------------------------//
std::vector<std::string> readLinesFromFile(const std::string &file_path) {
    std::ifstream file(file_path);
    std::vector<std::string> lines;
    std::string name;

    if (file.is_open()) {
        while (std::getline(file, name)) {
            //delete any leading whitespace
            name.erase(0, name.find_first_not_of(" \t\r\n"));
            //delete any trailing whitespace
            //the last non whitespace - +1 makes removing all the next whitespaces
            name.erase(name.find_last_not_of(" \t\r\n") + 1);
            //push the line if it is not empty
            if (!name.empty()) {
                lines.push_back(name);
            }
        }
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
    }

    return lines;
} 
//-----------------------------------------------//

//-----------------------------------------------//
class TensorArray {
    public:
        std::vector<torch::Tensor> tensors;
        int size = 0;
        
        void addTensor(torch::Tensor tensor) {
            tensors.push_back(tensor);
            size++;
            std::cout << "tensor added to TensorArray successfully"<<std::endl;
        }

        void printtensors() {
            for (size_t i = 0; i < tensors.size(); ++i) {
                std::cout << "Tensor " << i + 1 << ":\n" << tensors[i] << "\n";
            }
        }
};
//-----------------------------------------------//

//-----------------------------------------------//
#define MAX_FILENAME_LEN 200
class NameDataset : public torch::data::Dataset<NameDataset> {
    public:
        std::string data_dir;
        // std::time_t load_time; //usused
        std::vector<std::string> labels_set;

        std::vector<std::string> data;
        TensorArray data_tensors;
        std::vector<std::string> labels;
        TensorArray label_tensors;

        NameDataset(std::string data_dir = "NONE", torch::Tensor label_tensors_ = torch::tensor({}), torch::Tensor data_tensors_ = torch::tensor({})) {
            if (data_dir != "NONE") { 
                this->data_dir = data_dir;
                this->load();
            } else if (data_tensors_.size(0) != 0 && label_tensors_.size(0) != 0) {
                this->data_dir="None"; //shows that the tensors were passed directly
                label_tensors_.to(device);
                data_tensors_.to(device);
                this->data_tensors.tensors = data_tensors_;
                this->label_tensors.tensors = label_tensors_;

                for (auto data_tensor_ : label_tensors_) {
                    this->data_tensors.size += 1;
                    this->label_tensors.size += 1;
                }
            }
        }

        void load() {
            std::vector<std::string> txt_files;
            list_txt_files(this->data_dir, txt_files);
            for (std::string filename : txt_files) {
                int dot = filename.find('.'); //index of dot
                std::string label = filename.substr(0, dot); //from 0 to dot
                if (std::find(this->labels_set.begin(), this->labels_set.end(), label) == this->labels_set.end()) {
                    this->labels_set.push_back(label);
                }
                std::string file_path = this->data_dir + "/" + filename;
                std::vector<std::string> lines = readLinesFromFile(file_path);

                for (std::string name : lines) {
                    this->data.push_back(name);
                    this->data_tensors.addTensor(nameToTensor(name));
                    this->labels.push_back(label);
                }
            }

            for (int i=0; i<this->labels.size(); i++) {
                auto it = std::find(this->labels_set.begin(), this->labels_set.end(), this->labels[i]);
                int index = std::distance(this->labels_set.begin(), it); //it - this->labels_set.begin()
                this->label_tensors.addTensor(torch::tensor({index}, torch::kInt).to(device));
            }
        }

        c10::optional<size_t> size() const override {
            return static_cast<size_t>(this->label_tensors.size); //type of label_tensors = TensorArray
        }

        torch::data::Example<at::Tensor, at::Tensor> get(size_t index) override {
            at::Tensor label_tensor = this->label_tensors.tensors[index];
            at::Tensor data_tensor = this->data_tensors.tensors[index];
            return {data_tensor, label_tensor};
        }

};
//-----------------------------------------------//

//-----------------------------------------------//
struct CharRNN : torch::nn::Module {
    torch::nn::RNN rnn{nullptr};
    torch::nn::Linear h2o{nullptr};

    CharRNN(int input_size, int hidden_size, int output_size)
    : rnn(torch::nn::RNNOptions(input_size, hidden_size)), 
      h2o(hidden_size, output_size)
    {
        register_module("rnn", this->rnn);
        register_module("h2o", this->h2o);
        this->to(device);
        std::cout << "char rnn initialization completed"<<std::endl;
    }
    torch::Tensor forward(torch::Tensor name_tensor) {
        std::tuple<at::Tensor, at::Tensor> rnn_out = this->rnn(name_tensor);
        auto hidden_state = std::get<0>(std::get<1>(rnn_out));
        auto output = this->h2o(hidden_state);
        output = torch::log_softmax(output, /*dim=*/1);
        return output;
    }

};
//-----------------------------------------------//

//-----------------------------------------------//
std::vector<float> train(
    CharRNN rnn, 
    NameDataset training_data, 
    int n_batch_size = 64, 
    int report_every = 10, 
    int n_epoch = 10, 
    double learning_rate = 0.2) {

    // auto criterion = torch::nn::functional::nll_loss(torch::nn::functional::NLLLossFuncOptions().reduction(torch::kMean));
    auto criterion = torch::nn::functional::nll_loss.to(device);
    int data_size = training_data.size().value_or(0);
    float current_loss = 0.0;
    std::vector<float> all_losses;
    rnn.train();
    torch::optim::SGD optimizer(rnn->parameters(), torch::optim::SGDOptions(learning_rate)).to(device);

    for (int i=1; i<n_epoch+1; i++) {
        rnn.zero_grad();
        torch::Tensor batches = torch::arange(0, data_size, torch::TensorOptions().dtype(torch::kInt).device(device));
        auto random_indices = torch::randperm(data_size, torch::TensorOptions().device(device));
        batches = batches.indexSelect(0, random_indices);
        batches = torch::tensor_split(batches, data_size/n_batch_size);

        for (torch::Tensor batch : batches) {
            torch::Tensor batch_loss = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat));
            for (float j : batch) {
                int idx = static_cast<int>j; //i think we use dynamic_cast if we don't know the j type - like auto?
                auto [label_tensor, data_tensor] = training_data.get(idx);
                torch::Tensor output = rnn->forward(data_tensor);
                auto loss = criterion(output, label_tensor);
                batch_loss += loss;
            }
            batch_loss.backward();
            torch::nn::utils::clip_grad_norm_(rnn->parameters(), 3);
            optimizer.step();
            optimizer.zero_grad();
            current_loss += batch_loss.item<double>() / batch.size(0);
        }
        all_losses.push_back(current_loss);
        if (i % report_every == 0) {
            std::cout << "Epoch: " << i << " Loss: " << current_loss << std::endl;
            current_loss = 0.0;
        }
    }
    return all_losses;
};
//-----------------------------------------------//

//-----------------------------------------------//
int main() {
    initialize_device();

    std::string data_dir = "../../../Downloads/LJSpeech-1.1/data-wnlstms/names";
    NameDataset alldata(data_dir);
    std::cout << "Size of the data: " << alldata.size().value_or(0) << "\n";
    std::cout << "Size of label set: " << alldata.labels_set.size() << "\n";
    std::cout << "First label: " << alldata.labels_set[0] << "\n";

    //creating CharRNN and testing
    int n_hiddens = 128;
    CharRNN rnn = CharRNN(n_letters, n_hiddens, alldata.labels_set.size()); //even after split this is not going to change
    std::string name = "Albert";
    auto input = nameToTensor(name);
    auto output = rnn.forward(input);
    std::cout << output.size(0) << "\n";

    //splitting the data
    std::tuple<at::Tensor, at::Tensor> train_set, test_set = split_train_test(alldata, 0.85);
    //converting to namespace - sp can use indexing
    NameDataset n_train_set(label_tensors_ = train_set[0], data_tensors_ = train_set[1]);
    NameDataset n_test_set(label_tensors_ = test_set[0], data_tensors_ = test_set[1]);

    //training a model
    std::vector<float> all_losses = train(rnn, n_train_set);
    return 0;
}
//-----------------------------------------------//

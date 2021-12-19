#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "../fixedMatrix/fmtx.h"

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

#define SIGMOID 0
#define RELU 1


typedef struct fixedNNLayer {
    uint8_t layerType; //could be weights layer or hidden layer.
    uint8_t activationType; //could be sigmoid, tanh, relu, etc.
    uint8_t num_neurons; //number of neurons in the layer *might not be needed if we use matrix data.
    fixedMatrix* data;  //data for each layer
    fixedMatrix* weights; //these are the weights that will calculate the data. so Ws-->Data
    fixedMatrix* bias;  //bias is the bias that will be added to the data. TODO: implement it.
    struct fixedNNLayer* previous;  //the previous layer.
} fixedNNLayer;

//struct that contains the entire NN data
typedef struct fixedNN {
    uint8_t num_layers;
    fix16_t learning_rate;
    fixedNNLayer** layers; 
} fixedNN;

fixedNN* createNN(uint8_t numInputs); //only start with inputs. Every layer added could be hidden or output.
void freeNN(fixedNN* nn);
void freeNNLayer(fixedNNLayer* layer);
void forwardLayer(fixedNNLayer* layer);
void forwardNN(fixedNN* nn);
void initNNInputLayer(fixedNN* nn, fixedMatrix* inputs);
fixedNNLayer* addLayer(fixedNN* nn, uint8_t numNeurons, uint8_t layerType, uint8_t activationType); //adds a layer to the NN.
void initLayerWeights(fixedNNLayer* layer);
void setLearningRate(fixedNN* nn, fix16_t learningRate); //sets the learning rate for the NN.
void setInputs(fixedNN* nn, fixedMatrix* inputs); //sets the inputs for the NN.
void freeNN(fixedNN* nn); //frees the NN.
fixedNNLayer* createNNLayer(uint8_t layerType, uint8_t numNeurons); //0 for hidden, 1 for output. Weights random. bias 0 for now.
void initNNInput(fixedNN* nn, fixedMatrix* inputs); //initializes the input layer on NN creation. No weights on inputlayer.

void trainNN(fixedNN* nn, fixedMatrix* input, fixedMatrix* output); //trains the NN.

void printNNLayer(fixedNNLayer* layer); //prints the layer.

fix16_t sigmoid_function(fix16_t x);
fix16_t sigmoid_function_deriv(fix16_t x);
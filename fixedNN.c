#include "fixedNN.h"

fixedNN* createNN(uint8_t numInputs){
    fixedNN* nn = (fixedNN*)malloc(sizeof(fixedNN));
    nn->num_layers = 1; //only set up the input layer but we can set the values of it later.
    nn->learning_rate = fix16_from_float(0.3);
    nn->layers = (fixedNNLayer**)malloc(sizeof(fixedNNLayer*) * nn->num_layers);
    nn->layers[0] = createNNLayer(INPUT_LAYER, numInputs); //just init the layer and init the data to numInputs of all zeroes.
    nn->layers[0]->data = fixedMatrix_create(numInputs, 1);
    return nn;
}

void freeNN(fixedNN* nn){
    for(int i = 1; i < nn->num_layers; i++){
        freeNNLayer(nn->layers[i]);
    }
    //free the input layer stuff
    freeFixedMatrix(nn->layers[0]->data);
    free(nn->layers[0]);
    free(nn->layers);
    free(nn);
}

void freeNNLayer(fixedNNLayer* layer){
    freeFixedMatrix(layer->weights);
    freeFixedMatrix(layer->bias);
    freeFixedMatrix(layer->data);
    free(layer);
}

fixedNNLayer* addLayer(fixedNN* nn, uint8_t numNeurons, uint8_t layerType, uint8_t activationType){
    nn->layers = (fixedNNLayer**)realloc(nn->layers, sizeof(fixedNNLayer*) * (nn->num_layers + 1));
    nn->num_layers ++;
    nn->layers[nn->num_layers - 1] = createNNLayer(layerType, numNeurons);
    nn->layers[nn->num_layers - 1]->activationType = activationType; //set activationType
    nn->layers[nn->num_layers - 1]->previous = nn->layers[nn->num_layers - 2];
    initLayerWeights(nn->layers[nn->num_layers - 1]); //init weights on new layer
    // forwardLayer(nn->layers[nn->num_layers - 1]); //forward the new layer TODO: do this later.
    return nn->layers[nn->num_layers - 1];
}

void setInputs(fixedNN* nn, fixedMatrix* inputs){ //we only set it once the NN is finalized. so we can point prev to the right spot.
    nn->layers[0]->data = inputs; //set the inputs
    nn->layers[1]->previous = nn->layers[0]; //set the previous layer to the input layer.
    return; //very simple.
}


fixedNNLayer* createNNLayer(uint8_t layerType, uint8_t numNeurons){
    fixedNNLayer* layer = (fixedNNLayer*)malloc(sizeof(fixedNNLayer));
    layer->layerType = layerType;
    layer->num_neurons = numNeurons;
    layer->weights = NULL;
    layer->previous = NULL;
    layer->bias = NULL;
    return layer;
}

void forwardLayer(fixedNNLayer* layer){
    if(layer->layerType == INPUT_LAYER){
        return;
    }
    if(layer->previous == NULL){
        return;
    }
    if(layer->weights == NULL){
        initLayerWeights(layer);
    }
    //forward the layer
    //printf("grabbinf input data\n");
    fixedMatrix* input = layer->previous->data;
    if(layer->data){
        freeFixedMatrix(layer->data);
    }
    layer->data = staticMatrixMultiply(layer->weights, input); //matrix mult the weights and input to get data. TODO: add bias later.
    
    //time to do activation function!
    if(layer->activationType == SIGMOID){
        nonStaticMatrixMap(layer->data, layer->data, sigmoid_function); //maps all values to sigmoid_function
    }
    else if(layer->activationType == RELU){
        // relu(layer->data);
    }
}

void trainNN(fixedNN* nn, fixedMatrix* input, fixedMatrix* output){
    setInputs(nn, input); //set the input to the NN
    forwardNN(nn); //forward the NN

    //backpropogate the error
    fixedNNLayer* currentLayer = nn->layers[nn->num_layers - 1]; //set the current layer to the output layer.
    //calculate the error.
    fixedMatrix* output_error = staticMatrixSubtract(output, currentLayer->data); //targets - outputs


    //calculate the gradient
    fixedMatrix* output_gradient = staticMatrixMap(currentLayer->data, sigmoid_function_deriv); //sigmoid_gradient(outputs)
    nonstaticMatrixHadamard(output_gradient, output_error, output_gradient); //hadamard product of the error and the gradient.
    nonstaticMatrixMultiplyScalar(output_gradient, nn->learning_rate); //multiply by the learning rate.

    fixedMatrix* hidden_T = staticMatrixTranspose(currentLayer->previous->data); //transpose the previous layer's data.
    fixedMatrix* delta_weights = staticMatrixMultiply(output_gradient, hidden_T); //multiply the transposed previous layer by the gradient.

    nonStaticMatrixAdd(currentLayer->weights, delta_weights, currentLayer->weights); //add the delta weights to the weights.

    // printf("back propogated final weights.\n");
    //freeFixedMatrix(delta_weights);

    //backpropogate the first layer.
    //calculate the hidden layers error.
    fixedMatrix* hidden_error;
    while(currentLayer->previous->layerType != INPUT_LAYER){
        fixedMatrix* hid_T = staticMatrixTranspose(currentLayer->weights); //transpose the weights of the previous layer.
        hidden_error = staticMatrixMultiply(hid_T, output_error); //multiply the transposed weights by the error.
        freeFixedMatrix(output_error);
        freeFixedMatrix(hid_T);

        fixedMatrix* hidden_gradient = staticMatrixMap(currentLayer->previous->data, sigmoid_function_deriv); //sigmoid_gradient(previous layer)
        nonstaticMatrixHadamard(hidden_gradient, hidden_error, hidden_gradient); //hadamard product of the error and the gradient.
        nonstaticMatrixMultiplyScalar(hidden_gradient, nn->learning_rate); //multiply by the learning rate.
        currentLayer = currentLayer->previous; //set the current layer to the previous layer.
        fixedMatrix* input_T = staticMatrixTranspose(currentLayer->previous->data); //transpose the previous layer's data.
        delta_weights = staticMatrixMultiply(hidden_gradient, input_T); //multiply the transposed previous layer by the gradient.
        freeFixedMatrix(input_T);
        nonStaticMatrixAdd(currentLayer->weights, delta_weights, currentLayer->weights); //add the delta weights to the weights.
        freeFixedMatrix(hidden_gradient);

        output_error = hidden_error; //set the output error to the hidden error.
        // printf("back propogated hidden weights.\n");
    }

}



void forwardNN(fixedNN* nn){
    for(int i = 1; i < nn->num_layers; i++){
        forwardLayer(nn->layers[i]);
    }
}

void initLayerWeights(fixedNNLayer* layer){
    if(layer->layerType == INPUT_LAYER){ //no need to init weights on input layer
        return;
    }
    layer->weights = fixedMatrix_create(layer->num_neurons, layer->previous->num_neurons); //create weights matrix.
    layer->bias = fixedMatrix_create(layer->num_neurons, 1); //create bias matrix.
    fixedMatrix_set_all_random(layer->weights); //set random weights.
    return;
}

void printNNLayer(fixedNNLayer* layer){
    printf("Layer Type: %d\n", layer->layerType);
    printf("Number of Neurons: %d\n", layer->num_neurons);
    if(layer->data){
        printf("Layer Data:\n");
        printFixedMatrix(layer->data);
    }
    if(layer->weights){
        printf("Layer Weights:\n");
        printFixedMatrix(layer->weights);
    }
    
    if(layer->bias){
        printf("Layer Bias:\n");
        printFixedMatrix(layer->bias);
    }
    return;
}

fix16_t sigmoid_function(fix16_t x){
    return fix16_div(fix16_one, fix16_add(fix16_one, fix16_exp( fix16_mul(fix16_from_float(-1.0), x) )));
}

fix16_t sigmoid_function_deriv(fix16_t y){
    return fix16_mul(y, fix16_sub(fix16_one, y)); // y*(1-y)
}
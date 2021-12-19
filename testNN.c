#include "fixedNN.h"

int main(int argc, char *argv[])
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    srand(tv.tv_usec);

    //will try to create the XOR

    fixedMatrix* input = fixedMatrix_from_ints(2, 1, (int[]){1, 0}); // [1, 0] input
    fixedMatrix* target = fixedMatrix_from_ints(2, 1, (int[]){1, 0}); // [1] target output

    fixedMatrix* input1 = fixedMatrix_from_ints(2, 1, (int[]){0,1}); // [1, 0] input
    fixedMatrix* target1 = fixedMatrix_from_ints(2, 1, (int[]){1, 0}); // [1] target output

    fixedMatrix* input2 = fixedMatrix_from_ints(2, 1, (int[]){0,0}); // [1, 0] input
    fixedMatrix* target2 = fixedMatrix_from_ints(2, 1, (int[]){0, 1}); // [1] target output

    fixedMatrix* input3 = fixedMatrix_from_ints(2, 1, (int[]){1, 1}); // [1, 0] input
    fixedMatrix* target3 = fixedMatrix_from_ints(2, 1, (int[]){0, 1}); // [1] target output


    fixedNN* mainNN = createNN(2);
    fixedNNLayer* layerOne = addLayer(mainNN, 10, HIDDEN_LAYER, SIGMOID);
    fixedNNLayer* layerTwo = addLayer(mainNN, 7, HIDDEN_LAYER, SIGMOID);
    fixedNNLayer* layerTwoTwo = addLayer(mainNN, 5, HIDDEN_LAYER, SIGMOID);
    fixedNNLayer* layerThree = addLayer(mainNN, 2, OUTPUT_LAYER, SIGMOID);

    fixedMatrix* inputs[] = {input, input1, input2, input3};
    fixedMatrix* targets[] = {target, target1, target2, target3};


    for(int i = 0; i< 10000; i++){
        uint8_t randIndex = rand() % 4;
        trainNN(mainNN, inputs[randIndex], targets[randIndex]);
        printf("trained iteration: %d\n", i);
    }

   
    printf("\nafter training\n\n");
    //feed an input

    printf("XOR on (1,0) = 1\n");
    setInputs(mainNN, input);
    forwardNN(mainNN);
    printFixedMatrix(mainNN->layers[mainNN->num_layers-1]->data); //prints the output matrix of the NN

    printf("XOR on (0,1) = 1\n");
    setInputs(mainNN, input1);
    forwardNN(mainNN);
    printFixedMatrix(mainNN->layers[mainNN->num_layers-1]->data); //prints the output matrix of the NN
    printf("%d %d", mainNN->layers[mainNN->num_layers-1]->data->data[0], mainNN->layers[mainNN->num_layers-1]->data->data[1]);
    

    printf("XOR on (0, 0) = 0\n");
    setInputs(main, input2);
    forwardNN(main);
    printFixedMatrix(main->layers[main->num_layers-1]->data); //prints the output matrix of the NN

    printf("XOR on (1,1) = 0\n");
    setInputs(main, input3);
    forwardNN(main);
    printFixedMatrix(main->layers[main->num_layers-1]->data); //prints the output matrix of the NN


    freeNN(main);
    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.01
#define EPOCHS 100

// NUERAL NETOWRK STRUCTURE
typedef struct
{
  double weightsInputHidden[HIDDEN_SIZE][INPUT_SIZE];
  double weightsOutputHidden[OUTPUT_SIZE][HIDDEN_SIZE];
  double biasHidden[HIDDEN_SIZE];
  double biasOutput[OUTPUT_SIZE];
} NeuralNetwork;

// sigmoid activation function
double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x)); // why? because we want to output value between 0 and 1 since it is a probability
}

// forward propagation
double forwardPropagation(NeuralNetwork *nn, double input[INPUT_SIZE])
{
  // hidden layer
  double hidden[HIDDEN_SIZE];
  for (int i = 0; i < HIDDEN_SIZE; ++i)
  {
    hidden[i] = 0;
    for (int j = 0; j < INPUT_SIZE; ++j)
    {
      hidden[i] += nn->weightsInputHidden[i][j] * input[j];
    }
    hidden[i] = sigmoid(hidden[i] + nn->biasHidden[i]);
  }

  // output layer
  double output = 0;
  for (int i = 0; i < OUTPUT_SIZE; ++i)
  {
    output += nn->weightsOutputHidden[i][0] * hidden[i];
  }
  output = sigmoid(output + nn->biasOutput[0]);

  return output;
  
}

// backpropagation
void backwardPropagation(NeuralNetwork *nn,double input[INPUT_SIZE], double target)
{
  // forward pass
  double hidden[HIDDEN_SIZE];
  for (int i = 0; i < HIDDEN_SIZE; ++i)
  {
    hidden[i] = 0;
    for (int j = 0; j < INPUT_SIZE; ++j)
    {
      hidden[i] += nn->weightsInputHidden[i][j] * input[j];
    }
    hidden[i] = sigmoid(hidden[i] + nn->biasHidden[i]);
    
  }

  double output = 0;
  for (int i = 0; i < OUTPUT_SIZE; ++i)
  {
    output += nn->weightsOutputHidden[i][0] * hidden[i];
  }
  output = sigmoid(output + nn->biasOutput[0]);

  // backward pass
  double error = target - output;


  // update weights and biases
  for (int i = 0; i < HIDDEN_SIZE; ++i)
  {
    double deltaHidden = error * output * (1.0 - output) * nn->weightsOutputHidden[0][i];
    nn->biasHidden[i] += LEARNING_RATE * deltaHidden;
    for (int j = 0; j < INPUT_SIZE; ++j)
    {
      nn->weightsInputHidden[i][j] += LEARNING_RATE * deltaHidden * input[j];
    }
    
  }

  for (int i = 0; i < OUTPUT_SIZE; ++i)
  {
    double deltaOutput = error * output * (1.0 - output) * nn->weightsOutputHidden[i][0];
    nn->biasOutput[i] += LEARNING_RATE * deltaOutput;
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      nn->weightsOutputHidden[i][j] += LEARNING_RATE * deltaOutput * hidden[j];
    }
  }
}

// training function
void train(NeuralNetwork *nn,double trainingData[][INPUT_SIZE], double targets[],int epochs)
{
  for (int epoch = 0;epoch < epochs; ++epoch)
  {
    for (int i=0;i<INPUT_SIZE;++i)
    {
      backwardPropagation(nn, trainingData[i], targets[i]);
    }
  }
}

int main()
{
  NeuralNetwork nn;

  // initialize weights and biases randomly
  for (int i = 0; i < HIDDEN_SIZE; ++i)
  {
    for (int j = 0; j < INPUT_SIZE; ++j)
    {
      nn.weightsInputHidden[i][j] = (double)rand() / RAND_MAX;
    }
    nn.biasHidden[i] = (double)rand() / RAND_MAX;
  }

  for (int i = 0; i < OUTPUT_SIZE; ++i)
  {
    for (int j = 0; j < HIDDEN_SIZE; ++j)
    {
      nn.weightsOutputHidden[i][j] = (double)rand() / RAND_MAX;
    }
    nn.biasOutput[i] = 0.1;
  }
  
  // training data and targets
  double trainingData[4][2] = {{0,0},{0,1},{1,0},{1,1}};
  double targets[4] = {0,1,1,0};

  // train the neural network
  train(&nn, trainingData, targets, EPOCHS);
  // test the trained neural network
  printf("Predictions after training: \n");
  for (int i = 0; i < 4; ++i)
  {
    double prediction = forwardPropagation(&nn,trainingData[i]);
    printf("Input: %d %d, Target: %d, Prediction: %lf\n",(int)trainingData[i][0],(int)trainingData[i][1],(int)targets[i],prediction);
  }  
  
}

#include"input.h"

int main()
{
    float*** dyna_input = TENSOR3D_Create(1, 224, 224);
    for (int i = 0; i < 1; i++)
    {
        for (int j = 0; j < 224; j++)
        {
            for (int k = 0; k < 224; k++)
            {
                dyna_input[i][j][k] = (float)rand() / RAND_MAX;
            }
        }
    }
    //Conv1, BN1
    float ****filter_7x7 = TENSOR4D_Create(64, 1, 7, 7);
    read_file("src/input/net.0.0.weight_64.1.7.7.txt", filter_7x7, 64, 1, 7, 7);
    float* gama_1 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.0.1.weight_64.txt", gama_1, 64);
    float*  beta_1 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.0.1.bias_64.txt", beta_1, 64);
    //Conv2, BN2
    float ****conv1_filter_64x64x3x3_1 = TENSOR4D_Create(64, 64, 3, 3);
    read_file("src/input/net.b2.0.conv1.weight_64.64.3.3.txt", conv1_filter_64x64x3x3_1, 64, 64, 3, 3);
    float* gama_2 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.0.bn1.weight_64.txt", gama_2, 64);
    float*  beta_2 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.0.bn1.bias_64.txt", beta_2, 64);
    //Conv3, BN3
    float**** filter_64x64x3x3_2 = TENSOR4D_Create(64, 64, 3, 3);
    read_file("src/input/net.b2.0.conv2.weight_64.64.3.3.txt", filter_64x64x3x3_2, 64, 64, 3, 3);
    float* gama_3 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.0.bn2.weight_64.txt", gama_3, 64);
    float*  beta_3 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.0.bn2.bias_64.txt", beta_3, 64);
    //Conv4, BN4
    float**** filter_64x64x3x3_3 = TENSOR4D_Create(64, 64, 3, 3);
    read_file("src/input/net.b2.1.conv1.weight_64.64.3.3.txt", filter_64x64x3x3_3, 64, 64, 3, 3);
    float* gama_4 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.1.bn1.weight_64.txt", gama_4, 64);
    float*  beta_4 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.1.bn1.bias_64.txt", beta_4, 64);
    //Conv5, BN5
    float**** filter_64x64x3x3_4 = TENSOR4D_Create(64, 64, 3, 3);
    read_file("src/input/net.b2.1.conv2.weight_64.64.3.3.txt", filter_64x64x3x3_4, 64, 64, 3, 3);
    float* gama_5 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.1.bn2.weight_64.txt", gama_5, 64);
    float*  beta_5 = (float*)malloc(64 * sizeof(float));
    batchNorm_readData("src/input/net.b2.1.bn2.bias_64.txt", beta_5, 64);



    

    //initial layer
    float ***conv1 = Conv2D(dyna_input, 224, 224, 1, 64, filter_7x7, 7, 7, 3, 2, 0);//-> 112 x 112 x 64
    batchnorm_4d(conv1, 112, 112, 64, gama_1, beta_1, EPSILON);
    conv1 = ReLU(conv1, 112, 112, 64);
    float*** p_maxPooling = MAX_Pooling(conv1, 112, 112, 64, 3, 2, 1);                              // -> 56 x 56 x 64

    //RESIDUAL BLOCK 1
    float*** conv2 = Conv2D(p_maxPooling, 56, 56, 64, 64, conv1_filter_64x64x3x3_1, 3, 3, 1, 1, 0);     //-> 56 x 56 x 64
    batchnorm_4d(conv2, 56, 56, 64, gama_2, beta_2, EPSILON);
    conv2 = ReLU(conv2, 56, 56, 64);
    float*** conv3 = Conv2D(conv2, 56, 56, 64, 64, filter_64x64x3x3_2, 3, 3, 1, 1, 0);               //-> 56 x 56 x64
    batchnorm_4d(conv3, 56, 56, 64, gama_3, beta_3, EPSILON);
    //add skip connection
    conv3 = ADD_SKIP_CONNECTION(p_maxPooling, conv3, 56, 56, 64);
    //final output
    conv3 = ReLU(conv3, 56, 56, 64);                                                //-> 56 x 56 x 64

    float*** conv4 = Conv2D(conv3, 56, 56, 64, 64, filter_64x64x3x3_3, 3, 3, 1, 1, 0);     //-> 56 x 56 x 64
    batchnorm_4d(conv4, 56, 56, 64, gama_4, beta_4, EPSILON);
    conv4 = ReLU(conv4, 56, 56, 64);
    float*** conv5 = Conv2D(conv4, 56, 56, 64, 64, filter_64x64x3x3_4, 3, 3, 1, 1, 0);               //-> 56 x 56 x64
    batchnorm_4d(conv5, 56, 56, 64, gama_5, beta_5, EPSILON);
    //add skip connection
    conv5 = ADD_SKIP_CONNECTION(conv3, conv5, 56, 56, 64);
    //final output
    conv5 = ReLU(conv5, 56, 56, 64);                                                //-> 56x56x64



    //Conv6, BN6
    float ****filter_128x64x3x3 = TENSOR4D_Create(128, 64, 3, 3);// -> 128 x 64 x 3 x 3
    read_file("src/input/net.b3.0.conv1.weight_128.64.3.3.txt", filter_128x64x3x3, 128, 64, 3, 3);
    float* gama_6 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.0.bn1.weight_128.txt", gama_6, 128);
    float*  beta_6 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.0.bn1.bias_128.txt", beta_6, 128);
    //Conv7, BN7
    float ****filter_128x128x3x3_1 = TENSOR4D_Create(128, 128, 3, 3);// -> 128 x 64 x 3 x 3
    read_file("src/input/net.b3.0.conv2.weight_128.128.3.3.txt", filter_128x128x3x3_1, 128, 128, 3, 3);
    float* gama_7 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.0.bn2.weight_128.txt", gama_7, 128);
    float*  beta_7 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.0.bn2.bias_128.txt", beta_7, 128);
    //Conv_skip_1
    float ****filter_128x64x1x1 = TENSOR4D_Create(128, 64, 1, 1);// -> 128 x 64 x 3 x 3
    read_file("src/input/net.b3.0.conv3.weight_128.64.1.1.txt", filter_128x64x1x1, 128, 64, 1, 1);
    //Conv8, BN8
    float ****filter_128x128x3x3_2 = TENSOR4D_Create(128, 128, 3, 3);// -> 128 x 64 x 3 x 3
    read_file("src/input/net.b3.1.conv1.weight_128.128.3.3.txt", filter_128x128x3x3_2, 128, 128, 3, 3);
    float* gama_8 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.1.bn1.weight_128.txt", gama_8, 128);
    float*  beta_8 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.1.bn1.bias_128.txt", beta_8, 128);
    //Conv9, BN9
    float ****filter_128x128x3x3_3 = TENSOR4D_Create(128, 128, 3, 3);// -> 128 x 64 x 3 x 3
    read_file("src/input/net.b3.1.conv2.weight_128.128.3.3.txt", filter_128x128x3x3_3, 128, 128, 3, 3);
    float* gama_9 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.1.bn2.weight_128.txt", gama_9, 128);
    float*  beta_9 = (float*)malloc(128 * sizeof(float));
    batchNorm_readData("src/input/net.b3.1.bn2.bias_128.txt", beta_9, 128);



    


    //RESIDUAL BLOCK 2
    float*** conv6 = Conv2D(conv5, 56, 56, 64, 128, filter_128x64x3x3, 3, 3, 1, 2, 0);    //-> 28 x 28 x 128
    batchnorm_4d(conv6, 28, 28, 128, gama_6, beta_6, EPSILON);
    conv6 = ReLU(conv6, 28, 28, 128);
    float*** conv7 = Conv2D(conv6, 28, 28, 128, 128, filter_128x128x3x3_1, 3, 3, 1, 1, 0);             //-> 28 x 28 x 128
    batchnorm_4d(conv7, 28, 28, 128, gama_7, beta_7, EPSILON);
    // add skip connection
    float*** conv5_1 = Conv2D(conv5, 56, 56, 64, 128, filter_128x64x1x1, 1, 1, 0, 2, 0);              //-> 28 x 28 x 128
    conv7 = ADD_SKIP_CONNECTION(conv5_1, conv7, 28, 28, 128);
    //final output
    conv7 = ReLU(conv7, 28, 28, 128);                                                //-> 28 x 28 x 128

    float*** conv8 = Conv2D(conv7, 28, 28, 128, 128, filter_128x128x3x3_2, 3, 3, 1, 1, 0);   //-> 28 x 28 x 128
    batchnorm_4d(conv8, 28, 28, 128, gama_8, beta_8, EPSILON);
    conv8 = ReLU(conv8, 28, 28, 128);
    float*** conv9 = Conv2D(conv8, 28, 28, 128, 128, filter_128x128x3x3_3, 3, 3, 1, 1, 0);
    batchnorm_4d(conv9, 28, 28, 128, gama_9, beta_9, EPSILON);
    //add skip connection
    conv9 = ADD_SKIP_CONNECTION(conv7, conv9, 28, 28, 128);
    //final output
    conv9 = ReLU(conv9, 28, 28, 128);


    //Con10, BN10
    float ****filter_256x128x3x3 = TENSOR4D_Create(256, 128, 3, 3);// -> 256 x 128 x 3 x 3
    read_file("src/input/net.b4.0.conv1.weight_256.128.3.3.txt", filter_256x128x3x3, 256, 128, 3, 3);
    float* gama_10 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.0.bn1.weight_256.txt", gama_10, 256);
    float*  beta_10 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.0.bn1.bias_256.txt", beta_10, 256);
    //Conv11, BN11
    float ****filter_256x256x3x3_1 = TENSOR4D_Create(256, 256, 3, 3);// -> 256 x 256 x 3 x 3
    read_file("src/input/net.b4.0.conv2.weight_256.256.3.3.txt", filter_256x256x3x3_1, 256, 256, 3, 3);
    float* gama_11 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.0.bn2.weight_256.txt", gama_11, 256);
    float*  beta_11 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.0.bn2.bias_256.txt", beta_11, 256);
    //Conv_skip_2
    float ****filter_256x128x1x1 = TENSOR4D_Create(256, 128, 1, 1);// -> 256 x 256 x 3 x 3
    read_file("src/input/net.b4.0.conv3.weight_256.128.1.1.txt", filter_256x128x1x1, 256, 128, 1, 1);
    //Conv12, BN12
    float ****filter_256x256x3x3_2 = TENSOR4D_Create(256, 256, 3, 3);// -> 256 x 256 x 3 x 3
    read_file("src/input/net.b4.1.conv1.weight_256.256.3.3.txt", filter_256x256x3x3_2, 256, 256, 3, 3);
    float* gama_12 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.1.bn1.weight_256.txt", gama_12, 256);
    float*  beta_12 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.1.bn1.bias_256.txt", beta_12, 256);
    //Conv13, BN13
    float ****filter_256x256x3x3_3 = TENSOR4D_Create(256, 256, 3, 3);// -> 256 x 256 x 3 x 3
    read_file("src/input/net.b4.1.conv2.weight_256.256.3.3.txt", filter_256x256x3x3_3, 256, 256, 3, 3);
    float* gama_13 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.1.bn2.weight_256.txt", gama_13, 256);
    float*  beta_13 = (float*)malloc(256 * sizeof(float));
    batchNorm_readData("src/input/net.b4.1.bn2.bias_256.txt", beta_13, 256);



    
    //RESIDUAL BLOCK 3
    float*** conv10 = Conv2D(conv9, 28, 28, 128, 256, filter_256x128x3x3, 3, 3, 1, 2, 0);    //-> 14 x 14 x 256
    batchnorm_4d(conv10, 14, 14, 256, gama_10, beta_10, EPSILON);
    conv10 = ReLU(conv10, 14, 14, 256);
    float*** conv11 = Conv2D(conv10, 14, 14, 256, 256, filter_256x256x3x3_1, 3, 3, 1, 1, 0);             //-> 14 x 14 x 256
    batchnorm_4d(conv11, 14, 14, 256, gama_11, beta_11, EPSILON);
    // add skip connection
    float*** conv9_1 = Conv2D(conv9, 28, 28, 128, 256, filter_256x128x1x1, 1, 1, 0, 2, 0);             //-> 14 x 14 x 256
    conv11 = ADD_SKIP_CONNECTION(conv9_1, conv11, 14, 14, 256);
    //final output
    conv11 = ReLU(conv11, 14, 14, 256);                                                //-> 14 x 14 x 256

    float*** conv12 = Conv2D(conv11, 14, 14, 256, 256, filter_256x256x3x3_2, 3, 3, 1, 1, 0);
    batchnorm_4d(conv12, 14, 14, 256, gama_12, beta_12, EPSILON);
    conv12 = ReLU(conv12, 14, 14, 256);
    float*** conv13 = Conv2D(conv12, 14, 14, 256, 256, filter_256x256x3x3_3, 3, 3, 1, 1, 0);
    batchnorm_4d(conv13, 14, 14, 256, gama_13, beta_13, EPSILON);
    //add skip connection
    conv13 = ADD_SKIP_CONNECTION(conv11, conv13, 14, 14, 256);
    //final output
    conv13 = ReLU(conv13, 14, 14, 256);                                               //-> 14 x 14 x 256


    
    //Conv14, BN14
    float ****filter_512x256x3x3 = TENSOR4D_Create(512, 256, 3, 3);// -> 512 x 256 x 3 x 3
    read_file("src/input/net.b5.0.conv1.weight_512.256.3.3.txt", filter_512x256x3x3, 512, 256, 3, 3);
    float* gama_14 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.0.bn1.weight_512.txt", gama_14, 512);
    float*  beta_14 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.0.bn1.bias_512.txt", beta_14, 512);
    //Conv15, BN15
    float ****filter_512x512x3x3_1 = TENSOR4D_Create(512, 512, 3, 3);// -> 512 x 512 x 3 x 3
    read_file("src/input/net.b5.0.conv2.weight_512.512.3.3.txt", filter_512x512x3x3_1, 512, 512, 3, 3);
    float* gama_15 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.0.bn2.weight_512.txt", gama_15, 512);
    float*  beta_15 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.0.bn2.bias_512.txt", beta_15, 512);
    //Conv_skip_3
    float ****filter_512x256x1x1 = TENSOR4D_Create(512, 256, 1, 1);// -> 512 x 256 x 1 x 1
    read_file("src/input/net.b5.0.conv3.weight_512.256.1.1.txt", filter_512x256x1x1, 512, 256, 1, 1);
    //Conv16, BN16
    float ****filter_512x512x3x3_2 = TENSOR4D_Create(512, 512, 3, 3);// -> 512 x 512 x 3 x 3
    read_file("src/input/net.b5.1.conv1.weight_512.512.3.3.txt", filter_512x512x3x3_2, 512, 512, 3, 3);
    float* gama_16 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.1.bn1.weight_512.txt", gama_16, 512);
    float*  beta_16 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.1.bn1.bias_512.txt", beta_16, 512);
    //Conv17
    float ****filter_512x512x3x3_3 = TENSOR4D_Create(512, 512, 3, 3);// -> 512 x 512 x 3 x 3
    read_file("src/input/net.b5.1.conv2.weight_512.512.3.3.txt", filter_512x512x3x3_3, 512, 512, 3, 3);
    float* gama_17 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.1.bn2.weight_512.txt", gama_17, 512);
    float*  beta_17 = (float*)malloc(512 * sizeof(float));
    batchNorm_readData("src/input/net.b5.1.bn2.bias_512.txt", beta_17, 512);





    //RESIDUAL BLOCK 4
    float*** conv14 = Conv2D(conv13, 14, 14, 256, 512, filter_512x256x3x3, 3, 3, 1, 2, 0);    //-> 7 x 7 x 512
    batchnorm_4d(conv14, 7, 7, 512, gama_14, beta_14, EPSILON);
    conv14 = ReLU(conv14, 7, 7, 512);
    float*** conv15 = Conv2D(conv14, 7, 7, 512, 512, filter_512x512x3x3_1, 3, 3, 1, 1, 0);             //-> 7 x 7 x 512
    batchnorm_4d(conv15, 7, 7, 512, gama_15, beta_15, EPSILON);
    // add skip connection
    float*** conv13_1 = Conv2D(conv13, 14, 14, 256, 512, filter_512x256x1x1, 1, 1, 0, 2, 0);            //-> 7 x 7 x 512
    conv15 = ADD_SKIP_CONNECTION(conv13_1, conv15, 7, 7, 512);
    //final output
    conv15 = ReLU(conv15, 7, 7, 512);                                                //-> 7 x 7 x 512

    float*** conv16 = Conv2D(conv15, 7, 7, 512, 512, filter_512x512x3x3_2, 3, 3, 1, 1, 0);
    batchnorm_4d(conv16, 7, 7, 512, gama_16, beta_16, EPSILON);
    conv16 = ReLU(conv16, 7, 7, 512);
    float*** conv17 = Conv2D(conv16, 7, 7, 512, 512, filter_512x512x3x3_3, 3, 3, 1, 1, 0);
    batchnorm_4d(conv17, 7, 7, 512, gama_17, beta_17, EPSILON);
    //add skip connection
    conv17 = ADD_SKIP_CONNECTION(conv15, conv17, 7, 7, 512);
    //final output
    conv17 = ReLU(conv17, 7, 7, 512);                                               //-> 7 x 7 x 512

    // AVG POOLING LAYER
    float*** p_avgPooling = AVG_Pooling(conv17, 7, 7, 512, 7, 1, 0);                               //-> 1 x 1 x 512
    //FC LAYER
    float** weight_T = MATRIX_Create(512, 10);
    FC_readData("src/input/net.last.2.weight_10.512.txt", weight_T, 512, 10);
    float* output = (float*)malloc(sizeof(float) * 10);
    if(!output){
        return 1;
    }
    batchNorm_readData("src/input/net.last.2.bias_10.txt", output, 10);

    float* result = Flatten(p_avgPooling, 1, 1, 512);
    
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 512; j++){
            output[i] += result[j] * weight_T[i][j];
        }
    }
    output = Softmax(output, 10);
    for(int i = 0; i < 10; i++){
        printf("%f ", output[i]);
    }
    
    
    TENSOR3D_Free(dyna_input, 1, 224, 224);
    TENSOR3D_Free(conv1, 64, 112, 112);
    TENSOR3D_Free(p_maxPooling, 64, 56, 56);
    TENSOR3D_Free(conv2, 64, 56, 56);
    TENSOR3D_Free(conv3, 64, 56, 56);
    TENSOR3D_Free(conv4, 64, 56, 56);
    TENSOR3D_Free(conv5, 64, 56, 56);
    TENSOR3D_Free(conv6, 128, 28, 28);
    TENSOR3D_Free(conv7, 128, 28, 28);
    TENSOR3D_Free(conv5_1, 128, 28, 28);
    TENSOR3D_Free(conv8, 128, 28, 28);
    TENSOR3D_Free(conv9, 128, 28, 28);
    TENSOR3D_Free(conv10, 256, 14, 14);
    TENSOR3D_Free(conv11, 256, 14, 14);
    TENSOR3D_Free(conv9_1, 256, 14, 14);
    TENSOR3D_Free(conv12, 256, 14, 14);
    TENSOR3D_Free(conv13, 256, 14, 14);
    TENSOR3D_Free(conv14, 512, 7, 7);
    TENSOR3D_Free(conv15, 512, 7, 7);
    TENSOR3D_Free(conv13_1, 512, 7, 7);
    TENSOR3D_Free(conv16, 512, 7, 7);
    TENSOR3D_Free(conv17, 512, 7, 7);
    TENSOR3D_Free(p_avgPooling, 512, 1, 1);

    TENSOR4D_Free(filter_7x7, 64, 1, 7, 7);
    BatchNorm_free(gama_1, beta_1);
    TENSOR4D_Free(conv1_filter_64x64x3x3_1, 64, 64, 3, 3);
    BatchNorm_free(gama_2, beta_2);
    TENSOR4D_Free(filter_64x64x3x3_2, 64, 64, 3, 3);
    BatchNorm_free(gama_3, beta_3);
    TENSOR4D_Free(filter_64x64x3x3_3, 64, 64, 3, 3);
    BatchNorm_free(gama_4, beta_4);
    TENSOR4D_Free(filter_64x64x3x3_4, 64, 64, 3, 3);
    BatchNorm_free(gama_5, beta_5);

    TENSOR4D_Free(filter_128x64x3x3, 128, 64, 3, 3);
    BatchNorm_free(gama_6, beta_6);
    TENSOR4D_Free(filter_128x128x3x3_1, 128, 128, 3, 3);
    BatchNorm_free(gama_7, beta_7);
    TENSOR4D_Free(filter_128x64x1x1, 128, 64, 1, 1);
    TENSOR4D_Free(filter_128x128x3x3_2, 128, 128, 3, 3);
    BatchNorm_free(gama_8, beta_8);
    TENSOR4D_Free(filter_128x128x3x3_3, 128, 128, 3, 3);
    BatchNorm_free(gama_9, beta_9);

    TENSOR4D_Free(filter_256x128x3x3, 256, 128, 3, 3);
    BatchNorm_free(gama_10, beta_10);
    TENSOR4D_Free(filter_256x256x3x3_1, 256, 256, 3, 3);
    BatchNorm_free(gama_11, beta_11);
    TENSOR4D_Free(filter_256x128x1x1, 256, 128, 1, 1);
    TENSOR4D_Free(filter_256x256x3x3_2, 256, 256, 3, 3);
    BatchNorm_free(gama_12, beta_12);
    TENSOR4D_Free(filter_256x256x3x3_3, 256, 256, 3, 3);
    BatchNorm_free(gama_13, beta_13);

    TENSOR4D_Free(filter_512x256x3x3, 512, 256, 3, 3);
    BatchNorm_free(gama_14, beta_14);
    TENSOR4D_Free(filter_512x512x3x3_1, 512, 512, 3, 3);
    BatchNorm_free(gama_15, beta_15);
    TENSOR4D_Free(filter_512x256x1x1, 512, 256, 1, 1);
    TENSOR4D_Free(filter_512x512x3x3_2, 512, 512, 3, 3);
    BatchNorm_free(gama_16, beta_16);
    TENSOR4D_Free(filter_512x512x3x3_3, 512, 512, 3, 3);
    BatchNorm_free(gama_17, beta_17);

    MATRIX_Free(weight_T, 512, 10);

    free(result);
    free(output);
    return 0;
}



    
#include "input.h"



#define MNIST_SIZE 28
#define RESNET_SIZE 224
#define BATCH_SIZE 1

void read_file(const char *file_path, float ****buffer, int out_filter, int channel, int width, int height) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error: Cannot open file %s\n", file_path);
        exit(1);
    }

    for (int i = 0; i < out_filter; i++) {
        for(int j = 0; j < channel; j ++){
            for(int k = 0; k < width; k++){
                for(int z = 0; z < height; z++){
                    if (fscanf(file, "%f", &buffer[i][j][k][z]) != 1) {
                        printf("Error: Unexpected data format in file %s\n", file_path);
                        exit(1);
                    }                    
                }
            }
        }
    }
    fclose(file);
}

void FC_readData(const char *file_path, float **buffer, int width, int height)
{
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Error: Cannot open file %s\n", file_path);
        exit(1);
    }
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            if (fscanf(file, "%f", &buffer[i][j]) != 1) {
                printf("Error: Unexpected data format in file %s\n", file_path);
                exit(1);
            }                    
            
        }
    }

}

void batchNorm_readData(const char *file_path, float *buffer, int size)
{
    FILE *file = fopen(file_path, "r");
    if(!file){
        printf("Error: Cannot open file %s \n", file_path);
    }
    for(int i = 0; i < size; i++){
        if (fscanf(file, "%f", &buffer[i]) != 1) {
            printf("Error: Unexpected data format in file %s\n", file_path);
            exit(1);
        }                        
    }
}

void resize_image(unsigned char* input, float*** output) {
    float scale = (float)MNIST_SIZE / RESNET_SIZE;
    
    // Xử lý cho batch_size = 1
    for(int b = 0; b < BATCH_SIZE; b++) {
        for(int i = 0; i < RESNET_SIZE; i++) {
            for(int j = 0; j < RESNET_SIZE; j++) {
                // Tính vị trí tương ứng trong ảnh gốc
                int src_i = (int)(i * scale);
                int src_j = (int)(j * scale);
                
                // Đảm bảo không vượt quá kích thước ảnh gốc
                src_i = src_i < MNIST_SIZE ? src_i : MNIST_SIZE-1;
                src_j = src_j < MNIST_SIZE ? src_j : MNIST_SIZE-1;
                
                // Copy pixel value và normalize về range [0,1]
                output[b][i][j] = input[src_i * MNIST_SIZE + src_j] / 255.0f;
            }
        }
    }
}

// Hàm chính để xử lý ảnh MNIST
float*** preprocess_mnist(const char* filepath) {
    // Đọc ảnh từ file
    unsigned char* mnist_image = load_mnist_image(filepath);
    if (!mnist_image) {
        return NULL;
    }

    // Cấp phát mảng 3 chiều
    float*** resized_image = TENSOR3D_Create(BATCH_SIZE, RESNET_SIZE, RESNET_SIZE);
    if (!resized_image) {
        free(mnist_image);
        printf("Memory allocation failed!\n");
        return NULL;
    }
    
    // Resize và normalize ảnh
    resize_image(mnist_image, resized_image);
    
    // Giải phóng bộ nhớ tạm
    free(mnist_image);
    
    return resized_image;
}



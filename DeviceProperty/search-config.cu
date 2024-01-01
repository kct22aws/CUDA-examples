#include <stdio.h>

int main(void) {
    cudaDeviceProp prop;
    int device_id;

    cudaGetDevice( &device_id);

    printf("Current Device ID: %d\n", device_id);

    memset( &prop, 0, sizeof( cudaDeviceProp)); // copy 0 to cudaDeviceProp object or initialization.
    prop.major = 9;
    prop.minor = 1;

    cudaChooseDevice( &device_id, &prop); // device_id and prop will be reassigned by this function with pass by reference. 
    printf("CUDA DEVICE ID that best matches the requirement: %d\n", device_id);

    cudaSetDevice (device_id); // set device_id as where host threads will be executed.

}
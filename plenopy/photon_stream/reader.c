#include <math.h>


void c_stream2sequence(
    unsigned char* photon_stream,
    unsigned int photon_stream_length,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    unsigned short* sequence,
    unsigned int number_time_slices,
    unsigned int number_lixel,
    float *time_delay_mean,
    float time_slice_duration
) {
    unsigned int lixel = 0;
    for(unsigned int s=0; s<photon_stream_length; s++) {
        if(photon_stream[s] == NEXT_READOUT_CHANNEL_MARKER) {
            lixel++;
        }else{
            unsigned char raw_arrival_slice = photon_stream[s];
            float arrival_time = (float)raw_arrival_slice*time_slice_duration;
            arrival_time -= time_delay_mean[lixel];

            int arrival_slice = (int)round(arrival_time/time_slice_duration);

            if(arrival_slice < (int)number_time_slices && arrival_slice >= 0) {
                unsigned int index = arrival_slice*number_lixel + lixel;
                sequence[index] += 1;
            }
        }
    }

    return;
}

void c_stream2_cx_cy_arrivaltime_point_cloud(
    unsigned char* photon_stream,
    unsigned int photon_stream_length,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    float* point_cloud,
    float* cx,
    float* cy,
    float* time_delay_mean,
    float time_slice_duration,
    unsigned int* lixel_ids
) {
    unsigned int lixel = 0;
    unsigned int photon = 0;
    for (unsigned int s = 0; s < photon_stream_length; s++) {
        if (photon_stream[s] == NEXT_READOUT_CHANNEL_MARKER) {
            lixel++;
        } else {
            unsigned char raw_arrival_slice = photon_stream[s];
            float arrival_time = (float)raw_arrival_slice*time_slice_duration;
            arrival_time -= time_delay_mean[lixel];

            point_cloud[photon*3 + 0] = cx[lixel];
            point_cloud[photon*3 + 1] = cy[lixel];
            point_cloud[photon*3 + 2] = arrival_time;
            lixel_ids[photon] = lixel;
            photon++;
        }
    }
    return;
}

void c_photon_stream_to_arrival_slices_and_lixel_ids(
    unsigned char* photon_stream,
    unsigned int photon_stream_length,
    unsigned char NEXT_READOUT_CHANNEL_MARKER,
    unsigned char* arrival_slices,
    unsigned int* lixel_ids
) {
    unsigned int lixel = 0;
    unsigned int photon = 0;
    for (unsigned int s = 0; s < photon_stream_length; s++) {
        if (photon_stream[s] == NEXT_READOUT_CHANNEL_MARKER) {
            lixel++;
        } else {
            arrival_slices[photon] = photon_stream[s];
            lixel_ids[photon] = lixel;
            photon++;
        }
    }
    return;
}
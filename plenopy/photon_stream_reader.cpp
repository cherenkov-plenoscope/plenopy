#include <math.h>

extern "C"{

    void c_stream2sequence(
        unsigned char* photon_stream,
        unsigned int photon_stream_length,
        unsigned char NEXT_READOUT_CHANNEL_MARKER,
        unsigned short* sequence,
        unsigned int number_time_slices,
        unsigned int number_lixel,
        float *time_delay_mean
    ) {

        float time_slice_duration = 0.5e-9;

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
}
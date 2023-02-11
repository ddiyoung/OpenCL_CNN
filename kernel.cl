__kernel void convolution(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    int inDim,
    int outDim,
    int nbyn,
    int b_index,
    int f_index
){  
    int l_i = get_local_id(0);
    float sum = 0;
    int x = 0 , y = 0;
    int row = 0 , col = 0;
    int index = 0;

    int input_index =inDim * nbyn * nbyn * (get_group_id(0) / outDim);

    float l_sum = 0;

    for(int i = 0 ; i < inDim; i ++){
        sum = 0;
        for(int fRow = 0 ; fRow < 3; ++fRow){
            for(int fCol = 0; fCol < 3; ++fCol){
                x = fCol - 1;
                y = nbyn * (fRow - 1);
                index = l_i + x + y;
                row = (l_i / nbyn) + (fRow - 1);
                col = (l_i % nbyn) + x;
                if( row >= 0 && row < nbyn && col >= 0 && col < nbyn && index >= 0 && index < nbyn * nbyn){
                    sum += inputs[input_index + nbyn * nbyn * i + index] * filter[f_index + 9 * inDim * (get_group_id(0) % outDim) + 9 * i + 3*fRow + fCol];
                }
            }
        }
        l_sum += sum;
    }
    l_sum += filter[b_index + get_group_id(0) % outDim];
    if(l_sum < 0) outputs[get_global_id(0)] = 0;
    else outputs[get_global_id(0)] = l_sum;
}

__kernel void convolution2(
    __global float* inputs,
    __global float* outputs,
    __global float* filter,
    int inDim,
    int outDim,
    int nbyn,
    int b_index,
    int f_index,
    __local float* in_Dim_sum
){
    int l_i = get_local_id(0);
    int out_Dim = (get_group_id(0) / (nbyn*nbyn)) % outDim;
    int image_num = get_group_id(0) / (nbyn * nbyn * outDim);
    int input_index = nbyn * nbyn * inDim * image_num;
    int index = (get_group_id(0) % outDim);
    int index2;
    int row;
    int col;
    float sum = 0;
    int x = 0 , y = 0;

    for(int fRow = 0; fRow < 3; ++fRow){
        for(int fCol = 0; fCol < 3; ++fCol){
            x = fCol - 1;
            y = nbyn * (fRow - 1);
            row = (index / nbyn) + (fRow - 1);
            col = (index / nbyn) + x;
            index2 = index + x + y;
            if(row >= 0 && row < nbyn && col >= 0 && col < nbyn && index >= 0 && index < nbyn * nbyn){
                sum += inputs[input_index + nbyn * nbyn * l_i + index2] * filter[f_index + 9 * inDim * out_Dim + 9 * l_i + 3 * fRow + fCol];
            }
            if( fRow == 1 && fCol == 1 && l_i == 1 && get_group_id(0) == 0) printf("%d %d\n", input_index + nbyn * nbyn * l_i + index2, 9 * inDim * out_Dim + 9 * l_i + 3 * fRow + fCol);
        }
    }

    in_Dim_sum[l_i] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int p = inDim / 2 ; p >= 1; p = p >> 1){
        if(l_i < p) in_Dim_sum[l_i] += in_Dim_sum[l_i + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_i == 0){
        in_Dim_sum[0] += filter[b_index + out_Dim];
        if(in_Dim_sum[0] > 0) outputs[get_group_id(0)] = in_Dim_sum[0];
        else outputs[get_group_id(0)] = 0;
    }
}

__kernel void max_pooling(
    __global float* input,
    __global float* output,
    int inDim,
    int nbyn
){
    int l_i = get_local_id(0);
    float max = 0, temp;
    int row = l_i / nbyn;
    int col = l_i % nbyn;
    int x, y;
    if(row % 2 == 0 && col % 2 == 0){
        for(y = 0; y < 2; ++y){
            for(x = 0; x < 2; ++x){
                temp = input[nbyn * nbyn * get_group_id(0) + nbyn * (row + y) + col + x];
                if(max < temp) max = temp;
            }
        }
        output[(nbyn / 2) * (nbyn / 2) * get_group_id(0) + (nbyn / 2) * (row / 2) + (col / 2)] = max;
    }
}

__kernel void fc_layer(
    __global float* input,
    __global float* output,
    __global float* filter,
    __local float* l_sum,
    int inDim,
    int outDim,
    int f_index,
    int b_index
){
    int l_i = get_local_id(0);
    int i = get_global_id(0);
    l_sum[l_i] = (i < inDim * outDim * 750)? input[l_i + (get_group_id(0) / outDim) * inDim] * filter[f_index + inDim * (get_group_id(0) % outDim) + l_i] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int p = get_local_size(0) / 2 ; p >= 1; p = p >> 1){
        if(l_i < p) l_sum[l_i] += l_sum[l_i + p];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_i == 0){
        l_sum[0] += filter[b_index + get_group_id(0) % outDim];
        if(l_sum[0] > 0) output[get_group_id(0)] = l_sum[0];
        else output[get_group_id(0)] = 0;
    }
}
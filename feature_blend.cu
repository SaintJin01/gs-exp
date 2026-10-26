// Uses the buffer ABI of this repository's diff-gaussian-rasterization.
// Geometry remains frozen: d(pixel_features)/d(gaussian_features) = alpha*T.
// No geometry/opacity/covariance backward is necessary for SAM feature training.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template<typename T> T* take(char*& p, size_t count) {
    p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 127) & ~uintptr_t(127));
    T* value = reinterpret_cast<T*>(p);
    p += sizeof(T)*count;
    return value;
}
struct Buffers {
    float2* xy; float4* conic; uint32_t* ids; uint32_t* last; uint2* ranges;
};
Buffers parse(torch::Tensor geom, torch::Tensor bins, torch::Tensor image, int n, int pixels) {
    char* p = reinterpret_cast<char*>(geom.data_ptr());
    take<float>(p, n); take<bool>(p, n*3); take<int>(p, n);
    float2* xy = take<float2>(p, n);
    take<float>(p, n*6);
    float4* conic = take<float4>(p, n);
    p = reinterpret_cast<char*>(image.data_ptr());
    take<float>(p, pixels);
    uint32_t* last = take<uint32_t>(p, pixels);
    uint2* ranges = take<uint2>(p, pixels);
    // First binning allocation is point_list (the tensor base is CUDA-aligned).
    return {xy, conic, reinterpret_cast<uint32_t*>(bins.data_ptr()), last, ranges};
}

template<int CHANNELS, bool BACKWARD>
__global__ void blend(Buffers b, int width, int height, const float* features,
                      const float* grad_image, float* output, float background) {
    const int lane = threadIdx.y*16 + threadIdx.x;
    const int x = blockIdx.x*16 + threadIdx.x;
    const int y = blockIdx.y*16 + threadIdx.y;
    const int pixel = y*width+x;
    const bool inside = x<width && y<height;
    const uint32_t last = inside ? b.last[pixel] : 0;
    const uint2 range = b.ranges[blockIdx.y*gridDim.x+blockIdx.x];
    __shared__ float2 xy[256];
    __shared__ float4 conic[256];
    __shared__ uint32_t ids[256];
    float value[CHANNELS] = {0};
    float gradients[CHANNELS];
    if (BACKWARD && inside) {
        #pragma unroll
        for (int c=0;c<CHANNELS;c++) gradients[c] = grad_image[c*width*height+pixel];
    }
    float transmittance = 1;
    bool done = !inside || last==0;
    for (uint32_t start=range.x; start<range.y; start+=256) {
        if (__syncthreads_count(done)==256) break;
        if (start+lane<range.y) {
            const int id = b.ids[start+lane];
            ids[lane]=id; xy[lane]=b.xy[id]; conic[lane]=b.conic[id];
        }
        __syncthreads();
        const int count = min(256u, range.y-start);
        for (int j=0;j<count && !done;j++) {
            if (start-range.x+j>=last) { done=true; break; }
            const float dx=xy[j].x-x, dy=xy[j].y-y;
            const float4 co=conic[j];
            const float power=-.5f*(co.x*dx*dx+co.z*dy*dy)-co.y*dx*dy;
            if (power>0) continue;
            const float alpha=fminf(.99f,co.w*expf(power));
            if (alpha<1.f/255.f) continue;
            const float weight=alpha*transmittance;
            const int id=ids[j];
            #pragma unroll
            for (int c=0;c<CHANNELS;c++) {
                if (BACKWARD) atomicAdd(output+id*CHANNELS+c, weight*gradients[c]);
                else value[c]+=features[id*CHANNELS+c]*weight;
            }
            transmittance *= 1-alpha;
        }
        __syncthreads();
    }
    if (!BACKWARD && inside) {
        #pragma unroll
        for (int c=0;c<CHANNELS;c++) output[c*width*height+pixel]=value[c]+background*transmittance;
    }
}

torch::Tensor blend_forward(torch::Tensor features, torch::Tensor geom,
                            torch::Tensor bins, torch::Tensor image,
                            int64_t rendered, int64_t height, int64_t width, double background) {
    TORCH_CHECK(features.is_cuda() && features.is_contiguous() && features.scalar_type()==torch::kFloat32,
                "Expected contiguous CUDA float32 features");
    TORCH_CHECK(features.dim()==2 && (features.size(1)==16 || features.size(1)==32), "Feature backend requires 16 or 32 channels");
    c10::cuda::CUDAGuard guard(features.device());
    auto output=torch::empty({features.size(1),height,width}, features.options());
    if (!features.size(0) || !rendered) return output.fill_(background);
    Buffers b=parse(geom,bins,image,features.size(0),height*width);
    if (features.size(1)==16) {
    blend<16,false><<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,
                    at::cuda::getCurrentCUDAStream()>>>(b,width,height,features.data_ptr<float>(),nullptr,
                                                       output.data_ptr<float>(),background);
    } else {
    blend<32,false><<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,
                    at::cuda::getCurrentCUDAStream()>>>(b,width,height,features.data_ptr<float>(),nullptr,
                                                       output.data_ptr<float>(),background);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
torch::Tensor blend_backward(torch::Tensor grad, torch::Tensor geom,
                             torch::Tensor bins, torch::Tensor image,
                             int64_t count, int64_t rendered) {
    c10::cuda::CUDAGuard guard(grad.device());
    auto output=torch::zeros({count,grad.size(0)}, grad.options());
    if (!count || !rendered) return output;
    const int height=grad.size(1),width=grad.size(2);
    Buffers b=parse(geom,bins,image,count,height*width);
    if (grad.size(0)==16) {
    blend<16,true><<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,
                   at::cuda::getCurrentCUDAStream()>>>(b,width,height,nullptr,grad.data_ptr<float>(),
                                                      output.data_ptr<float>(),0);
    } else {
    blend<32,true><<<dim3((width+15)/16,(height+15)/16),dim3(16,16),0,
                   at::cuda::getCurrentCUDAStream()>>>(b,width,height,nullptr,grad.data_ptr<float>(),
                                                      output.data_ptr<float>(),0);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

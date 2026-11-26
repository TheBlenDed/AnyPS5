#include "VulkanFixture.h"
#include "BufferRangeModel.h"
#include "RangeImplementation.inc"
#include <cstring>
#include <random>
#include <string_view>

void Check(const BufferCache& cache, BufferCache::OverlapResult r, uint64_t address, uint64_t size) {
    Require(r.begin <= address && r.end >= address + size && r.end > r.begin, "request covered");
    for (auto it = cache.m_buffers.begin(); it != cache.m_buffers.end(); ++it) {
        const auto& b = cache.m_slot_buffers[it->second];
        const bool intersects = r.begin < b.begin+b.size && b.begin < r.end;
        if (intersects) Require(b.begin >= r.begin && b.begin+b.size <= r.end, "partial old buffer");
        auto inside = r.first != r.last && it->first >= r.first->first &&
                      (r.last == cache.m_buffers.end() || it->first < r.last->first);
        Require(inside == intersects, "copy set");
    }
}

void GpuCopy(Vulkan& gpu, BufferCache& cache, BufferCache::OverlapResult r) {
    GpuBuffer target(gpu, r.end-r.begin);
    std::vector<unsigned char> expected(r.end-r.begin, 7);
    std::memset(target.mapped,7,expected.size());
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex=gpu.family;
    VkCommandPool pool{};
    Require(vkCreateCommandPool(gpu.device,&pi,nullptr,&pool)==VK_SUCCESS,"pool");
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool=pool; ai.commandBufferCount=1;
    VkCommandBuffer cmd{};
    Require(vkAllocateCommandBuffers(gpu.device,&ai,&cmd)==VK_SUCCESS,"command");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    Require(vkBeginCommandBuffer(cmd,&bi)==VK_SUCCESS,"begin");
    std::vector<GpuBuffer> old;
    for (auto it=r.first; it!=r.last; ++it) {
        const auto& b=cache.m_slot_buffers[it->second];
        auto& src=old.emplace_back(gpu,b.size);
        auto bytes=static_cast<unsigned char*>(src.mapped);
        for (size_t i=0;i<b.size;++i) bytes[i]=static_cast<unsigned char>((it->second+1)*43+i%181);
        auto offset=b.begin-r.begin;
        std::memcpy(expected.data()+offset,bytes,b.size);
        VkBufferCopy op{0,offset,b.size};
        vkCmdCopyBuffer(cmd,src.handle,target.handle,1,&op);
    }
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    Require(vkEndCommandBuffer(cmd)==VK_SUCCESS,"end");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
    Require(vkQueueSubmit(gpu.queue,1,&si,VK_NULL_HANDLE)==VK_SUCCESS,"submit");
    Require(vkQueueWaitIdle(gpu.queue)==VK_SUCCESS,"completed");
    Require(std::memcmp(expected.data(),target.mapped,expected.size())==0,"all bytes and gaps");
    for(auto& b:old) b.Release(gpu.device);
    target.Release(gpu.device);
    vkDestroyCommandPool(gpu.device,pool,nullptr);
}

int main(int argc,char**argv) {
    Require(argc==2,"mode");
    Vulkan gpu;
    const std::string_view mode=argv[1];
    constexpr uint64_t p=16384, room=128*p, end=0x5059520000ull;
    if(mode=="shapes") {
        std::mt19937 gen(20261126);
        for(unsigned n=0;n<4000;++n) {
            BufferCache c;
            uint64_t addr=4*room;
            for(int i=0;i<20;++i) {
                auto size=p*(1+gen()%260);
                c.m_buffers.emplace(addr,c.m_slot_buffers.size());
                c.m_slot_buffers.push_back({addr,size,int(gen()%24)});
                addr+=size+p*(gen()%3);
            }
            uint64_t req=4*room+(gen()%((addr-4*room)/p))*p, bytes=p*(1+gen()%240);
            Check(c,c.ResolveOverlaps(req,bytes),req,bytes);
        }
        return 0;
    }
    BufferCache c;
    const bool left=mode=="left" || mode=="lower-bound";
    const bool bound=mode=="lower-bound" || mode=="upper-bound";
    uint64_t edge=end;
    if(bound) edge=left?5*p:LOWER_ADDRESS_SIZE-4*p;
    auto add=[&](uint64_t b,uint64_t s,int score) {
        c.m_buffers.emplace(b,c.m_slot_buffers.size());
        c.m_slot_buffers.push_back({b,s,score});
    };
    constexpr uint64_t large=29556736;
    if(left) {add(edge-3*p,3*p,2);add(edge,large,15);}
    else {add(edge-large,large,15);add(edge,3*p,2);}
    if(mode=="third") add(edge+6*p,4*p,1);
    const uint64_t req=edge-p;
    auto result=c.ResolveOverlaps(req,2*p);
    Check(c,result,req,2*p);
    if(CANDIDATE) {
        if(left) Require(result.begin==(bound?2*p:edge-3*p-room) && result.end==edge+large,"left reserve");
        else Require(result.begin==edge-large && result.end==(bound?LOWER_ADDRESS_SIZE:edge+3*p+room),"right reserve");
    } else if (!left) {
        Require(result.begin==edge-large-room && result.end==edge+3*p,"old direction");
    }
    GpuCopy(gpu,c,result);
    std::printf("begin=%llx end=%llx bytes confirmed\n",(unsigned long long)result.begin,(unsigned long long)result.end);
}

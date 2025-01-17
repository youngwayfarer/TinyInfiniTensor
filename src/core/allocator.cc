#include "core/allocator.h"
#include <cstddef>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        this->used += size;

        size_t addr = this->peak;
        bool flag = false;
        size_t last_addr = this->peak;

        // find a free block that is large enough
        for (auto it = free_blocks.begin(); it != free_blocks.end(); it++)
        {
            if (it->first + it->second == this->peak)
            {
                last_addr = it->first;
                flag = true;
            }

            if (it->second >= size)
            {
                addr = it->first;
                size_t block_size = it->second;
                free_blocks.erase(it);
                if (block_size > size)
                {
                    free_blocks[addr + size] = block_size - size;
                }
                return addr;
            }
        }

        // if no free block is large enough, allocate a new block
        // there is a free block that is the last block
        if (flag)
        {
            // merge with the last block
            size_t block_size = this->peak - last_addr;
            this->peak += (size - block_size);
            free_blocks.erase(last_addr);
            return last_addr;
        }
        // allocate a new block
        this->peak += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if (addr + size == this->peak)
        {
            this->peak -= size;
            return;
        }

        for (auto it = free_blocks.begin(); it != free_blocks.end(); it++)
        {
            // merge with the previous block
            if (it->first + it->second == addr)
            {
                addr = it->first;
                size += it->second;
                free_blocks.erase(it);
            }

            // merge with the next block
            if (addr + size == it->first)
            {
                size += it->second;
                free_blocks.erase(it);
            }
        }

        free_blocks[addr] = size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}

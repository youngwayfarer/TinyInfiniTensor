#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // broadcast first
        int rankA = inputs[0]->getRank();
        int rankB = inputs[1]->getRank();
        auto dimA = inputs[0]->getDims();
        auto dimB = inputs[1]->getDims();
        if (rankA < 2 || rankB < 2)
        {
            return {};
        }

        int maxRank = std::max(rankA, rankB);
        Shape result(maxRank);
        int i = rankA - 3;
        int j = rankB - 3;
        for (; i >= 0 && j >= 0; --i, --j)
        {
            if (dimA[i] == dimB[j] || dimB[j] == 1)
            {
                result[std::max(i, j)] = dimA[i];
            }
            else if (dimA[i] == 1)
            {
                result[std::max(i, j)] = dimB[j];
            }
            else
            {
                return {};
            }
        }
        for (; i >= 0; --i)
        {
            result[i] = dimA[i];
        }
        for (; j >= 0; --j)
        {
            result[j] = dimB[j];
        }

        // transpose
        int m = dimA[rankA - 2];
        int n = dimB[rankB - 1];
        int ka = dimA[rankA - 1];
        int kb = dimB[rankB - 2];
        if (transA)
        {
            std::swap(m, ka);
        }
        if (transB)
        {
            std::swap(n, kb);
        }
        if (ka != kb)
        {
            return {};
        }

        result[maxRank - 2] = m;
        result[maxRank - 1] = n;
        return {{result}};
    }

} // namespace infini
#include "ExperienceBuffer.h"

#include <ATen/Parallel.h>

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {

}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {

	auto tIndices = torch::from_blob(
		const_cast<int64_t*>(indices),
		{ static_cast<int64_t>(size) },
		torch::TensorOptions().dtype(torch::kInt64)
	);

	ExperienceTensors result;

	auto* toPtr = result.begin();
	auto* fromPtr = data.begin();
	int64_t tensorCount = static_cast<int64_t>(result.end() - result.begin());
	at::parallel_for(0, tensorCount, 1, [&](int64_t start, int64_t end) {
		for (int64_t i = start; i < end; i++)
			toPtr[i] = torch::index_select(fromPtr[i], 0, tIndices);
	});

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	int64_t expSize = data.states.size(0);

	// Make list of shuffled sample indices
	size_t expSizeU = static_cast<size_t>(expSize);
	std::vector<int64_t> indices(expSizeU);
	std::iota(indices.begin(), indices.end(), 0); // Fill ascending indices
	std::shuffle(indices.begin(), indices.end(), rng);

	// Get a sample set from each of the batches
	std::vector<ExperienceTensors> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= expSize; startIdx += batchSize) {

		int curBatchSize = batchSize;
		if (startIdx + batchSize * 2 > expSize) {
			// Last batch of the iteration
			if (overbatching) {
				// Extend batch size to the end of the experience
				curBatchSize = expSize - startIdx;
			}
		}

		result.push_back(_GetSamples(indices.data() + startIdx, curBatchSize));
	}

	return result;
}

#pragma once

#include <string>
#include <vector>
#include "MolecularDatasetStore.h"
#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/AtomicCloudAlignmentEvaluator.h"
#include "Aligners/AtomicBottleneckEvaluator.h"
#include "Visualization/AtomicMatchingViz.h"
#include "AtomicClassifier.h"
#include "CPU/AtomicClassifierExpress.h"   // <-- brings in AtomicExpressParams & Result

class AtomicOperations
{
public:
    explicit AtomicOperations(MolecularDatasetStore &store);

    // Atomic Neighborhood Management
    void generateAllAtomicClouds();
    void generateAtomicCloudsFor(size_t molIdx);
    void clearAtomicClouds();
    size_t atomicSize() const;
    std::string getAtomicName(size_t index) const;
    size_t getAtomicAtomCount(size_t index) const;

    // Descriptors & Distances
    void computeAtomicDescriptors();
    float computeAtomicPairDistance(size_t idxA, size_t idxB, const std::string &metric, size_t N = 0) const;
    bool computeAtomicAlignerDistance(size_t idxA, size_t idxB, size_t N,
                                      const AlignmentOptions &opt = AlignmentOptions());

    // Classification & Visualization
    std::vector<AtomicClassifyMatch> runBatchAtomicClassify(const std::string &csvPath,
                                                            const AtomicClassifyParams &p,
                                                            AtomicClassifyStats &st);
    bool visualizeAtomicIndex(size_t index) const;
    bool unviewAtomicIndex(size_t index) const;

    // Fast Stage-1-only SRV pair counter
    AtomicExpressResult runAtomicExpressCount(const AtomicExpressParams &p,
                                              bool verifyAgainstBruteForce = false);

private:
    MolecularDatasetStore &m_store;
};
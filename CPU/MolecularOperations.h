#pragma once

#include <string>
#include <vector>
#include "MolecularDatasetStore.h"
#include "Aligners/CloudAlignmentEvaluator.h"
#include "Visualization/Matchingvisualizer.h"
#include "Duplicatechecker.h"
#include "Aligners/ProjectionBasedInvariantDistance.h"

class MolecularOperations
{
public:
    explicit MolecularOperations(MolecularDatasetStore &store);

    // Descriptors & Search
    void ensureDescriptorsUpToDate();
    void computeDescriptors();
    void flushDescriptors();
    bool searchKNN(size_t index, const std::string &kStr, const std::string &metricType, const std::string &csvPath = "", bool restrictToLast = false);
    bool searchKNNForAll(const std::string &kStr, const std::string &metricType, const std::string &csvPath);
    float computePairMetric(size_t idxA, size_t idxB, const std::string &metricType);
    void printDistanceDistribution() const;

    std::vector<float> getSDV(size_t index) const;
    std::vector<float> getSRV(size_t index) const;
    std::vector<float> getPDD(size_t index) const;

    // Duplicate Detection
    std::vector<DupMatch> runDuplicateCheck(size_t queryIdx, const std::string &csvPath, const DupCheckParams &p, DupRunStats &st);
    std::vector<DupMatch> runBatchDuplicateCheck(const std::string &csvPath, const DupCheckParams &p, DupRunStats &st);

    

    // Alignment & Match Management
    bool printPairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt = AlignmentOptions());
    AlignmentResult computePairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt = AlignmentOptions());
    bool printMatchInfo(size_t a, size_t b) const;
    bool setMatchVisible(size_t a, size_t b, bool v);
    void setAllMatchesVisible(bool v);
    bool removeMatch(size_t a, size_t b);
    bool applyAlignment(size_t a, size_t b);
    bool removeByIndex(size_t index);

    // Visualization & Clusters
    bool visualizeIndex(size_t index) const;
    bool unviewIndex(size_t index) const;
    void exploreClusters();
    void setGraphVisible(bool visible) const;
    void clearGraph() const;
    bool printFullProjectionDistance(size_t idxA, size_t idxB, const PbiOptions &opt = PbiOptions());

private:
    MolecularDatasetStore &m_store;

    static void writeMatchesCsv(const std::string &csvPath, const std::vector<DupMatch> &matches);
};
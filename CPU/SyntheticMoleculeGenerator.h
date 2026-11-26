#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cctype>
#include "MolecularDatasetStore.h"

class SyntheticMoleculeGenerator {
public:
    /**
     * @brief Generates synthetic QM9-like molecules with configurable atom sizes, isometric copies, and noise.
     * @param store Reference to MolecularDatasetStore to populate.
     * @param numBaseMolecules Number of distinct base structures to generate.
     * @param copiesPerMolecule Number of rotated/noisy isometric variants per base.
     * @param minAtoms Minimum number of atoms per molecule.
     * @param maxAtoms Maximum number of atoms per molecule.
     * @param noiseSigma Standard deviation of Gaussian-like noise added to isometric copies.
     */
    static size_t generate(MolecularDatasetStore& store, size_t numBaseMolecules = 50, size_t copiesPerMolecule = 3, 
                           size_t minAtoms = 5, size_t maxAtoms = 25, float noiseSigma = 0.01f) 
    {
        std::cout << "[SyntheticGenerator] Generating synthetic molecular dataset (Atoms: " << minAtoms << "-" << maxAtoms << ")...\n";
        
        const std::vector<std::string> atomTypes = {"C", "N", "O", "F", "H"};
        std::random_device rd;
        std::mt19937 gen(42); 
        std::uniform_int_distribution<size_t> atomCountDist(minAtoms, maxAtoms);
        std::uniform_int_distribution<size_t> typeDist(0, atomTypes.size() - 1);
        std::uniform_real_distribution<float> coordDist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> noiseDist(-noiseSigma, noiseSigma);
        std::uniform_real_distribution<float> rotDist(-3.14159f, 3.14159f);

        size_t totalGenerated = 0;

        for (size_t b = 0; b < numBaseMolecules; ++b) {
            size_t numAtoms = atomCountDist(gen);
            std::vector<AtomRecord> baseAtoms(numAtoms);
            for (size_t i = 0; i < numAtoms; ++i) {
                baseAtoms[i] = {
                    atomTypes[typeDist(gen)],
                    coordDist(gen),
                    coordDist(gen),
                    coordDist(gen),
                    0.0
                };
            }

            std::string baseFilename = "synth_mol_" + std::to_string(b) + "_base.xyz";
            std::string baseComment = "Generated QM9-like base molecule " + std::to_string(b);
            
            store.addSyntheticEntry(baseFilename, baseComment, baseAtoms);
            totalGenerated++;

            float rx = rotDist(gen), ry = rotDist(gen), rz = rotDist(gen);
            float cx = std::cos(rx), sx = std::sin(rx);
            float cy = std::cos(ry), sy = std::sin(ry);
            float cz = std::cos(rz), sz = std::sin(rz);

            for (size_t c = 0; c < copiesPerMolecule; ++c) {
                std::vector<AtomRecord> copyAtoms(numAtoms);
                for (size_t i = 0; i < numAtoms; ++i) {
                    float x = baseAtoms[i].x;
                    float y = baseAtoms[i].y;
                    float z = baseAtoms[i].z;

                    float y1 = y * cx - z * sx;
                    float z1 = y * sx + z * cx;
                    float x1 = x;

                    float x2 = x1 * cy + z1 * sy;
                    float z2 = -x1 * sy + z1 * cy;
                    float y2 = y1;

                    float xFinal = x2 * cz - y2 * sz + noiseDist(gen);
                    float yFinal = x2 * sz + y2 * cz + noiseDist(gen);
                    float zFinal = z2 + noiseDist(gen);

                    copyAtoms[i] = {
                        baseAtoms[i].symbol,
                        xFinal,
                        yFinal,
                        zFinal,
                        baseAtoms[i].charge
                    };
                }

                std::string copyFilename = "synth_mol_" + std::to_string(b) + "_copy_" + std::to_string(c) + ".xyz";
                std::string copyComment = "Isometric copy " + std::to_string(c) + " of base " + std::to_string(b);

                store.addSyntheticEntry(copyFilename, copyComment, copyAtoms);
                totalGenerated++;
            }
        }

        std::cout << "[SyntheticGenerator] Successfully generated and added " << totalGenerated << " synthetic molecules.\n";
        return totalGenerated;
    }

    // ------------------------------------------------------------------
    //  Degenerate / near-degenerate cloud categories (bit flags, OR them)
    // ------------------------------------------------------------------
    enum DegenerateKind : unsigned {
        DEG_COLLINEAR          = 1u << 0,  // exact line (1D)                       -> triangle test fails, 1D path
        DEG_NEAR_COLLINEAR     = 1u << 1,  // line + transverse jitter (eps sweep)  -> straddles the 1e-10 area threshold
        DEG_PLANAR             = 1u << 2,  // exact plane (2D)
        DEG_NEAR_PLANAR        = 1u << 3,  // plane + normal jitter (eps sweep)
        DEG_COINCIDENT         = 1u << 4,  // all atoms at one point (0D)           -> must fail gracefully
        DEG_NEAR_COINCIDENT    = 1u << 5,  // tiny blob of radius eps
        DEG_DUPLICATE_POINTS   = 1u << 6,  // 3D cloud where ~1/3 of atoms share exact coordinates
        DEG_EQUAL_SPACED_LINE  = 1u << 7,  // collinear, equal spacing (all distance ties)
        DEG_CENTROSYMMETRIC    = 1u << 8,  // P and -P pairs (inversion symmetry; many equivalent matchings)
        DEG_REGULAR_POLYGON    = 1u << 9,  // planar ring, n-fold symmetric
        DEG_CUBE_SHELLS        = 1u << 10, // cube vertices in scaled shells (high point-group symmetry)
        DEG_LINE_PLUS_OUTLIER  = 1u << 11, // collinear + one off-axis atom (thin triangle, valid but ill-conditioned)
        DEG_TINY_SCALE         = 1u << 12, // generic 3D cloud scaled by 1e-4 (absolute-threshold check)
        DEG_FAR_COLLINEAR      = 1u << 13, // collinear line far (~500) from origin (float cancellation on centering)
        DEG_TWO_ATOMS          = 1u << 14, // n == 2 (always collinear; smallest valid input)
        // ---- symmetric / nearly symmetric families (eps sweep applies to NEAR_*) ----
        DEG_NEAR_CENTROSYMMETRIC = 1u << 15, // P/-P pairs + isotropic jitter
        DEG_MIRROR_PLANE         = 1u << 16, // exact mirror symmetry (Cs): P and its reflection
        DEG_NEAR_MIRROR_PLANE    = 1u << 17, // mirror pairs + jitter
        DEG_PRISM                = 1u << 18, // two stacked n-gons (D_nh-like): C_n axis + mirror planes
        DEG_NEAR_PRISM           = 1u << 19, // prism + jitter
        DEG_NEAR_POLYGON         = 1u << 20, // planar ring + jitter (also leaves the plane)
        DEG_NEAR_CUBE_SHELLS     = 1u << 21, // cube shells + jitter
        DEG_ALL                  = (1u << 22) - 1u
    };
    static constexpr unsigned NUM_DEG_KINDS = 22;

    static const char* degenerateKindName(unsigned kind) {
        switch (kind) {
            case DEG_COLLINEAR:         return "collinear";
            case DEG_NEAR_COLLINEAR:    return "nearcollinear";
            case DEG_PLANAR:            return "planar";
            case DEG_NEAR_PLANAR:       return "nearplanar";
            case DEG_COINCIDENT:        return "coincident";
            case DEG_NEAR_COINCIDENT:   return "nearcoincident";
            case DEG_DUPLICATE_POINTS:  return "duplicates";
            case DEG_EQUAL_SPACED_LINE: return "equalline";
            case DEG_CENTROSYMMETRIC:   return "centrosym";
            case DEG_REGULAR_POLYGON:   return "polygon";
            case DEG_CUBE_SHELLS:       return "cubeshells";
            case DEG_LINE_PLUS_OUTLIER: return "lineoutlier";
            case DEG_TINY_SCALE:        return "tinyscale";
            case DEG_FAR_COLLINEAR:     return "farcollinear";
            case DEG_TWO_ATOMS:         return "twoatoms";
            case DEG_NEAR_CENTROSYMMETRIC: return "nearcentrosym";
            case DEG_MIRROR_PLANE:      return "mirror";
            case DEG_NEAR_MIRROR_PLANE: return "nearmirror";
            case DEG_PRISM:             return "prism";
            case DEG_NEAR_PRISM:        return "nearprism";
            case DEG_NEAR_POLYGON:      return "nearpolygon";
            case DEG_NEAR_CUBE_SHELLS:  return "nearcubeshells";
            default:                    return "unknown";
        }
    }

    // Group masks (handy for the CLI)
    static constexpr unsigned DEG_GROUP_SYMMETRIC =
        DEG_EQUAL_SPACED_LINE | DEG_CENTROSYMMETRIC | DEG_REGULAR_POLYGON | DEG_CUBE_SHELLS | DEG_MIRROR_PLANE | DEG_PRISM;
    static constexpr unsigned DEG_GROUP_NEAR_SYMMETRIC =
        DEG_NEAR_CENTROSYMMETRIC | DEG_NEAR_MIRROR_PLANE | DEG_NEAR_PRISM | DEG_NEAR_POLYGON | DEG_NEAR_CUBE_SHELLS;
    static constexpr unsigned DEG_GROUP_DEGENERATE =
        DEG_ALL & ~DEG_GROUP_SYMMETRIC & ~DEG_GROUP_NEAR_SYMMETRIC;

    /// Parses "collinear,nearplanar,symmetric" -> bitmask. Accepts every name from degenerateKindName()
    /// plus the groups: all | degenerate | symmetric | nearsymmetric. Unknown tokens are appended to *unknown.
  static unsigned parseDegenerateKinds(const std::string& list, std::string* unknown = nullptr) {
    unsigned mask = 0;
    std::stringstream ss(list);
    std::string tok;
    
    while (std::getline(ss, tok, ',')) {
        // Trim whitespace
        tok.erase(tok.begin(), std::find_if(tok.begin(), tok.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        tok.erase(std::find_if(tok.rbegin(), tok.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), tok.end());

        if (tok.empty()) continue;
        std::transform(tok.begin(), tok.end(), tok.begin(), [](unsigned char ch) { 
            return std::tolower(ch); 
        });

        if (tok == "all") { mask |= DEG_ALL; continue; }
        if (tok == "degenerate") { mask |= DEG_GROUP_DEGENERATE; continue; }
        if (tok == "symmetric") { mask |= DEG_GROUP_SYMMETRIC; continue; }
        if (tok == "nearsymmetric") { mask |= DEG_GROUP_NEAR_SYMMETRIC; continue; }

        bool found = false;
        for (unsigned bit = 0; bit < NUM_DEG_KINDS; ++bit) {
            if (tok == degenerateKindName(1u << bit)) {
                mask |= (1u << bit);
                found = true;
                break;
            }
        }
        if (!found && unknown) {
            *unknown += (unknown->empty() ? "" : ", ") + tok;
        }
    }
    return mask;
}

    /**
     * @brief Generates degenerate / near-degenerate point clouds (plus rotated, translated, shuffled copies).
     *
     * Each selected category gets `basesPerKind` base clouds. Every base is followed by `copiesPerBase`
     * copies: random proper rotation + random translation + optional atom reordering + isotropic noise.
     * Entry names look like  synth_deg_<kind>_<b>_base.xyz / synth_deg_<kind>_<b>_copy_<c>.xyz,
     * so the evaluator test can pair base <-> copies by name and group results by category.
     *
     * @param kindMask      OR of DegenerateKind flags (DEG_ALL = everything).
     * @param basesPerKind  Base clouds per category.
     * @param copiesPerBase Copies per base.
     * @param minAtoms      Min atoms (clamped to >=2; categories needing more are bumped automatically).
     * @param maxAtoms      Max atoms (clamped to <=64, the evaluator's MAX_VERTICES).
     * @param noiseSigma    Per-coordinate uniform noise on COPIES. 0 keeps degenerate copies exactly degenerate;
     *                      >0 makes B 3D-noisy while A (the base) stays degenerate. Note: if noiseSigma >> nearEps,
     *                      the near-degenerate structure is swamped in the copy.
     * @param nearEps       Jitter amplitude for NEAR_* categories. <=0 sweeps {1e-7..1e-2} across bases so
     *                      you land on both sides of the evaluator's collinearity threshold.
     * @param shuffleCopies Randomly permute atom order in copies so the matching is non-trivial.
     * @param seed          RNG seed (deterministic).
     * @param singleElement If true every atom is "C". Needed for symmetric categories if you run the evaluator with
     *                      requireSameSymbol, otherwise random symbols break the symmetry you want to test.
     */
    static size_t generateDegenerate(MolecularDatasetStore& store,
                                     unsigned kindMask = DEG_ALL,
                                     size_t basesPerKind = 10,
                                     size_t copiesPerBase = 3,
                                     size_t minAtoms = 4,
                                     size_t maxAtoms = 25,
                                     float noiseSigma = 0.0f,
                                     float nearEps = 0.0f,
                                     bool shuffleCopies = true,
                                     unsigned seed = 1234,
                                     bool singleElement = false)
    {
        const size_t MAXV = 64;
        minAtoms = std::max<size_t>(2, minAtoms);
        maxAtoms = std::min(MAXV, std::max(minAtoms, maxAtoms));

        std::cout << "[SyntheticGenerator] Generating degenerate clouds (mask=0x" << std::hex << kindMask << std::dec
                  << ", atoms " << minAtoms << "-" << maxAtoms << ")...\n";

        const std::vector<std::string> atomTypes = {"C", "N", "O", "F", "H"};
        const float epsSweep[] = {1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f};
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> atomCountDist(minAtoms, maxAtoms);
        std::uniform_int_distribution<size_t> typeDist(0, atomTypes.size() - 1);
        std::uniform_real_distribution<float> U01(0.0f, 1.0f);
        std::uniform_real_distribution<float> coordDist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> rotDist(-3.14159f, 3.14159f);
        std::uniform_real_distribution<float> shiftDist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> noiseDist(-std::max(noiseSigma, 0.0f), std::max(noiseSigma, 0.0f));

        struct V3 { float x, y, z; };
        auto cross = [](const V3& a, const V3& b) { return V3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; };
        auto normalize = [](V3 v) { float l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); if (l < 1e-20f) return V3{1,0,0}; return V3{v.x/l, v.y/l, v.z/l}; };
        auto randDir = [&]() {
            // rejection sampling on the unit ball -> uniform direction
            for (;;) {
                V3 v{coordDist(gen) / 3.0f, coordDist(gen) / 3.0f, coordDist(gen) / 3.0f};
                float l2 = v.x*v.x + v.y*v.y + v.z*v.z;
                if (l2 > 0.01f && l2 <= 1.0f) return normalize(v);
            }
        };
        auto randPerp = [&](const V3& d) {
            V3 r = randDir();
            V3 p = cross(d, r);
            return normalize(p);
        };

        size_t total = 0;

        for (unsigned bit = 0; bit < NUM_DEG_KINDS; ++bit) {
            const unsigned kind = 1u << bit;
            if (!(kindMask & kind)) continue;
            const std::string kname = degenerateKindName(kind);

            for (size_t b = 0; b < basesPerKind; ++b) {
                size_t n = atomCountDist(gen);
                if (kind == DEG_TWO_ATOMS) n = 2;
                if (kind == DEG_LINE_PLUS_OUTLIER || kind == DEG_DUPLICATE_POINTS) n = std::max<size_t>(n, 4);
                const float eps = (nearEps > 0.0f) ? nearEps : epsSweep[b % (sizeof(epsSweep) / sizeof(epsSweep[0]))];

                std::vector<V3> P(n, V3{0, 0, 0});
                float epsUsed = 0.0f;

                switch (kind) {
                case DEG_COLLINEAR:
                case DEG_NEAR_COLLINEAR:
                case DEG_FAR_COLLINEAR:
                case DEG_TWO_ATOMS:
                case DEG_LINE_PLUS_OUTLIER: {
                    V3 d = randDir(), u = randPerp(d), v = cross(d, u);
                    V3 c0{0, 0, 0};
                    if (kind == DEG_FAR_COLLINEAR) c0 = V3{500.0f * d.y + 300.0f, -400.0f, 500.0f * d.z};
                    for (size_t i = 0; i < n; ++i) {
                        float t = coordDist(gen);
                        float ju = 0.0f, jv = 0.0f;
                        if (kind == DEG_NEAR_COLLINEAR) {
                            ju = eps * (2.0f * U01(gen) - 1.0f);
                            jv = eps * (2.0f * U01(gen) - 1.0f);
                            epsUsed = eps;
                        }
                        P[i] = {c0.x + t*d.x + ju*u.x + jv*v.x,
                                c0.y + t*d.y + ju*u.y + jv*v.y,
                                c0.z + t*d.z + ju*u.z + jv*v.z};
                    }
                    if (kind == DEG_LINE_PLUS_OUTLIER) {
                        float off = 0.5f + 1.5f * U01(gen);
                        P[n - 1].x += off * u.x; P[n - 1].y += off * u.y; P[n - 1].z += off * u.z;
                    }
                    break;
                }
                case DEG_EQUAL_SPACED_LINE: {
                    V3 d = randDir();
                    float step = 0.5f + U01(gen);
                    for (size_t i = 0; i < n; ++i)
                        P[i] = {step * i * d.x, step * i * d.y, step * i * d.z};
                    break;
                }
                case DEG_PLANAR:
                case DEG_NEAR_PLANAR: {
                    V3 u = randDir(), v = randPerp(u), w = cross(u, v);
                    for (size_t i = 0; i < n; ++i) {
                        float a = coordDist(gen), c = coordDist(gen), h = 0.0f;
                        if (kind == DEG_NEAR_PLANAR) { h = eps * (2.0f * U01(gen) - 1.0f); epsUsed = eps; }
                        P[i] = {a*u.x + c*v.x + h*w.x, a*u.y + c*v.y + h*w.y, a*u.z + c*v.z + h*w.z};
                    }
                    break;
                }
                case DEG_COINCIDENT:
                case DEG_NEAR_COINCIDENT: {
                    for (size_t i = 0; i < n; ++i) {
                        float j = (kind == DEG_NEAR_COINCIDENT) ? eps : 0.0f;
                        P[i] = {1.0f + j * (2*U01(gen) - 1), -2.0f + j * (2*U01(gen) - 1), 0.5f + j * (2*U01(gen) - 1)};
                    }
                    if (kind == DEG_NEAR_COINCIDENT) epsUsed = eps;
                    break;
                }
                case DEG_DUPLICATE_POINTS: {
                    for (size_t i = 0; i < n; ++i) P[i] = {coordDist(gen), coordDist(gen), coordDist(gen)};
                    const size_t nd = std::max<size_t>(1, n / 3);
                    for (size_t k = 0; k < nd; ++k) {
                        size_t dst = 1 + (gen() % (n - 1));
                        size_t src = gen() % dst;
                        P[dst] = P[src];
                    }
                    break;
                }
                case DEG_CENTROSYMMETRIC:
                case DEG_NEAR_CENTROSYMMETRIC: {
                    const size_t half = n / 2;
                    for (size_t i = 0; i < half; ++i) {
                        V3 p{coordDist(gen), coordDist(gen), coordDist(gen)};
                        P[i] = p;
                        P[half + i] = {-p.x, -p.y, -p.z};
                    }
                    if (n % 2) P[n - 1] = {0, 0, 0};
                    break;
                }
                case DEG_REGULAR_POLYGON:
                case DEG_NEAR_POLYGON: {
                    V3 u = randDir(), v = randPerp(u);
                    const float r = 1.0f + 2.0f * U01(gen);
                    for (size_t i = 0; i < n; ++i) {
                        float ang = 2.0f * 3.14159265f * float(i) / float(n);
                        P[i] = {r*(std::cos(ang)*u.x + std::sin(ang)*v.x),
                                r*(std::cos(ang)*u.y + std::sin(ang)*v.y),
                                r*(std::cos(ang)*u.z + std::sin(ang)*v.z)};
                    }
                    break;
                }
                case DEG_CUBE_SHELLS:
                case DEG_NEAR_CUBE_SHELLS: {
                    for (size_t i = 0; i < n; ++i) {
                        const size_t shell = i / 8, idx = i % 8;
                        const float s = 1.0f + 0.37f * shell;
                        P[i] = {(idx & 1 ? s : -s), (idx & 2 ? s : -s), (idx & 4 ? s : -s)};
                    }
                    break;
                }
                case DEG_MIRROR_PLANE:
                case DEG_NEAR_MIRROR_PLANE: {
                    const V3 w = randDir();
                    const size_t half = n / 2;
                    for (size_t i = 0; i < half; ++i) {
                        V3 p{coordDist(gen), coordDist(gen), coordDist(gen)};
                        float d = p.x*w.x + p.y*w.y + p.z*w.z;
                        P[i] = p;
                        P[half + i] = {p.x - 2*d*w.x, p.y - 2*d*w.y, p.z - 2*d*w.z};
                    }
                    if (n % 2) { // odd atom lies in the mirror plane
                        V3 p{coordDist(gen), coordDist(gen), coordDist(gen)};
                        float d = p.x*w.x + p.y*w.y + p.z*w.z;
                        P[n - 1] = {p.x - d*w.x, p.y - d*w.y, p.z - d*w.z};
                    }
                    break;
                }
                case DEG_PRISM:
                case DEG_NEAR_PRISM: {
                    const V3 ax = randDir(), u = randPerp(ax), v = cross(ax, u);
                    const size_t m = std::max<size_t>(1, n / 2);
                    const float r = 1.0f + 2.0f * U01(gen), h = 0.5f + 1.5f * U01(gen);
                    for (size_t i = 0; i < m; ++i) {
                        float ang = 2.0f * 3.14159265f * float(i) / float(m);
                        float ca = r * std::cos(ang), sa = r * std::sin(ang);
                        V3 ring{ca*u.x + sa*v.x, ca*u.y + sa*v.y, ca*u.z + sa*v.z};
                        P[i]     = {ring.x + h*ax.x, ring.y + h*ax.y, ring.z + h*ax.z};
                        P[m + i] = {ring.x - h*ax.x, ring.y - h*ax.y, ring.z - h*ax.z};
                    }
                    for (size_t i = 2 * m; i < n; ++i) P[i] = {0, 0, 0}; // odd atom on the C_n axis
                    break;
                }
                case DEG_TINY_SCALE: {
                    for (size_t i = 0; i < n; ++i) P[i] = {1e-4f * coordDist(gen), 1e-4f * coordDist(gen), 1e-4f * coordDist(gen)};
                    break;
                }
                default: break;
                }

                const unsigned nearSymMask = DEG_NEAR_CENTROSYMMETRIC | DEG_NEAR_MIRROR_PLANE | DEG_NEAR_PRISM |
                                             DEG_NEAR_POLYGON | DEG_NEAR_CUBE_SHELLS;
                if (kind & nearSymMask) {
                    for (auto& q : P) {
                        q.x += eps * (2.0f * U01(gen) - 1.0f);
                        q.y += eps * (2.0f * U01(gen) - 1.0f);
                        q.z += eps * (2.0f * U01(gen) - 1.0f);
                    }
                    epsUsed = eps;
                }

                std::vector<std::string> sym(n);
                for (size_t i = 0; i < n; ++i) sym[i] = singleElement ? std::string("C") : atomTypes[typeDist(gen)];

                std::string tag = "synth_deg_" + kname + "_" + std::to_string(b);
                std::string desc = std::string("Degenerate category=") + kname + " n=" + std::to_string(n);
                if (epsUsed > 0.0f) desc += " eps=" + std::to_string(epsUsed);

                // base
                {
                    std::vector<AtomRecord> atoms(n);
                    for (size_t i = 0; i < n; ++i) atoms[i] = {sym[i], P[i].x, P[i].y, P[i].z, 0.0};
                    store.addSyntheticEntry(tag + "_base.xyz", desc + " (base)", atoms);
                    ++total;
                }

                // copies
                for (size_t c = 0; c < copiesPerBase; ++c) {
                    float rx = rotDist(gen), ry = rotDist(gen), rz = rotDist(gen);
                    float cx = std::cos(rx), sx = std::sin(rx), cy = std::cos(ry), sy = std::sin(ry), cz = std::cos(rz), sz = std::sin(rz);
                    float tx = shiftDist(gen), ty = shiftDist(gen), tz = shiftDist(gen);

                    std::vector<size_t> perm(n);
                    std::iota(perm.begin(), perm.end(), size_t(0));
                    if (shuffleCopies) std::shuffle(perm.begin(), perm.end(), gen);

                    std::vector<AtomRecord> atoms(n);
                    for (size_t i = 0; i < n; ++i) {
                        float x = P[i].x, y = P[i].y, z = P[i].z;
                        float y1 = y * cx - z * sx, z1 = y * sx + z * cx, x1 = x;
                        float x2 = x1 * cy + z1 * sy, z2 = -x1 * sy + z1 * cy, y2 = y1;
                        float xf = x2 * cz - y2 * sz + tx;
                        float yf = x2 * sz + y2 * cz + ty;
                        float zf = z2 + tz;
                        if (noiseSigma > 0.0f) { xf += noiseDist(gen); yf += noiseDist(gen); zf += noiseDist(gen); }
                        atoms[perm[i]] = {sym[i], xf, yf, zf, 0.0};
                    }
                    store.addSyntheticEntry(tag + "_copy_" + std::to_string(c) + ".xyz",
                                            desc + " (copy " + std::to_string(c) + ", shuffled=" + (shuffleCopies ? "1" : "0") + ")",
                                            atoms);
                    ++total;
                }
            }
        }

        std::cout << "[SyntheticGenerator] Added " << total << " degenerate/near-degenerate molecules.\n";
        return total;
    }
};
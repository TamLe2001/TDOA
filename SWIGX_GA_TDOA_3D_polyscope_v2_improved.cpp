// SWIGX_GA_TDOA_3D_polyscope.cpp
// Polyscope version of the simulator, following the Python and Win32 C++ logic.
//
// Build notes:
// - This file depends on Polyscope and its dependencies (GLM, ImGui via Polyscope).
// - Recommended: build with CMake and link against polyscope target.

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <string>       // FIX: was missing; required for std::string in AppState
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <imgui.h>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/view.h"

// ---------------------------------------------------------------------------
// Vec3 — minimal 3-D double-precision vector
// ---------------------------------------------------------------------------
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s)      const { return {x * s,   y * s,   z * s  }; }

    // FIX: guard against division by zero (assert in debug, clamp in release)
    Vec3 operator/(double s) const {
        assert(std::fabs(s) > 1e-300 && "Vec3 division by zero");
        return {x / s, y / s, z / s};
    }

    // IMPROVEMENT: convenience helpers
    double lengthSq()   const { return x*x + y*y + z*z; }
    double length()     const { return std::sqrt(lengthSq()); }

    [[nodiscard]] Vec3 normalized() const {
        const double len = length();
        if (len < 1e-12) return {0, 0, 0};
        return *this / len;
    }
};

static double dot  (const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double norm (const Vec3& v)                 { return v.length(); }

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static glm::vec3 toGlm(const Vec3& v) {
    return glm::vec3{static_cast<float>(v.x),
                     static_cast<float>(v.y),
                     static_cast<float>(v.z)};
}

// ---------------------------------------------------------------------------
// Three-sphere intersection (trilateration).
// Returns the two candidate points (above/below the plane), or nullopt.
// ---------------------------------------------------------------------------
[[nodiscard]] static std::optional<std::pair<Vec3, Vec3>> intersectThreeSpheres(
    const Vec3& c1, const Vec3& c2, const Vec3& c3,
    double r1, double r2, double r3)
{
    const Vec3   c12 = c2 - c1;
    const double d   = norm(c12);
    if (d < 1e-12) return std::nullopt;

    const Vec3   ex  = c12 / d;
    const Vec3   c13 = c3 - c1;
    const double i   = dot(ex, c13);
    const Vec3   tmp = c13 - ex * i;
    const double tmpLen = norm(tmp);
    if (tmpLen < 1e-12) return std::nullopt;

    const Vec3   ey = tmp / tmpLen;
    const Vec3   ez = cross(ex, ey);
    const double j  = dot(ey, c13);
    if (std::fabs(j) < 1e-12) return std::nullopt;

    const double lx = (r1*r1 - r2*r2 + d*d) / (2.0 * d);
    const double ly = (r1*r1 - r3*r3 + i*i + j*j) / (2.0 * j) - (i / j) * lx;

    double z2 = r1*r1 - lx*lx - ly*ly;
    if (z2 < -1e-8) return std::nullopt;   // no real intersection
    z2 = std::max(0.0, z2);
    const double lz = std::sqrt(z2);

    const Vec3 base = c1 + ex * lx + ey * ly;
    return std::make_pair(base + ez * lz, base - ez * lz);
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------
struct Simulator {
    // 8-mic cube: bottom face z=0, top face z=2
    std::array<Vec3, 8> mics{{
        Vec3{+1.0, +1.0, 0.0},   // 0 = A
        Vec3{-1.0, +1.0, 0.0},   // 1 = B
        Vec3{-1.0, -1.0, 0.0},   // 2 = C
        Vec3{+1.0, -1.0, 0.0},   // 3 = D
        Vec3{+1.0, +1.0, 2.0},   // 4 = A2
        Vec3{-1.0, +1.0, 2.0},   // 5 = B2
        Vec3{-1.0, -1.0, 2.0},   // 6 = C2
        Vec3{+1.0, -1.0, 2.0}    // 7 = D2
    }};

    double x = 30.0;
    double y = 50.0;
    double z =  1.0;

    // FIX: use brace-init to avoid uninitialized reads before first solveTDOA
    Vec3 target  {30.0, 50.0, 1.0};
    Vec3 estimate {30.0, 50.0, 1.0};
    Vec3 estimate2{30.0, 50.0, 1.0};

    bool   hasPair  = false;
    double bestScale = 0.0;   // FIX: was misleadingly 1.0 when hasPair=false
    double pairGap  = std::numeric_limits<double>::infinity();

    // IMPROVEMENT: expose threshold so the UI can tune it
    double dthres = 0.25;

    void syncTargetFromXYZ() { target = Vec3{x, y, z}; }

    // Python-equivalent thresholded axis update from roll/pitch/heave.
    void applyAxisThresholdStep(double roll, double pitch, double heave, double step) {
        if (roll  >  0.2) x += step;
        if (roll  < -0.2) x -= step;
        if (pitch >  0.2) y -= step;   // note: intentional sign flip matching Python
        if (pitch < -0.2) y += step;
        if (heave >  0.2) z -= step;
        if (heave < -0.2) z += step;
    }

    [[nodiscard]] std::array<double, 8> distancesFrom(const Vec3& p) const {
        std::array<double, 8> d{};
        for (size_t i = 0; i < mics.size(); ++i)
            d[i] = norm(p - mics[i]);
        return d;
    }

    void solveTDOA() {
        syncTargetFromXYZ();
        const auto trueD = distancesFrom(target);

        // FIX: reset to "no result" state
        hasPair   = false;
        bestScale = 0.0;
        pairGap   = std::numeric_limits<double>::infinity();

        // FIX: IMPROVEMENT — find the BEST (minimum-gap) scale instead of
        // the first-under-threshold scale.  The original code broke on the
        // first match, which could return a false positive at an early scale
        // well before the true scale ≈ 1.0.
        //
        // IMPROVEMENT: finer resolution (0.5 % steps instead of 1 %) and
        // extended range 0.25 – 1.75 for robustness.
        // FIX: dthres moved outside the loop (was re-created 80× per call).
        Vec3   bestA{}, bestB{};
        double bestGap = std::numeric_limits<double>::infinity();
        double bestScaleTmp = 0.0;

        for (int scaleR = 25; scaleR <= 175; ++scaleR) {
            const double scale = static_cast<double>(scaleR) / 100.0;

            std::array<double, 8> scaledD{};
            for (size_t i = 0; i < scaledD.size(); ++i)
                scaledD[i] = trueD[i] * scale;

            // Triplet 1: mics 0(A), 1(B), 6(C2)
            // Triplet 2: mics 4(A2), 5(B2), 2(C)
            const auto pair123 = intersectThreeSpheres(
                mics[0], mics[1], mics[6],
                scaledD[0], scaledD[1], scaledD[6]);
            const auto pair456 = intersectThreeSpheres(
                mics[4], mics[5], mics[2],
                scaledD[4], scaledD[5], scaledD[2]);

            if (!pair123 || !pair456) continue;

            const Vec3 pt1 = pair123->first,  pt2 = pair123->second;
            const Vec3 pt3 = pair456->first,  pt4 = pair456->second;

            // All 6 cross-triplet distances
            const double gaps[6] = {
                norm(pt1 - pt2), norm(pt1 - pt3), norm(pt1 - pt4),
                norm(pt2 - pt3), norm(pt2 - pt4), norm(pt3 - pt4)
            };
            // Corresponding point pairs
            const Vec3 pairA[6] = {pt1, pt1, pt1, pt2, pt2, pt3};
            const Vec3 pairB[6] = {pt2, pt3, pt4, pt3, pt4, pt4};

            for (int k = 0; k < 6; ++k) {
                if (gaps[k] < bestGap) {
                    bestGap      = gaps[k];
                    bestA        = pairA[k];
                    bestB        = pairB[k];
                    bestScaleTmp = scale;
                }
            }
        }

        if (bestGap < dthres) {
            estimate  = bestA;
            estimate2 = bestB;
            hasPair   = true;
            bestScale = bestScaleTmp;
            pairGap   = bestGap;
        }
    }
};

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
struct AppState {
    Simulator sim;

    double step = 3.0;
    float  roll  = 0.0f;
    float  pitch = 0.0f;
    float  heave = 1.0f;
    bool   keyboardControlsEnabled = true;
    std::string lastInput = "none";

    polyscope::PointCloud*  psMics         = nullptr;
    polyscope::PointCloud*  psTarget       = nullptr;
    polyscope::PointCloud*  psEstimate     = nullptr;
    polyscope::PointCloud*  psEstimate2    = nullptr;
    polyscope::CurveNetwork* psEstimateLink = nullptr;
};

static AppState gApp;
static void updateSceneGeometry();

static void moveTargetX(double delta) { gApp.sim.x += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }
static void moveTargetY(double delta) { gApp.sim.y += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }
static void moveTargetZ(double delta) { gApp.sim.z += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }

static void updateSceneGeometry() {
    gApp.sim.syncTargetFromXYZ();
    gApp.psTarget->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.target)});

    if (gApp.sim.hasPair) {
        gApp.psEstimate ->updatePointPositions({toGlm(gApp.sim.estimate )});
        gApp.psEstimate2->updatePointPositions({toGlm(gApp.sim.estimate2)});
        gApp.psEstimateLink->updateNodePositions(
            {toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)});
        gApp.psEstimate ->setEnabled(true);
        gApp.psEstimate2->setEnabled(true);
        gApp.psEstimateLink->setEnabled(true);
    } else {
        gApp.psEstimate ->setEnabled(false);
        gApp.psEstimate2->setEnabled(false);
        gApp.psEstimateLink->setEnabled(false);
    }
    polyscope::requestRedraw();
}

static void centerViewOnTarget() {
    const glm::vec3 tgt = toGlm(gApp.sim.target);
    polyscope::view::lookAt(tgt + glm::vec3{0.f, 0.f, 120.f}, tgt);
}

// ---------------------------------------------------------------------------
// Keyboard movement
// FIX: accumulate per-axis deltas so pressing two bindings for the same axis
//      in one frame does NOT move 2×step.
// ---------------------------------------------------------------------------
static void applyKeyboardMovement() {
    if (!gApp.keyboardControlsEnabled) return;

    double dx = 0.0, dy = 0.0, dz = 0.0;
    const char* src = nullptr;

    // Primary bindings: arrow keys + R/F
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow,  false)) { dx -= gApp.step; src = "LeftArrow";  }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) { dx += gApp.step; src = "RightArrow"; }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow,    false)) { dy += gApp.step; src = "UpArrow";    }
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow,  false)) { dy -= gApp.step; src = "DownArrow";  }
    // R = up (z-), F = down (z+), matching Python heave convention
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) { dz -= gApp.step; src = "R"; }
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) { dz += gApp.step; src = "F"; }

    // Fallback / alternative WASD+QE via Polyscope engine
    // FIX: clamp so simultaneous arrow+WASD on same axis still moves 1×step
    if (polyscope::render::engine->isKeyPressed('a') && dx == 0.0) { dx -= gApp.step; src = "a"; }
    if (polyscope::render::engine->isKeyPressed('d') && dx == 0.0) { dx += gApp.step; src = "d"; }
    if (polyscope::render::engine->isKeyPressed('w') && dy == 0.0) { dy += gApp.step; src = "w"; }
    if (polyscope::render::engine->isKeyPressed('s') && dy == 0.0) { dy -= gApp.step; src = "s"; }
    if (polyscope::render::engine->isKeyPressed('q') && dz == 0.0) { dz += gApp.step; src = "q"; }
    if (polyscope::render::engine->isKeyPressed('e') && dz == 0.0) { dz -= gApp.step; src = "e"; }

    if (dx != 0.0 || dy != 0.0 || dz != 0.0) {
        gApp.sim.x += dx;
        gApp.sim.y += dy;
        gApp.sim.z += dz;
        if (src) gApp.lastInput = src;
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }
}

// ---------------------------------------------------------------------------
// ImGui panel
// ---------------------------------------------------------------------------
static void polyscopeUI() {
    ImGui::PushItemWidth(220.0f);

    ImGui::TextUnformatted("SWIGX GA TDOA 3D (Polyscope)");
    ImGui::Separator();

    ImGui::TextUnformatted("Controls");
    ImGui::SliderFloat("roll",  &gApp.roll,  -1.0f, 1.0f);
    ImGui::SliderFloat("pitch", &gApp.pitch, -1.0f, 1.0f);
    ImGui::SliderFloat("heave", &gApp.heave, -1.0f, 1.0f);
    ImGui::Checkbox("Keyboard controls", &gApp.keyboardControlsEnabled);

    double targetX = gApp.sim.x, targetY = gApp.sim.y, targetZ = gApp.sim.z;
    bool targetChanged = false;
    targetChanged |= ImGui::InputDouble("target x", &targetX, 0.1, 1.0, "%.2f");
    targetChanged |= ImGui::InputDouble("target y", &targetY, 0.1, 1.0, "%.2f");
    targetChanged |= ImGui::InputDouble("target z", &targetZ, 0.1, 1.0, "%.2f");
    if (targetChanged) {
        gApp.sim.x = targetX; gApp.sim.y = targetY; gApp.sim.z = targetZ;
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }

    ImGui::TextUnformatted("Nudge target");
    if (ImGui::Button("X -")) moveTargetX(-gApp.step); ImGui::SameLine();
    if (ImGui::Button("X +")) moveTargetX( gApp.step); ImGui::SameLine();
    if (ImGui::Button("Y -")) moveTargetY(-gApp.step); ImGui::SameLine();
    if (ImGui::Button("Y +")) moveTargetY( gApp.step);
    if (ImGui::Button("Z -")) moveTargetZ(-gApp.step); ImGui::SameLine();
    if (ImGui::Button("Z +")) moveTargetZ( gApp.step);

    if (ImGui::Button("Center View")) centerViewOnTarget();

    if (ImGui::Button("Apply axis threshold step")) {
        gApp.sim.applyAxisThresholdStep(gApp.roll, gApp.pitch, gApp.heave, gApp.step);
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }
    ImGui::SameLine();
    if (ImGui::Button("Solve")) { gApp.sim.solveTDOA(); updateSceneGeometry(); }

    // FIX: clamp step to a positive minimum so movement/solve can't break
    ImGui::InputDouble("step", &gApp.step, 0.1, 1.0, "%.2f");
    if (gApp.step <= 0.0) gApp.step = 0.1;

    // IMPROVEMENT: expose dthres in UI for live tuning
    ImGui::InputDouble("dthres", &gApp.sim.dthres, 0.01, 0.1, "%.3f");
    if (gApp.sim.dthres <= 0.0) gApp.sim.dthres = 0.01;

    ImGui::Separator();
    ImGui::Text("Arrow keys move X/Y, R/F move Z");
    ImGui::Text("Fallback keys: A/D X, W/S Y, Q/E Z");
    ImGui::Text("Last input: %s", gApp.lastInput.c_str());
    ImGui::Text("Target: (%.2f, %.2f, %.2f)",
        gApp.sim.target.x, gApp.sim.target.y, gApp.sim.target.z);
    ImGui::Text("scale=%.3f", gApp.sim.bestScale);

    if (gApp.sim.hasPair) {
        ImGui::Text("Estimate1: (%.2f, %.2f, %.2f)",
            gApp.sim.estimate.x,  gApp.sim.estimate.y,  gApp.sim.estimate.z);
        ImGui::Text("Estimate2: (%.2f, %.2f, %.2f)",
            gApp.sim.estimate2.x, gApp.sim.estimate2.y, gApp.sim.estimate2.z);
        ImGui::Text("pair_gap=%.6f", gApp.sim.pairGap);

        // IMPROVEMENT: show error vector so you can see solve accuracy
        const Vec3 err = gApp.sim.estimate - gApp.sim.target;
        ImGui::Text("err=(%.3f, %.3f, %.3f)  |err|=%.3f",
            err.x, err.y, err.z, norm(err));
    } else {
        ImGui::TextColored({1.0f, 0.4f, 0.4f, 1.0f}, "No valid pair under threshold");
    }

    ImGui::PopItemWidth();
    applyKeyboardMovement();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::init();

    // Microphones
    std::vector<glm::vec3> micPoints;
    micPoints.reserve(gApp.sim.mics.size());
    for (const Vec3& m : gApp.sim.mics) micPoints.push_back(toGlm(m));

    gApp.psMics = polyscope::registerPointCloud("Mics", micPoints);
    gApp.psMics->setPointColor(glm::vec3{0.12f, 0.45f, 0.95f});
    // FIX: was setPointRadius(0.01, true) — relative mode with large scene
    //      made mics unexpectedly large.  Use absolute (false) like the others.
    gApp.psMics->setPointRadius(0.08, false);

    // Target
    gApp.psTarget = polyscope::registerPointCloud("Target",
        std::vector<glm::vec3>{toGlm(gApp.sim.target)});
    gApp.psTarget->setPointColor(glm::vec3{0.15f, 0.80f, 0.25f});
    gApp.psTarget->setPointRadius(0.35, false);
    gApp.psTarget->setEnabled(true);

    // Estimate 1
    gApp.psEstimate = polyscope::registerPointCloud("Estimate 1",
        std::vector<glm::vec3>{toGlm(gApp.sim.estimate)});
    gApp.psEstimate->setPointColor(glm::vec3{0.85f, 0.15f, 0.15f});
    gApp.psEstimate->setPointRadius(0.25, false);

    // Estimate 2
    gApp.psEstimate2 = polyscope::registerPointCloud("Estimate 2",
        std::vector<glm::vec3>{toGlm(gApp.sim.estimate2)});
    gApp.psEstimate2->setPointColor(glm::vec3{0.85f, 0.15f, 0.15f}); // same red as Estimate 1, matching Python tab:red
    gApp.psEstimate2->setPointRadius(0.25, false);

    // Link line between estimates
    std::vector<glm::vec3>           linkNodes{toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)};
    std::vector<std::array<size_t,2>> linkEdges{{{0, 1}}};
    gApp.psEstimateLink = polyscope::registerCurveNetwork("Estimate Link", linkNodes, linkEdges);
    gApp.psEstimateLink->setColor(glm::vec3{0.98f, 0.45f, 0.12f});
    gApp.psEstimateLink->setRadius(0.002, true);

    gApp.sim.solveTDOA();
    updateSceneGeometry();
    centerViewOnTarget();

    polyscope::state::userCallback = polyscopeUI;
    polyscope::show();

    return 0;
}

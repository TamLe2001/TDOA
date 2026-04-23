// SWIGX_GA_TDOA_3D_polyscope.cpp
// Polyscope version of the simulator, following the Python and Win32 C++ logic.
//
// Build notes:
// - This file depends on Polyscope and its dependencies (GLM, ImGui via Polyscope).
// - Recommended: build with CMake and link against polyscope target.
//
// Build command (MSVC on Windows, VS2022 bundled CMake):
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --config Release

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <string>   // required for std::string in AppState
#include <utility>
#include <vector>

#include "imgui.h"
#include <glm/glm.hpp>

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

    // FIX: guard against division by zero
    Vec3 operator/(double s) const {
        assert(std::fabs(s) > 1e-300 && "Vec3 division by zero");
        return {x / s, y / s, z / s};
    }

    // IMPROVEMENT: convenience helpers
    double lengthSq() const { return x*x + y*y + z*z; }
    double length()   const { return std::sqrt(lengthSq()); }

    [[nodiscard]] Vec3 normalized() const {
        const double len = length();
        if (len < 1e-12) return {0, 0, 0};
        return *this / len;
    }
};

static double dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static double norm(const Vec3& v) { return v.length(); }

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

[[nodiscard]] static std::optional<std::pair<Vec3, Vec3>> intersectThreeSpheres(
    const Vec3& c1, const Vec3& c2, const Vec3& c3,
    double r1, double r2, double r3);

struct Sphere {
    Vec3 center;
    double radius = 0.0;
};

static Sphere roundFromCenterRadius(const Vec3& center, double radius) {
    return Sphere{center, radius};
}

static std::optional<std::pair<Vec3, Vec3>> pointPairToEndPoints(
    const Sphere& s1,
    const Sphere& s2,
    const Sphere& s3
) {
    return intersectThreeSpheres(
        s1.center, s2.center, s3.center,
        s1.radius, s2.radius, s3.radius);
}

// ---------------------------------------------------------------------------
// Three-sphere intersection (trilateration).
// Returns the two candidate points (above/below the plane), or nullopt.
// ---------------------------------------------------------------------------
[[nodiscard]] static std::optional<std::pair<Vec3, Vec3>> intersectThreeSpheres(
    const Vec3& c1, const Vec3& c2, const Vec3& c3,
    double r1, double r2, double r3)
{
    const Vec3 c12 = c2 - c1;
    const double d = norm(c12);
    if (d < 1e-12) return std::nullopt;

    const Vec3 ex = c12 / d;
    const Vec3 c13 = c3 - c1;
    const double i = dot(ex, c13);
    const Vec3 tmp = c13 - ex * i;
    const double tmpLen = norm(tmp);
    if (tmpLen < 1e-12) return std::nullopt;

    const Vec3 ey = tmp / tmpLen;
    const Vec3 ez = cross(ex, ey);
    const double j = dot(ey, c13);
    if (std::fabs(j) < 1e-12) return std::nullopt;

    const double lx = (r1*r1 - r2*r2 + d*d) / (2.0 * d);
    const double ly = (r1*r1 - r3*r3 + i*i + j*j) / (2.0 * j) - (i / j) * lx;

    double z2 = r1*r1 - lx*lx - ly*ly;
    if (z2 < -1e-8) return std::nullopt; // no real intersection
    z2 = std::max(0.0, z2);
    const double lz = std::sqrt(z2);

    const Vec3 base = c1 + ex * lx + ey * ly;
    return std::make_pair(base + ez * lz, base - ez * lz);
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------
struct Simulator {
    // 8-mic cube: enlarged footprint to increase sensor area
    static constexpr double kSensorHalfSpan = 10;
    static constexpr double kSensorTopZ = 10.0;

    std::array<Vec3, 8> mics{{
        Vec3{+kSensorHalfSpan, +kSensorHalfSpan, 0.0},         // 0 = A
        Vec3{-kSensorHalfSpan, +kSensorHalfSpan, 0.0},         // 1 = B
        Vec3{-kSensorHalfSpan, -kSensorHalfSpan, 0.0},         // 2 = C
        Vec3{+kSensorHalfSpan, -kSensorHalfSpan, 0.0},         // 3 = D
        Vec3{+kSensorHalfSpan, +kSensorHalfSpan, kSensorTopZ}, // 4 = A2
        Vec3{-kSensorHalfSpan, +kSensorHalfSpan, kSensorTopZ}, // 5 = B2
        Vec3{-kSensorHalfSpan, -kSensorHalfSpan, kSensorTopZ}, // 6 = C2
        Vec3{+kSensorHalfSpan, -kSensorHalfSpan, kSensorTopZ}  // 7 = D2
    }};

    double x = 30.0;
    double y = 50.0;
    double z = 1.0;

    // FIX: brace-init avoids uninitialized reads before first solveTDOA
    Vec3 target  {30.0, 50.0, 1.0};
    Vec3 estimate {30.0, 50.0, 1.0};
    Vec3 estimate2{30.0, 50.0, 1.0};

    bool   hasPair   = false;
    double bestScale = 0.0; // FIX: was misleadingly 1.0 when hasPair=false
    double pairGap   = std::numeric_limits<double>::infinity();

    // IMPROVEMENT: expose threshold so the UI can tune it live
    double dthres = 0.25;

    void syncTargetFromXYZ() { target = Vec3{x, y, z}; }

    // Python-equivalent thresholded axis update from roll/pitch/heave.
    void applyAxisThresholdStep(double roll, double pitch, double heave, double step) {
        if (roll  >  0.2) x += step;
        if (roll  < -0.2) x -= step;
        if (pitch >  0.2) y -= step; // intentional sign flip matching Python
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

        // Python-equivalent loop: build the eight spheres for each scale,
        // intersect the two triplets, then accept the first point-pair match
        // under threshold.
        for (int scaleR = 40; scaleR < 120; ++scaleR) {
            const double scale = static_cast<double>(scaleR) / 100.0;

            const Sphere S1 = roundFromCenterRadius(mics[0], trueD[0] * scale);
            const Sphere S2 = roundFromCenterRadius(mics[1], trueD[1] * scale);
            const Sphere S3 = roundFromCenterRadius(mics[2], trueD[2] * scale);
            const Sphere S4 = roundFromCenterRadius(mics[3], trueD[3] * scale);

            const Sphere S1_2 = roundFromCenterRadius(mics[4], trueD[4] * scale);
            const Sphere S2_2 = roundFromCenterRadius(mics[5], trueD[5] * scale);
            const Sphere S3_2 = roundFromCenterRadius(mics[6], trueD[6] * scale);
            const Sphere S4_2 = roundFromCenterRadius(mics[7], trueD[7] * scale);

            // Python triplets:
            // pt1,pt2 <- point_pair_to_end_points(fast_dual(S1 ^ S2 ^ S3_2))
            // pt3,pt4 <- point_pair_to_end_points(fast_dual(S1_2 ^ S2_2 ^ S3))
            const auto pair123 = pointPairToEndPoints(S1, S2, S3_2);
            const auto pair456 = pointPairToEndPoints(S1_2, S2_2, S3);

            if (!pair123 || !pair456) continue;

            const Vec3 pt1 = pair123->first,  pt2 = pair123->second;
            const Vec3 pt3 = pair456->first,  pt4 = pair456->second;

            // All 6 cross-triplet distances
            const double gaps[6] = {
                norm(pt1 - pt2), norm(pt1 - pt3), norm(pt1 - pt4),
                norm(pt2 - pt3), norm(pt2 - pt4), norm(pt3 - pt4)
            };
            const Vec3 pairA[6] = {pt1, pt1, pt1, pt2, pt2, pt3};
            const Vec3 pairB[6] = {pt2, pt3, pt4, pt3, pt4, pt4};

            for (int k = 0; k < 6; ++k) {
                if (gaps[k] < dthres) {
                    estimate  = pairA[k];
                    estimate2 = pairB[k];
                    hasPair   = true;
                    bestScale = scale;
                    pairGap   = gaps[k];
                    return;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
struct AppState {
    Simulator sim;

    double step  = 3.0;
    float  roll  = 0.0f;
    float  pitch = 0.0f;
    float  heave = 1.0f;
    bool   keyboardControlsEnabled = true;
    std::string lastInput = "none";

    polyscope::PointCloud*   psMics         = nullptr;
    polyscope::PointCloud*   psTarget       = nullptr;
    polyscope::PointCloud*   psEstimate     = nullptr;
    polyscope::PointCloud*   psEstimate2    = nullptr;
    polyscope::CurveNetwork* psEstimateLink = nullptr;
};

static AppState gApp;
static void updateSceneGeometry(); // forward declaration

static void moveTargetX(double delta) { gApp.sim.x += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }
static void moveTargetY(double delta) { gApp.sim.y += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }
static void moveTargetZ(double delta) { gApp.sim.z += delta; gApp.sim.solveTDOA(); updateSceneGeometry(); }

static void updateSceneGeometry() {
    gApp.sim.syncTargetFromXYZ();
    gApp.psTarget->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.target)});

    if (gApp.sim.hasPair) {
        // FIX (C2672): explicit std::vector<glm::vec3> — braced-init-list cannot
        //              deduce the template argument V for updatePointPositions(const V&)
        gApp.psEstimate ->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.estimate )});
        gApp.psEstimate2->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.estimate2)});
        // FIX (C2672): same fix for updateNodePositions
        gApp.psEstimateLink->updateNodePositions(
            std::vector<glm::vec3>{toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)});
        gApp.psEstimate ->setEnabled(true);
        gApp.psEstimate2->setEnabled(true);
        gApp.psEstimateLink->setEnabled(true);
    } else {
        // Keep the estimate markers visible even when no valid pair is found.
        // This makes it easier to verify the scene is rendering and the solver is updating.
        gApp.psEstimate->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.target)});
        gApp.psEstimate2->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.target)});
        gApp.psEstimate ->setEnabled(true);
        gApp.psEstimate2->setEnabled(true);
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
// in one frame does NOT move 2×step.
// ---------------------------------------------------------------------------
static void applyKeyboardMovement() {
    if (!gApp.keyboardControlsEnabled) return;

    double dx = 0.0, dy = 0.0, dz = 0.0;
    const char* src = nullptr;

    // Primary bindings: arrow keys + R/F
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow,  false)) { dx -= gApp.step; src = "LeftArrow"; }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) { dx += gApp.step; src = "RightArrow"; }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow,    false)) { dy += gApp.step; src = "UpArrow"; }
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow,  false)) { dy -= gApp.step; src = "DownArrow"; }
    // R = z-,  F = z+,  matching Python heave sign convention
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) { dz -= gApp.step; src = "R"; }
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) { dz += gApp.step; src = "F"; }

    // Fallback WASD+QE via ImGui key enum
    // FIX (runtime): was polyscope::render::engine->isKeyPressed('a') etc. using
    //     lowercase ASCII (97-122).  GLFW key codes for letters are uppercase ASCII
    //     (65-90 = GLFW_KEY_A..Z), so those calls never fired.
    //     ImGui::IsKeyPressed(ImGuiKey_X) handles the platform mapping correctly.
    // FIX: clamp so simultaneous arrow+WASD on same axis still moves only 1×step
    if (ImGui::IsKeyPressed(ImGuiKey_A, false) && dx == 0.0) { dx -= gApp.step; src = "A"; }
    if (ImGui::IsKeyPressed(ImGuiKey_D, false) && dx == 0.0) { dx += gApp.step; src = "D"; }
    if (ImGui::IsKeyPressed(ImGuiKey_W, false) && dy == 0.0) { dy += gApp.step; src = "W"; }
    if (ImGui::IsKeyPressed(ImGuiKey_S, false) && dy == 0.0) { dy -= gApp.step; src = "S"; }
    if (ImGui::IsKeyPressed(ImGuiKey_Q, false) && dz == 0.0) { dz += gApp.step; src = "Q"; }
    if (ImGui::IsKeyPressed(ImGuiKey_E, false) && dz == 0.0) { dz -= gApp.step; src = "E"; }

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
    ImGui::Text("Fallback keys: A/D=X, W/S=Y, Q/E=Z");
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

        // IMPROVEMENT: show error vector for solve accuracy
        const Vec3 err = gApp.sim.estimate - gApp.sim.target;
        ImGui::Text("err=(%.3f, %.3f, %.3f) |err|=%.3f",
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
    // FIX: absolute radius (false) — relative mode inflated mics in large scenes
    gApp.psMics->setPointRadius(0.18, false);

    // Target
    gApp.psTarget = polyscope::registerPointCloud("Target",
        std::vector<glm::vec3>{toGlm(gApp.sim.target)});
    gApp.psTarget->setPointColor(glm::vec3{0.05f, 0.95f, 0.15f});
    gApp.psTarget->setPointRadius(0.75, false);
    gApp.psTarget->setEnabled(true);

    // Estimate 1
    gApp.psEstimate = polyscope::registerPointCloud("Estimate 1",
        std::vector<glm::vec3>{toGlm(gApp.sim.estimate)});
    gApp.psEstimate->setPointColor(glm::vec3{1.00f, 0.10f, 0.10f});
    gApp.psEstimate->setPointRadius(0.65, false);

    // Estimate 2
    gApp.psEstimate2 = polyscope::registerPointCloud("Estimate 2",
        std::vector<glm::vec3>{toGlm(gApp.sim.estimate2)});
    gApp.psEstimate2->setPointColor(glm::vec3{1.00f, 0.10f, 0.10f});
    gApp.psEstimate2->setPointRadius(0.65, false);

    // Link line between estimates
    std::vector<glm::vec3> linkNodes{toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)};
    std::vector<std::array<size_t, 2>> linkEdges{{{0, 1}}};
    gApp.psEstimateLink = polyscope::registerCurveNetwork("Estimate Link", linkNodes, linkEdges);
    gApp.psEstimateLink->setColor(glm::vec3{0.98f, 0.45f, 0.12f});
    // FIX (C4305): float literal avoids double->float truncation warning
    gApp.psEstimateLink->setRadius(0.002f, true);

    gApp.sim.solveTDOA();
    updateSceneGeometry();
    centerViewOnTarget();

    polyscope::state::userCallback = polyscopeUI;
    polyscope::show();

    return 0;
}

// SWIGX_GA_TDOA_3D_polyscope.cpp
// Polyscope version of the simulator, following the Python and Win32 C++ logic.
//
// Build notes:
// - This file depends on Polyscope and its dependencies (GLM, ImGui via Polyscope).
// - Recommended: build with CMake and link against polyscope target.

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <imgui.h>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/view.h"

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(double s) const {
        return {x * s, y * s, z * s};
    }

    Vec3 operator/(double s) const {
        return {x / s, y / s, z / s};
    }
};

static double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static double norm(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

static glm::vec3 toGlm(const Vec3& v) {
    return glm::vec3{
        static_cast<float>(v.x),
        static_cast<float>(v.y),
        static_cast<float>(v.z)
    };
}

static std::optional<std::pair<Vec3, Vec3>> intersectThreeSpheres(
    const Vec3& c1,
    const Vec3& c2,
    const Vec3& c3,
    double r1,
    double r2,
    double r3
) {
    const Vec3 c12 = c2 - c1;
    const double d = norm(c12);
    if (d < 1e-12) {
        return std::nullopt;
    }

    const Vec3 ex = c12 / d;
    const Vec3 c13 = c3 - c1;
    const double i = dot(ex, c13);
    const Vec3 temp = c13 - ex * i;
    const double tempNorm = norm(temp);
    if (tempNorm < 1e-12) {
        return std::nullopt;
    }

    const Vec3 ey = temp / tempNorm;
    const Vec3 ez = cross(ex, ey);
    const double j = dot(ey, c13);
    if (std::fabs(j) < 1e-12) {
        return std::nullopt;
    }

    const double x = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    const double y = (r1 * r1 - r3 * r3 + i * i + j * j) / (2.0 * j) - (i / j) * x;

    double z2 = r1 * r1 - x * x - y * y;
    if (z2 < -1e-8) {
        return std::nullopt;
    }
    z2 = std::max(0.0, z2);
    const double z = std::sqrt(z2);

    const Vec3 base = c1 + ex * x + ey * y;
    return std::make_optional(std::make_pair(base + ez * z, base - ez * z));
}

struct Simulator {
    std::array<Vec3, 8> mics{
        Vec3{+1.0, +1.0, 0.0},
        Vec3{-1.0, +1.0, 0.0},
        Vec3{-1.0, -1.0, 0.0},
        Vec3{+1.0, -1.0, 0.0},
        Vec3{+1.0, +1.0, 2.0},
        Vec3{-1.0, +1.0, 2.0},
        Vec3{-1.0, -1.0, 2.0},
        Vec3{+1.0, -1.0, 2.0}
    };

    double x = 30.0;
    double y = 50.0;
    double z = 1.0;

    Vec3 target{x, y, z};
    Vec3 estimate{30.0, 50.0, 1.0};
    Vec3 estimate2{30.0, 50.0, 1.0};

    bool hasPair = false;
    double bestScale = 1.0;
    double pairGap = std::numeric_limits<double>::infinity();

    void syncTargetFromXYZ() {
        target = Vec3{x, y, z};
    }

    // Python-equivalent thresholded axis update from roll/pitch/heave.
    void applyAxisThresholdStep(double roll, double pitch, double heave, double step) {
        if (roll > 0.2) {
            x += step;
        }
        if (roll < -0.2) {
            x -= step;
        }

        if (pitch > 0.2) {
            y -= step;
        }
        if (pitch < -0.2) {
            y += step;
        }

        if (heave > 0.2) {
            z -= step;
        }
        if (heave < -0.2) {
            z += step;
        }
    }

    std::array<double, 8> distancesFrom(const Vec3& p) const {
        std::array<double, 8> d{};
        for (size_t i = 0; i < mics.size(); ++i) {
            d[i] = norm(p - mics[i]);
        }
        return d;
    }

    void solveTDOA() {
        syncTargetFromXYZ();
        const std::array<double, 8> trueD = distancesFrom(target);

        hasPair = false;
        bestScale = 1.0;
        pairGap = std::numeric_limits<double>::infinity();

        for (int scaleR = 40; scaleR < 120; ++scaleR) {
            const double scale = static_cast<double>(scaleR) / 100.0;
            std::array<double, 8> scaledD{};
            for (size_t i = 0; i < scaledD.size(); ++i) {
                scaledD[i] = trueD[i] * scale;
            }

            // Python triplets:
            // pt1,pt2 <- intersect(A, B, C2)
            // pt3,pt4 <- intersect(A2, B2, C)
            const auto pair123 = intersectThreeSpheres(
                mics[0], mics[1], mics[6],
                scaledD[0], scaledD[1], scaledD[6]
            );
            const auto pair456 = intersectThreeSpheres(
                mics[4], mics[5], mics[2],
                scaledD[4], scaledD[5], scaledD[2]
            );

            if (!pair123 || !pair456) {
                continue;
            }

            const Vec3 pt1 = pair123->first;
            const Vec3 pt2 = pair123->second;
            const Vec3 pt3 = pair456->first;
            const Vec3 pt4 = pair456->second;

            const double dT12 = norm(pt1 - pt2);
            const double dT13 = norm(pt1 - pt3);
            const double dT14 = norm(pt1 - pt4);
            const double dT23 = norm(pt2 - pt3);
            const double dT24 = norm(pt2 - pt4);
            const double dT34 = norm(pt3 - pt4);

            const double dthres = 0.25;
            auto selectPair = [&](const Vec3& a, const Vec3& b, double gap) {
                estimate = a;
                estimate2 = b;
                hasPair = true;
                bestScale = scale;
                pairGap = gap;
            };

            if (dT12 < dthres) {
                selectPair(pt1, pt2, dT12);
                break;
            }
            if (dT13 < dthres) {
                selectPair(pt1, pt3, dT13);
                break;
            }
            if (dT14 < dthres) {
                selectPair(pt1, pt4, dT14);
                break;
            }
            if (dT23 < dthres) {
                selectPair(pt2, pt3, dT23);
                break;
            }
            if (dT24 < dthres) {
                selectPair(pt2, pt4, dT24);
                break;
            }
            if (dT34 < dthres) {
                selectPair(pt3, pt4, dT34);
                break;
            }
        }
    }
};

struct AppState {
    Simulator sim;

    double step = 3.0;
    float roll = 0.0f;
    float pitch = 0.0f;
    float heave = 1.0f;
    bool keyboardControlsEnabled = true;
    std::string lastInput = "none";

    polyscope::PointCloud* psMics = nullptr;
    polyscope::PointCloud* psTarget = nullptr;
    polyscope::PointCloud* psEstimate = nullptr;
    polyscope::PointCloud* psEstimate2 = nullptr;
    polyscope::CurveNetwork* psEstimateLink = nullptr;
};

static AppState gApp;
static void updateSceneGeometry();

static void moveTargetX(double delta) {
    gApp.sim.x += delta;
    gApp.sim.solveTDOA();
    updateSceneGeometry();
}

static void moveTargetY(double delta) {
    gApp.sim.y += delta;
    gApp.sim.solveTDOA();
    updateSceneGeometry();
}

static void moveTargetZ(double delta) {
    gApp.sim.z += delta;
    gApp.sim.solveTDOA();
    updateSceneGeometry();
}

static void updateSceneGeometry() {
    gApp.sim.syncTargetFromXYZ();

    gApp.psTarget->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.target)});

    if (gApp.sim.hasPair) {
        gApp.psEstimate->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.estimate)});
        gApp.psEstimate2->updatePointPositions(std::vector<glm::vec3>{toGlm(gApp.sim.estimate2)});

        std::vector<glm::vec3> linkNodes{toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)};
        gApp.psEstimateLink->updateNodePositions(linkNodes);

        gApp.psEstimate->setEnabled(true);
        gApp.psEstimate2->setEnabled(true);
        gApp.psEstimateLink->setEnabled(true);
    } else {
        gApp.psEstimate->setEnabled(false);
        gApp.psEstimate2->setEnabled(false);
        gApp.psEstimateLink->setEnabled(false);
    }

    polyscope::requestRedraw();
}

static void centerViewOnTarget() {
    const glm::vec3 targetPos = toGlm(gApp.sim.target);
    const glm::vec3 cameraPos = targetPos + glm::vec3{0.0f, 0.0f, 120.0f};
    polyscope::view::lookAt(cameraPos, targetPos);
}

static void applyKeyboardMovement() {
    if (!gApp.keyboardControlsEnabled) {
        return;
    }

    bool moved = false;
    const char* movementSource = nullptr;

    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false)) {
        gApp.sim.x -= gApp.step;
        moved = true;
        movementSource = "LeftArrow";
    }
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
        gApp.sim.x += gApp.step;
        moved = true;
        movementSource = "RightArrow";
    }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, false)) {
        gApp.sim.y += gApp.step;
        moved = true;
        movementSource = "UpArrow";
    }
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, false)) {
        gApp.sim.y -= gApp.step;
        moved = true;
        movementSource = "DownArrow";
    }

    // Keep same sign convention as Python heave branch and current C++ version.
    if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
        gApp.sim.z -= gApp.step;
        moved = true;
        movementSource = "R";
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        gApp.sim.z += gApp.step;
        moved = true;
        movementSource = "F";
    }

    // Reliable fallback: Polyscope's backend key helper supports a-z / 0-9.
    if (polyscope::render::engine->isKeyPressed('a')) {
        gApp.sim.x -= gApp.step;
        moved = true;
        movementSource = "a";
    }
    if (polyscope::render::engine->isKeyPressed('d')) {
        gApp.sim.x += gApp.step;
        moved = true;
        movementSource = "d";
    }
    if (polyscope::render::engine->isKeyPressed('w')) {
        gApp.sim.y += gApp.step;
        moved = true;
        movementSource = "w";
    }
    if (polyscope::render::engine->isKeyPressed('s')) {
        gApp.sim.y -= gApp.step;
        moved = true;
        movementSource = "s";
    }
    if (polyscope::render::engine->isKeyPressed('q')) {
        gApp.sim.z += gApp.step;
        moved = true;
        movementSource = "q";
    }
    if (polyscope::render::engine->isKeyPressed('e')) {
        gApp.sim.z -= gApp.step;
        moved = true;
        movementSource = "e";
    }

    if (moved) {
        if (movementSource != nullptr) {
            gApp.lastInput = movementSource;
        }
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }
}

static void polyscopeUI() {
    ImGui::PushItemWidth(220.0f);

    ImGui::TextUnformatted("SWIGX GA TDOA 3D (Polyscope)");
    ImGui::Separator();

    ImGui::TextUnformatted("Controls");
    ImGui::SliderFloat("roll", &gApp.roll, -1.0f, 1.0f);
    ImGui::SliderFloat("pitch", &gApp.pitch, -1.0f, 1.0f);
    ImGui::SliderFloat("heave", &gApp.heave, -1.0f, 1.0f);
    ImGui::Checkbox("Keyboard controls", &gApp.keyboardControlsEnabled);

    double targetX = gApp.sim.x;
    double targetY = gApp.sim.y;
    double targetZ = gApp.sim.z;
    bool targetChanged = false;

    targetChanged |= ImGui::InputDouble("target x", &targetX, 0.1, 1.0, "%.2f");
    targetChanged |= ImGui::InputDouble("target y", &targetY, 0.1, 1.0, "%.2f");
    targetChanged |= ImGui::InputDouble("target z", &targetZ, 0.1, 1.0, "%.2f");

    if (targetChanged) {
        gApp.sim.x = targetX;
        gApp.sim.y = targetY;
        gApp.sim.z = targetZ;
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }

    ImGui::TextUnformatted("Nudge target");
    if (ImGui::Button("X -")) moveTargetX(-gApp.step);
    ImGui::SameLine();
    if (ImGui::Button("X +")) moveTargetX(gApp.step);
    ImGui::SameLine();
    if (ImGui::Button("Y -")) moveTargetY(-gApp.step);
    ImGui::SameLine();
    if (ImGui::Button("Y +")) moveTargetY(gApp.step);
    if (ImGui::Button("Z -")) moveTargetZ(-gApp.step);
    ImGui::SameLine();
    if (ImGui::Button("Z +")) moveTargetZ(gApp.step);

    if (ImGui::Button("Center View")) {
        centerViewOnTarget();
    }

    if (ImGui::Button("Apply axis threshold step")) {
        gApp.sim.applyAxisThresholdStep(gApp.roll, gApp.pitch, gApp.heave, gApp.step);
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }

    ImGui::SameLine();
    if (ImGui::Button("Solve")) {
        gApp.sim.solveTDOA();
        updateSceneGeometry();
    }

    ImGui::InputDouble("step", &gApp.step, 0.1, 1.0, "%.2f");

    ImGui::Separator();
    ImGui::Text("Arrow keys move X/Y, R/F move Z");
    ImGui::Text("Fallback keys: A/D X, W/S Y, Q/E Z");
    ImGui::Text("Last input: %s", gApp.lastInput.c_str());
    ImGui::Text("Target: (%.2f, %.2f, %.2f)", gApp.sim.target.x, gApp.sim.target.y, gApp.sim.target.z);
    ImGui::Text("scale=%.2f", gApp.sim.bestScale);

    if (gApp.sim.hasPair) {
        ImGui::Text("Estimate1: (%.2f, %.2f, %.2f)", gApp.sim.estimate.x, gApp.sim.estimate.y, gApp.sim.estimate.z);
        ImGui::Text("Estimate2: (%.2f, %.2f, %.2f)", gApp.sim.estimate2.x, gApp.sim.estimate2.y, gApp.sim.estimate2.z);
        ImGui::Text("pair_gap=%.6f", gApp.sim.pairGap);
    } else {
        ImGui::TextUnformatted("No valid pair under threshold");
    }

    ImGui::PopItemWidth();

    applyKeyboardMovement();
}

int main() {
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::init();

    std::vector<glm::vec3> micPoints;
    micPoints.reserve(gApp.sim.mics.size());
    for (const Vec3& m : gApp.sim.mics) {
        micPoints.push_back(toGlm(m));
    }

    gApp.psMics = polyscope::registerPointCloud("Mics", micPoints);
    gApp.psMics->setPointColor(glm::vec3{0.12f, 0.45f, 0.95f});
    gApp.psMics->setPointRadius(0.01, true);

    gApp.psTarget = polyscope::registerPointCloud("Target", std::vector<glm::vec3>{toGlm(gApp.sim.target)});
    gApp.psTarget->setPointColor(glm::vec3{0.15f, 0.8f, 0.25f});
    gApp.psTarget->setPointRadius(0.35, false);
    gApp.psTarget->setEnabled(true);

    gApp.psEstimate = polyscope::registerPointCloud("Estimate 1", std::vector<glm::vec3>{toGlm(gApp.sim.estimate)});
    gApp.psEstimate->setPointColor(glm::vec3{0.85f, 0.15f, 0.15f});
    gApp.psEstimate->setPointRadius(0.25, false);

    gApp.psEstimate2 = polyscope::registerPointCloud("Estimate 2", std::vector<glm::vec3>{toGlm(gApp.sim.estimate2)});
    gApp.psEstimate2->setPointColor(glm::vec3{0.95f, 0.55f, 0.15f});
    gApp.psEstimate2->setPointRadius(0.25, false);

    std::vector<glm::vec3> linkNodes{toGlm(gApp.sim.estimate), toGlm(gApp.sim.estimate2)};
    std::vector<std::array<size_t, 2>> linkEdges{{{0, 1}}};
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

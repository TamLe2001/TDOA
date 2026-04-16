// SWIGX_GA_TDOA_3D_v01.cpp
// Win32 popup version (no terminal) of the C++ simulator.
//
// Build (LLVM/clang++ on MSVC target):
//   "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -O2 -o SWIGX_GA_TDOA_3D_v01.exe SWIGX_GA_TDOA_3D_v01.cpp -lgdi32 -luser32 -Xlinker /SUBSYSTEM:WINDOWS

#define NOMINMAX
#include <windows.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

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

static double norm(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

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
    double pairGap = 0.0;

    void syncTargetFromXYZ() {
        target = Vec3{x, y, z};
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

        bestScale = 1.0;
        pairGap = std::numeric_limits<double>::infinity();
        hasPair = false;

        for (int scaleR = 40; scaleR < 120; ++scaleR) {
            const double scale = static_cast<double>(scaleR) / 100.0;
            std::array<double, 8> scaledD{};
            for (size_t i = 0; i < scaledD.size(); ++i) {
                scaledD[i] = trueD[i] * scale;
            }

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
};

static AppState gApp;

static int mapCoord(double value, double minV, double maxV, int pxMin, int pxMax) {
    const double t = (value - minV) / (maxV - minV);
    const double clamped = std::max(0.0, std::min(1.0, t));
    return static_cast<int>(pxMin + clamped * (pxMax - pxMin));
}

static void drawPoint(HDC hdc, int x, int y, int radius, COLORREF color) {
    HPEN pen = CreatePen(PS_SOLID, 1, color);
    HBRUSH brush = CreateSolidBrush(color);
    HGDIOBJ oldPen = SelectObject(hdc, pen);
    HGDIOBJ oldBrush = SelectObject(hdc, brush);

    Ellipse(hdc, x - radius, y - radius, x + radius, y + radius);

    SelectObject(hdc, oldPen);
    SelectObject(hdc, oldBrush);
    DeleteObject(brush);
    DeleteObject(pen);
}

static void drawCross(HDC hdc, int x, int y, int radius, COLORREF color) {
    HPEN pen = CreatePen(PS_SOLID, 2, color);
    HGDIOBJ oldPen = SelectObject(hdc, pen);

    MoveToEx(hdc, x - radius, y - radius, nullptr);
    LineTo(hdc, x + radius, y + radius);
    MoveToEx(hdc, x - radius, y + radius, nullptr);
    LineTo(hdc, x + radius, y - radius);

    SelectObject(hdc, oldPen);
    DeleteObject(pen);
}

static void renderScene(HWND hwnd, HDC hdc) {
    RECT client{};
    GetClientRect(hwnd, &client);

    HBRUSH bg = CreateSolidBrush(RGB(245, 248, 252));
    FillRect(hdc, &client, bg);
    DeleteObject(bg);

    const int margin = 24;
    RECT plotRect{margin, margin + 30, client.right - margin, client.bottom - margin};

    HPEN borderPen = CreatePen(PS_SOLID, 1, RGB(90, 100, 120));
    HGDIOBJ oldPen = SelectObject(hdc, borderPen);
    HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
    Rectangle(hdc, plotRect.left, plotRect.top, plotRect.right, plotRect.bottom);
    SelectObject(hdc, oldBrush);
    SelectObject(hdc, oldPen);
    DeleteObject(borderPen);

    const double worldMin = -70.0;
    const double worldMax = 70.0;

    const int zeroX = mapCoord(0.0, worldMin, worldMax, plotRect.left, plotRect.right);
    const int zeroY = mapCoord(0.0, worldMin, worldMax, plotRect.bottom, plotRect.top);

    HPEN axisPen = CreatePen(PS_DOT, 1, RGB(170, 170, 170));
    oldPen = SelectObject(hdc, axisPen);
    MoveToEx(hdc, plotRect.left, zeroY, nullptr);
    LineTo(hdc, plotRect.right, zeroY);
    MoveToEx(hdc, zeroX, plotRect.top, nullptr);
    LineTo(hdc, zeroX, plotRect.bottom);
    SelectObject(hdc, oldPen);
    DeleteObject(axisPen);

    for (const Vec3& m : gApp.sim.mics) {
        const int sx = mapCoord(m.x, worldMin, worldMax, plotRect.left, plotRect.right);
        const int sy = mapCoord(m.y, worldMin, worldMax, plotRect.bottom, plotRect.top);
        drawCross(hdc, sx, sy, 5, RGB(30, 100, 220));
    }

    const int tx = mapCoord(gApp.sim.target.x, worldMin, worldMax, plotRect.left, plotRect.right);
    const int ty = mapCoord(gApp.sim.target.y, worldMin, worldMax, plotRect.bottom, plotRect.top);
    drawPoint(hdc, tx, ty, 6, RGB(20, 160, 60));

    if (gApp.sim.hasPair) {
        const int ex = mapCoord(gApp.sim.estimate.x, worldMin, worldMax, plotRect.left, plotRect.right);
        const int ey = mapCoord(gApp.sim.estimate.y, worldMin, worldMax, plotRect.bottom, plotRect.top);
        drawPoint(hdc, ex, ey, 5, RGB(210, 40, 40));

        const int ex2 = mapCoord(gApp.sim.estimate2.x, worldMin, worldMax, plotRect.left, plotRect.right);
        const int ey2 = mapCoord(gApp.sim.estimate2.y, worldMin, worldMax, plotRect.bottom, plotRect.top);
        drawPoint(hdc, ex2, ey2, 4, RGB(250, 130, 40));
    }

    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(30, 30, 40));

    std::wstringstream line1;
    line1 << L"Arrow keys: move X/Y   R/F: Z+/Z-   Esc: quit";
    TextOutW(hdc, margin, 6, line1.str().c_str(), static_cast<int>(line1.str().size()));

    std::wstringstream line2;
    line2 << std::fixed << std::setprecision(2)
          << L"Target(" << gApp.sim.target.x << L", " << gApp.sim.target.y << L", " << gApp.sim.target.z << L")"
          << L"  scale=" << std::setprecision(2) << gApp.sim.bestScale;
    if (gApp.sim.hasPair) {
        line2 << L"  Estimate(" << gApp.sim.estimate.x << L", " << gApp.sim.estimate.y << L", " << gApp.sim.estimate.z << L")"
              << L"  pair_gap=" << std::setprecision(4) << gApp.sim.pairGap;
    } else {
        line2 << L"  no valid pair under threshold";
    }
    TextOutW(hdc, margin, client.bottom - 20, line2.str().c_str(), static_cast<int>(line2.str().size()));
}

static void moveAndResolve(WPARAM key) {
    bool moved = false;

    if (key == VK_UP) {
        gApp.sim.y += gApp.step;
        moved = true;
    } else if (key == VK_DOWN) {
        gApp.sim.y -= gApp.step;
        moved = true;
    } else if (key == VK_LEFT) {
        gApp.sim.x -= gApp.step;
        moved = true;
    } else if (key == VK_RIGHT) {
        gApp.sim.x += gApp.step;
        moved = true;
    } else if (key == 'R') {
        gApp.sim.z -= gApp.step;
        moved = true;
    } else if (key == 'F') {
        gApp.sim.z += gApp.step;
        moved = true;
    }

    if (moved) {
        gApp.sim.solveTDOA();
    }
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    (void)lParam;

    switch (msg) {
        case WM_CREATE:
            gApp.sim.solveTDOA();
            return 0;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
                return 0;
            }
            moveAndResolve(wParam);
            InvalidateRect(hwnd, nullptr, TRUE);
            return 0;

        case WM_PAINT: {
            PAINTSTRUCT ps{};
            HDC hdc = BeginPaint(hwnd, &ps);
            renderScene(hwnd, hdc);
            EndPaint(hwnd, &ps);
            return 0;
        }

        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;

        default:
            return DefWindowProcW(hwnd, msg, wParam, lParam);
    }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    const wchar_t kClassName[] = L"SWIGX_GA_TDOA_3D_WindowClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = kClassName;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);

    if (!RegisterClassW(&wc)) {
        return 1;
    }

    HWND hwnd = CreateWindowExW(
        0,
        kClassName,
        L"SWIGX GA TDOA 3D (C++ Popup)",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1000,
        700,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (hwnd == nullptr) {
        return 1;
    }

    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg{};
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    return 0;
}

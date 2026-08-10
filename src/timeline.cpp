#include "timeline.h"
#include "shader_manager.h"
#include "math_util.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

using json = nlohmann::json;

const char* interp_name(Interp i) {
    switch (i) {
        case Interp::Step:   return "step";
        case Interp::Linear: return "linear";
        case Interp::Smooth: return "smooth";
        case Interp::Bezier: return "bezier";
    }
    return "smooth";
}

static Interp interp_from_name(const std::string& s) {
    if (s == "step")   return Interp::Step;
    if (s == "linear") return Interp::Linear;
    if (s == "bezier") return Interp::Bezier;
    return Interp::Smooth;
}

// ---- Track ----

float Track::eval(float t) const {
    if (keys.empty()) return 0.0f;
    if (t <= keys.front().t) return keys.front().v;
    if (t >= keys.back().t)  return keys.back().v;

    // First key strictly after t; the segment is [i-1, i].
    size_t i = 1;
    while (i < keys.size() && keys[i].t <= t) i++;
    const Key& a = keys[i - 1];
    const Key& b = keys[i];

    float dt = b.t - a.t;
    if (dt <= 1e-9f) return b.v;
    float u = (t - a.t) / dt;

    // The left key owns the segment: its mode decides how we get to the next.
    switch (a.interp) {
        case Interp::Step:   return a.v;
        case Interp::Linear: return a.v + (b.v - a.v) * u;
        case Interp::Bezier: return hermite(a.v, b.v, a.out_tan, b.in_tan, dt, u);
        case Interp::Smooth: {
            // Endpoint segments duplicate the outer key so the curve doesn't
            // overshoot off the ends of the track.
            float p0 = (i >= 2)              ? keys[i - 2].v : a.v;
            float p3 = (i + 1 < keys.size()) ? keys[i + 1].v : b.v;
            return catmull_rom(p0, a.v, b.v, p3, u);
        }
    }
    return a.v;
}

int Track::insert(float t, float v, float merge_window, Interp interp) {
    for (size_t i = 0; i < keys.size(); i++) {
        if (std::fabs(keys[i].t - t) <= merge_window) {
            keys[i].v = v;
            return (int)i;
        }
    }
    Key k;
    k.t = t;
    k.v = v;
    k.interp = interp;
    auto it = std::upper_bound(keys.begin(), keys.end(), t,
                               [](float x, const Key& e) { return x < e.t; });
    it = keys.insert(it, k);
    return (int)(it - keys.begin());
}

void Track::erase_at(int index) {
    if (index >= 0 && index < (int)keys.size()) keys.erase(keys.begin() + index);
}

// ---- Timeline queries ----

const Track* Timeline::find(const std::string& shader, const std::string& param, int comp) const {
    for (const auto& t : tracks) {
        if (t.component == comp && t.param == param && t.shader == shader) return &t;
    }
    return nullptr;
}

Track* Timeline::find(const std::string& shader, const std::string& param, int comp) {
    return const_cast<Track*>(const_cast<const Timeline*>(this)->find(shader, param, comp));
}

bool Timeline::has_track(const std::string& shader, const std::string& param) const {
    for (const auto& t : tracks) {
        if (t.param == param && t.shader == shader) return true;
    }
    return false;
}

// ---- Timeline mutation ----

Track& Timeline::get_or_create(const std::string& shader, const ShaderParam& p, int comp) {
    if (Track* t = find(shader, p.name, comp)) return *t;
    Track t;
    t.shader = shader;
    t.param = p.name;
    t.component = comp;
    tracks.push_back(std::move(t));
    revision++;
    return tracks.back();
}

// Int and Enum reach the shader as floats truncated by int() — a lerp between
// two enum indices spends most of the segment on a value that isn't either
// one. Default those tracks to Step.
static Interp default_interp_for(const ShaderParam& p) {
    return (p.type == ShaderParam::Int || p.type == ShaderParam::Enum) ? Interp::Step
                                                                      : Interp::Smooth;
}

void Timeline::key_param(float t, const std::string& shader, const ShaderParam& p) {
    Interp mode = default_interp_for(p);
    float window = frame_time() * 0.5f;
    for (int c = 0; c < p.component_count; c++) {
        Track& tr = get_or_create(shader, p, c);
        tr.insert(t, p.current_val[c], window, mode);
    }
    revision++;
}

void Timeline::remove_param(const std::string& shader, const std::string& param) {
    auto it = std::remove_if(tracks.begin(), tracks.end(), [&](const Track& t) {
        return t.shader == shader && t.param == param;
    });
    if (it != tracks.end()) {
        tracks.erase(it, tracks.end());
        revision++;
    }
}

void Timeline::key_camera(float t, const Camera& cam) {
    CameraKey k;
    k.t = t;
    std::memcpy(k.target, cam.target, sizeof(float) * 3);

    float d[3];
    v3::sub(cam.pos, cam.target, d);
    float dist = v3::length(d);
    if (dist > 0.0f) {
        // Divide by the distance we already have rather than calling
        // v3::normalize, whose 1e-8 guard would zero the direction on a deep
        // zoom. Distances far below that are the normal case here — an
        // arbitrary floor would quietly truncate the dive.
        k.dir[0] = d[0] / dist;
        k.dir[1] = d[1] / dist;
        k.dir[2] = d[2] / dist;
    } else {
        // Truly degenerate (pos == target): the direction is undefined, so
        // pick one rather than storing a zero vector that would slerp to junk.
        k.dir[0] = 0; k.dir[1] = 0; k.dir[2] = 1;
        dist = 1e-20f;
    }
    k.log_dist = std::log(dist);
    k.fov = cam.fov;

    float window = frame_time() * 0.5f;
    for (size_t i = 0; i < cam_keys.size(); i++) {
        if (std::fabs(cam_keys[i].t - t) <= window) {
            k.interp = cam_keys[i].interp;
            cam_keys[i] = k;
            revision++;
            return;
        }
    }
    auto it = std::upper_bound(cam_keys.begin(), cam_keys.end(), t,
                               [](float x, const CameraKey& e) { return x < e.t; });
    cam_keys.insert(it, k);
    revision++;
}

void Timeline::remove_camera_key(int index) {
    if (index >= 0 && index < (int)cam_keys.size()) {
        cam_keys.erase(cam_keys.begin() + index);
        revision++;
    }
}

void Timeline::clear() {
    tracks.clear();
    cam_keys.clear();
    revision++;
}

// ---- Evaluation ----

void Timeline::eval_camera(float t, Camera& cam) const {
    if (cam_keys.empty()) return;

    const CameraKey* a = &cam_keys.front();
    const CameraKey* b = &cam_keys.front();
    float u = 0.0f;
    size_t i = 0;

    if (t >= cam_keys.back().t) {
        a = b = &cam_keys.back();
    } else if (t > cam_keys.front().t) {
        i = 1;
        while (i < cam_keys.size() && cam_keys[i].t <= t) i++;
        a = &cam_keys[i - 1];
        b = &cam_keys[i];
        float dt = b->t - a->t;
        u = dt > 1e-9f ? (t - a->t) / dt : 0.0f;
        if (a->interp == Interp::Step) u = 0.0f;
    }

    float target[3], dir[3], log_dist, fov;

    if (a == b || a->interp == Interp::Step) {
        std::memcpy(target, a->target, sizeof(float) * 3);
        std::memcpy(dir, a->dir, sizeof(float) * 3);
        log_dist = a->log_dist;
        fov = a->fov;
    } else if (a->interp == Interp::Linear) {
        v3::lerp(a->target, b->target, u, target);
        v3::slerp(a->dir, b->dir, u, dir);
        log_dist = a->log_dist + (b->log_dist - a->log_dist) * u;
        fov = a->fov + (b->fov - a->fov) * u;
    } else {
        // Catmull-Rom the target and scalars, slerp the direction. Direction
        // stays a two-key slerp on purpose: a spline through unit vectors
        // leaves the sphere and needs renormalizing, which reintroduces the
        // rate non-uniformity the spline was for.
        const CameraKey& k0 = (i >= 2)               ? cam_keys[i - 2] : *a;
        const CameraKey& k3 = (i + 1 < cam_keys.size()) ? cam_keys[i + 1] : *b;
        for (int c = 0; c < 3; c++) {
            target[c] = catmull_rom(k0.target[c], a->target[c], b->target[c], k3.target[c], u);
        }
        v3::slerp(a->dir, b->dir, u, dir);
        // log-space so a zoom covers a constant *ratio* per second. Linear
        // world-space distance across orders of magnitude is a lunge then a
        // crawl — the whole point of fractal flythroughs is the constant rate.
        log_dist = catmull_rom(k0.log_dist, a->log_dist, b->log_dist, k3.log_dist, u);
        fov      = catmull_rom(k0.fov, a->fov, b->fov, k3.fov, u);
    }

    std::memcpy(cam.target, target, sizeof(float) * 3);
    cam.fov = fov;
    v3::mad(cam.target, dir, std::exp(log_dist), cam.pos);
}

void Timeline::apply(float t, ShaderManager& shaders, Camera& cam) const {
    if (!enabled) return;   // keys are kept, they just stop driving anything
    for (const auto& tr : tracks) {
        if (!tr.enabled || tr.keys.empty()) continue;
        if (tr.shader == editing_shader && tr.param == editing_param) continue;
        auto& params = shaders.get_params_mut(tr.shader);
        for (auto& p : params) {
            if (p.name != tr.param) continue;
            if (tr.component >= p.component_count) break;  // param changed shape on reload
            float v = tr.eval(t);
            if (p.type == ShaderParam::Int || p.type == ShaderParam::Enum) {
                v = std::round(v);
            }
            p.current_val[tr.component] = v;
            break;
        }
    }
    if (camera_driven()) eval_camera(t, cam);
}

// ---- Serialization ----

void Timeline::to_json(json& j) const {
    j = json::object();
    j["duration"] = duration;
    j["fps"] = fps;
    j["loop"] = loop;
    j["drive_camera"] = drive_camera;

    json cam_j = json::array();
    for (const auto& k : cam_keys) {
        cam_j.push_back({
            {"t", k.t},
            {"target", {k.target[0], k.target[1], k.target[2]}},
            {"dir", {k.dir[0], k.dir[1], k.dir[2]}},
            {"log_dist", k.log_dist},
            {"fov", k.fov},
            {"interp", interp_name(k.interp)},
        });
    }
    j["camera"] = cam_j;

    json tracks_j = json::array();
    for (const auto& tr : tracks) {
        json keys_j = json::array();
        for (const auto& k : tr.keys) {
            json kj = {{"t", k.t}, {"v", k.v}, {"interp", interp_name(k.interp)}};
            if (k.interp == Interp::Bezier) {
                kj["in_tan"] = k.in_tan;
                kj["out_tan"] = k.out_tan;
            }
            keys_j.push_back(kj);
        }
        tracks_j.push_back({
            {"shader", tr.shader},
            {"param", tr.param},
            {"component", tr.component},
            {"enabled", tr.enabled},
            {"keys", keys_j},
        });
    }
    j["tracks"] = tracks_j;
}

// clear()-first, so applying a block is idempotent: whatever the document
// omits is left empty rather than inherited from what was loaded before it.
// `enabled` is deliberately absent — that is a UI preference, not content.
void Timeline::from_json(const json& j) {
    clear();
    if (!j.is_object()) return;
    // Neither array means this is not an animation document — e.g. a preset
    // written while /timeline/enabled was still a normal field, which lands
    // here as {"enabled": true}. Treat it as "no animation" rather than let
    // its absent keys reset duration and fps to defaults.
    if (!j.contains("camera") && !j.contains("tracks")) return;

    duration     = j.value("duration", 10.0f);
    fps          = j.value("fps", 30.0f);
    loop         = j.value("loop", true);
    drive_camera = j.value("drive_camera", true);

    if (j.contains("camera") && j["camera"].is_array()) {
        for (const auto& kj : j["camera"]) {
            CameraKey k;
            k.t = kj.value("t", 0.0f);
            if (kj.contains("target") && kj["target"].size() >= 3) {
                for (int c = 0; c < 3; c++) k.target[c] = kj["target"][c].get<float>();
            }
            if (kj.contains("dir") && kj["dir"].size() >= 3) {
                for (int c = 0; c < 3; c++) k.dir[c] = kj["dir"][c].get<float>();
            }
            v3::normalize(k.dir, k.dir);
            k.log_dist = kj.value("log_dist", 0.0f);
            k.fov      = kj.value("fov", 1.2f);
            k.interp   = interp_from_name(kj.value("interp", std::string("smooth")));
            cam_keys.push_back(k);
        }
        std::sort(cam_keys.begin(), cam_keys.end(),
                  [](const CameraKey& a, const CameraKey& b) { return a.t < b.t; });
    }

    if (j.contains("tracks") && j["tracks"].is_array()) {
        for (const auto& tj : j["tracks"]) {
            Track tr;
            tr.shader    = tj.value("shader", std::string());
            tr.param     = tj.value("param", std::string());
            tr.component = tj.value("component", 0);
            tr.enabled   = tj.value("enabled", true);
            if (tr.shader.empty() || tr.param.empty()) continue;
            if (tj.contains("keys") && tj["keys"].is_array()) {
                for (const auto& kj : tj["keys"]) {
                    Key k;
                    k.t       = kj.value("t", 0.0f);
                    k.v       = kj.value("v", 0.0f);
                    k.interp  = interp_from_name(kj.value("interp", std::string("smooth")));
                    k.in_tan  = kj.value("in_tan", 0.0f);
                    k.out_tan = kj.value("out_tan", 0.0f);
                    tr.keys.push_back(k);
                }
                std::sort(tr.keys.begin(), tr.keys.end(),
                          [](const Key& a, const Key& b) { return a.t < b.t; });
            }
            tracks.push_back(std::move(tr));
        }
    }

    revision++;
}

bool Timeline::save(const std::string& path) const {
    json j;
    to_json(j);

    // Atomic write, same as StateSerializer: a crash mid-save must not leave a
    // truncated animation where a good one used to be.
    std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out) return false;
        out << j.dump(2);
        if (!out) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool Timeline::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;

    json j;
    try {
        in >> j;
    } catch (const std::exception&) {
        return false;
    }
    from_json(j);
    return true;
}

#include "traversability_grid/elevation_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>

namespace traversability_grid
{
namespace
{
constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int32_t kNoKey = std::numeric_limits<int32_t>::min();
// 4 floats; GCC/Clang vector extension, baseline SIMD min/max on x86 (SSE) and ARM (NEON)
typedef float Vec4 __attribute__((vector_size(16)));

// Offsets (di, dj) with di^2 + dj^2 <= k^2, k = round(r / res).
std::vector<std::pair<int, int>> disk(double r, double res)
{
  const int k = static_cast<int>(std::round(r / res));
  std::vector<std::pair<int, int>> o;
  for (int a = -k; a <= k; ++a) {
    for (int b = -k; b <= k; ++b) {
      if (a * a + b * b <= k * k) {o.emplace_back(a, b);}
    }
  }
  return o;
}

// Batcher odd-even merge sort for the next power of two, pruned to n inputs: the padding
// slots hold +inf, so comparators touching them never move anything.
// Only outputs [0, keep) are needed: comparators that cannot reach them are dropped.
std::vector<std::pair<int, int>> sortNet(int n, int keep)
{
  int N = 1;
  while (N < n) {N *= 2;}
  std::vector<std::pair<int, int>> net;
  for (int p = 1; p < N; p *= 2) {
    for (int k = p; k >= 1; k /= 2) {
      for (int j = k % p; j <= N - 1 - k; j += 2 * k) {
        for (int i = 0; i <= std::min(k - 1, N - j - k - 1); ++i) {
          if ((i + j) / (2 * p) == (i + j + k) / (2 * p) && i + j + k < n) {net.emplace_back(i + j, i + j + k);}
        }
      }
    }
  }
  std::vector<char> need(n, 0);
  std::fill(need.begin(), need.begin() + keep, 1);
  std::vector<std::pair<int, int>> kept;
  for (auto it = net.rbegin(); it != net.rend(); ++it) {
    if (need[it->first] || need[it->second]) {
      kept.push_back(*it);
      need[it->first] = need[it->second] = 1;
    }
  }
  return {kept.rbegin(), kept.rend()};
}

double median(std::vector<double> & v)
{
  const size_t h = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + h, v.end());
  const double hi = v[h];
  if (v.size() % 2) {return hi;}
  const double lo = *std::max_element(v.begin(), v.begin() + h);
  return 0.5 * (lo + hi);
}

// Solve the 3x3 normal equations of z = a + b*dx + c*dy. Returns false if singular.
bool planeFit(const std::vector<double> & dx, const std::vector<double> & dy,
              const std::vector<double> & d, const std::vector<char> & m, double c[3])
{
  double A[3][4] = {};
  for (size_t k = 0; k < d.size(); ++k) {
    if (!m[k]) {continue;}
    const double r[3] = {1.0, dx[k], dy[k]};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {A[i][j] += r[i] * r[j];}
      A[i][3] += r[i] * d[k];
    }
  }
  for (int col = 0; col < 3; ++col) {
    int piv = col;
    for (int r = col + 1; r < 3; ++r) {
      if (std::abs(A[r][col]) > std::abs(A[piv][col])) {piv = r;}
    }
    if (std::abs(A[piv][col]) < 1e-12) {return false;}
    std::swap(A[col], A[piv]);
    for (int r = 0; r < 3; ++r) {
      if (r == col) {continue;}
      const double f = A[r][col] / A[col][col];
      for (int j = col; j < 4; ++j) {A[r][j] -= f * A[col][j];}
    }
  }
  for (int i = 0; i < 3; ++i) {c[i] = A[i][3] / A[i][i];}
  return true;
}
}  // namespace

ElevationGrid::ElevationGrid(const Params & p, double ox, double oy)
: p_(p), ox_(ox), oy_(oy), N_(1), net_(sortNet(p.k_med * p.k_med, p.k_med * p.k_med / 2 + 1))
{
  while (N_ < p.ring_cells) {N_ *= 2;}   // power of two: slot() is a bit mask
  const size_t n = static_cast<size_t>(N_) * N_;
  cells_.assign(n, Cell{kNaNf, std::numeric_limits<float>::infinity(), kNaNf, -1e9, -1e9, kNoKey, kNoKey});
  stamp_.assign(n, 0);
  obj_stamp_.assign(n, 0);
  best_.assign(n, 0);
  zlo_.assign(n, 0.0);
  zhi_.assign(n, 0.0);
  rlo_.assign(n, 0.0);
}

long ElevationGrid::gi(double x) const {return static_cast<long>(std::floor((x - ox_) / p_.res));}
long ElevationGrid::gj(double y) const {return static_cast<long>(std::floor((y - oy_) / p_.res));}

size_t ElevationGrid::slot(long i, long j) const
{
  const long a = i & (N_ - 1);   // two's complement: also right for negative indices
  const long b = j & (N_ - 1);
  return static_cast<size_t>(a) * N_ + static_cast<size_t>(b);
}

ElevationGrid::Cell & ElevationGrid::touch(long i, long j)
{
  Cell & c = cells_[slot(i, j)];
  if (c.ki != i || c.kj != j) {
    c = Cell{kNaNf, std::numeric_limits<float>::infinity(), kNaNf, -1e9, -1e9,
      static_cast<int32_t>(i), static_cast<int32_t>(j)};
  }
  return c;
}

const ElevationGrid::Cell * ElevationGrid::find(long i, long j) const
{
  const Cell & c = cells_[slot(i, j)];
  return (c.ki == i && c.kj == j) ? &c : nullptr;
}

void ElevationGrid::update(double t, const std::vector<Point> & pts, double x, double y, double yaw)
{
  const double cy = std::cos(yaw), sy = std::sin(yaw);
  std::vector<Point> W(pts.size());
  for (size_t k = 0; k < pts.size(); ++k) {
    const Point & q = pts[k];
    W[k] = {q.x * cy - q.y * sy + x, q.x * sy + q.y * cy + y, q.z + zs_};
  }
  align(t, W, x, y);
  zs_ += c_[0];
  for (auto & q : W) {q.z += c_[0] + c_[1] * (q.x - x) + c_[2] * (q.y - y);}
  add(t, W, x, y);
}

// Height offset and slope of the scan vs the map: trimmed least squares of
// (map ground - scan cell minimum) = a + b*dx + c*dy, starting from the previous slope.
// The slope is then pulled toward level so it cannot drift without bound.
void ElevationGrid::align(double t, const std::vector<Point> & W, double cx, double cy)
{
  ++scan_;
  ti_.clear();
  tj_.clear();
  for (size_t k = 0; k < W.size(); ++k) {
    const double ex = W[k].x - cx, ey = W[k].y - cy;
    const double r = std::sqrt(ex * ex + ey * ey);
    if (!(r > p_.fit_rmin && r < p_.fit_rmax)) {continue;}
    const long i = gi(W[k].x), j = gj(W[k].y);
    const size_t s = slot(i, j);
    if (stamp_[s] != scan_) {
      stamp_[s] = scan_;
      best_[s] = static_cast<int32_t>(k);
      ti_.push_back(i);
      tj_.push_back(j);
    } else if (W[k].z < W[best_[s]].z) {
      best_[s] = static_cast<int32_t>(k);
    }
  }
  std::vector<double> dx, dy, d;
  const double b0 = bc_[0], c0 = bc_[1];
  for (size_t n = 0; n < ti_.size(); ++n) {
    const Cell * c = find(ti_[n], tj_[n]);
    if (!c || std::isnan(c->zavg) || t - c->tmin > p_.t_exp || t - c->tmin <= 0.05) {continue;}
    const Point & q = W[best_[slot(ti_[n], tj_[n])]];
    dx.push_back(q.x - cx);
    dy.push_back(q.y - cy);
    d.push_back(static_cast<double>(c->zavg) - (q.z + b0 * dx.back() + c0 * dy.back()));
  }
  std::vector<char> m(d.size());
  size_t cnt = 0;
  for (size_t k = 0; k < d.size(); ++k) {cnt += (m[k] = std::abs(d[k]) < 0.15);}
  if (cnt < 300) {
    c_[0] = 0.0; c_[1] = b0; c_[2] = c0;
    return;
  }
  double c[3] = {0.0, 0.0, 0.0};
  std::vector<double> res(d.size());
  for (int it = 0; it < 3; ++it) {
    if (!planeFit(dx, dy, d, m, c)) {break;}
    double s = 0.0, s2 = 0.0;
    size_t nm = 0;
    for (size_t k = 0; k < d.size(); ++k) {
      res[k] = d[k] - (c[0] + c[1] * dx[k] + c[2] * dy[k]);
      if (m[k]) {s += res[k]; ++nm;}
    }
    const double mean = s / nm;
    for (size_t k = 0; k < d.size(); ++k) {
      if (m[k]) {s2 += (res[k] - mean) * (res[k] - mean);}
    }
    const double th = std::max(0.02, 2.5 * std::sqrt(s2 / nm));
    for (size_t k = 0; k < d.size(); ++k) {m[k] = std::abs(res[k]) < th;}
  }
  c[1] += b0;
  c[2] += c0;
  const bool moved = !has_last_ || std::hypot(cx - last_x_, cy - last_y_) > 0.02;
  has_last_ = true;
  last_x_ = cx;
  last_y_ = cy;
  if (moved || !p_.freeze_standing) {
    c[1] += p_.slope_pull * (0.0 - c[1]);
    c[2] += p_.slope_pull * (0.0 - c[2]);
  }
  bc_[0] = c[1];
  bc_[1] = c[2];
  std::copy(c, c + 3, c_);
}

void ElevationGrid::add(double t, const std::vector<Point> & W, double cx, double cy)
{
  ++scan_;
  ti_.clear();
  tj_.clear();
  for (const auto & q : W) {
    const long i = gi(q.x), j = gj(q.y);
    Cell & c = touch(i, j);
    const double ex = q.x - cx, ey = q.y - cy;
    const double rr = std::sqrt(ex * ex + ey * ey);
    // reference ground: the cell's stored height if fresh, else robot ground + grade allowance
    const bool fr = t - c.tmin <= p_.t_exp && !std::isnan(c.zavg);
    const double ref = fr ? static_cast<double>(c.zavg) : zs_ + std::min(0.15 * rr, p_.ovh_cap);
    if (!(q.z - ref < p_.h_ovh)) {continue;}
    const size_t s = slot(i, j);
    if (stamp_[s] != scan_) {
      stamp_[s] = scan_;
      zlo_[s] = zhi_[s] = q.z;
      rlo_[s] = rr;
      ti_.push_back(i);
      tj_.push_back(j);
    } else {
      if (q.z < zlo_[s]) {zlo_[s] = q.z; rlo_[s] = rr;}
      zhi_[s] = std::max(zhi_[s], q.z);
    }
  }
  // cells spanning more than obj_ext in this scan hold an object: keep them and their
  // 8 neighbours out of the ground layer
  for (size_t n = 0; n < ti_.size(); ++n) {
    const size_t s = slot(ti_[n], tj_[n]);
    if (zhi_[s] - zlo_[s] <= p_.obj_ext) {continue;}
    for (int a = -1; a <= 1; ++a) {
      for (int b = -1; b <= 1; ++b) {obj_stamp_[slot(ti_[n] + a, tj_[n] + b)] = scan_;}
    }
  }
  for (size_t n = 0; n < ti_.size(); ++n) {
    const size_t s = slot(ti_[n], tj_[n]);
    Cell & c = cells_[s];
    const double v = zlo_[s], rv = rlo_[s];
    if (obj_stamp_[s] != scan_) {
      const bool fresh = std::isnan(c.zavg) || t - c.tmin > p_.t_exp;
      bool keep = fresh || rv <= c.rng + p_.r_slack;   // nearer observations win
      // a first observation far above the robot's ground cannot be ground (canopy)
      if (fresh) {keep = keep && v <= zs_ + std::min(0.15 * rv, p_.ovh_cap) + p_.gnd_max;}
      if (keep) {
        if (fresh) {
          c.rng = static_cast<float>(rv);
          c.zavg = static_cast<float>(v);
        } else {
          c.rng = static_cast<float>(std::min(static_cast<double>(c.rng), rv));
          const double oa = c.zavg;
          c.zavg = static_cast<float>(oa + 0.2 * (v - oa));
        }
        c.tmin = t;
      }
    }
    // obstacle top: max over the last t_obs
    const double top = zhi_[s];
    if (t - c.tmax > p_.t_obs || std::isnan(c.zmax) || top > c.zmax) {
      c.zmax = static_cast<float>(top);
      c.tmax = t;
    }
  }
}

void ElevationGrid::features(const std::vector<float> & z, const std::vector<float> & zmax, int n,
                             std::vector<float> & step, std::vector<float> * hobs, bool split) const
{
  const size_t nn = static_cast<size_t>(n) * n;
  auto in = [n](int i, int j) {return i >= 0 && i < n && j >= 0 && j < n;};
  // nan-aware k x k median, only where observed. Missing values become +inf and sort to
  // the end; a branch-free sorting network runs on L cells at once (vectorizes).
  std::vector<float> hs(nn, kNaNf);
  const int h = p_.k_med / 2, K = p_.k_med * p_.k_med;
  constexpr int L = 8;
  constexpr float kInf = std::numeric_limits<float>::infinity();
  // padded copy with missing -> +inf, and the number of observed cells per k x k window
  const int Wp = n + 2 * h + L;
  std::vector<float> pz(static_cast<size_t>(n + 2 * h) * Wp, kInf);
  std::vector<int> colc(static_cast<size_t>(n + 2 * h) * n, 0), cnt(nn, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (!std::isnan(z[i * n + j])) {pz[(i + h) * Wp + j + h] = z[i * n + j];}
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int c = 0;
      for (int b = std::max(-h, -j); b <= std::min(h, n - 1 - j); ++b) {c += !std::isnan(z[i * n + j + b]);}
      colc[(i + h) * n + j] = c;   // row-wise window count
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int c = 0;
      for (int a = -h; a <= h; ++a) {c += colc[(i + h + a) * n + j];}
      cnt[i * n + j] = c;
    }
  }
  auto medianRows = [&](int i_begin, int i_end) {
  std::vector<float> v(static_cast<size_t>(K) * L);
  for (int i = i_begin; i < i_end; ++i) {
    for (int j0 = 0; j0 < n; j0 += L) {
      bool any = false;
      for (int l = 0; l < L && j0 + l < n; ++l) {any = any || !std::isnan(z[i * n + j0 + l]);}
      if (!any) {continue;}
      int q = 0;
      for (int a = 0; a <= 2 * h; ++a) {
        for (int b = 0; b <= 2 * h; ++b, ++q) {
          std::copy_n(&pz[(i + a) * Wp + j0 + b], L, &v[q * L]);
        }
      }
      // branch-free sorting network on L cells at once; +inf sorts to the end
      for (const auto & [a, b] : net_) {
        for (int l = 0; l < L; l += 4) {
          Vec4 x, y;
          std::memcpy(&x, &v[a * L + l], sizeof x);
          std::memcpy(&y, &v[b * L + l], sizeof y);
          const Vec4 lo = y < x ? y : x, hi = y < x ? x : y;
          std::memcpy(&v[a * L + l], &lo, sizeof lo);
          std::memcpy(&v[b * L + l], &hi, sizeof hi);
        }
      }
      for (int l = 0; l < L && j0 + l < n; ++l) {
        const int k = i * n + j0 + l;
        if (std::isnan(z[k])) {continue;}
        const int m = cnt[k] / 2;
        hs[k] = cnt[k] % 2 ? v[m * L + l] : (v[(m - 1) * L + l] + v[m * L + l]) / 2.0f;
      }
    }
  }
  };
  if (split) {   // rows are independent: top half on a second thread
    auto top = std::async(std::launch::async, medianRows, 0, n / 2);
    medianRows(n / 2, n);
    top.get();
  } else {
    medianRows(0, n);
  }
  // bridge unobserved gaps (curb face, occlusion shadow) with the nearest observed height
  std::vector<std::pair<int, int>> ring;
  const int R = static_cast<int>(p_.fill / p_.res) + 1;
  for (int a = -R; a <= R; ++a) {
    for (int b = -R; b <= R; ++b) {
      if ((a || b) && std::sqrt(static_cast<double>(a * a + b * b)) * p_.res <= p_.fill) {ring.emplace_back(a, b);}
    }
  }
  // nearest first; ties go to the smaller j, then the smaller i (matches scipy's EDT)
  std::sort(ring.begin(), ring.end(), [](auto & u, auto & v) {
      const int du = u.first * u.first + u.second * u.second, dv = v.first * v.first + v.second * v.second;
      if (du != dv) {return du < dv;}
      return u.second != v.second ? u.second < v.second : u.first < v.first;});
  // NaN-padded copy so the search needs no bounds checks
  const int Wf = n + 2 * R;
  std::vector<float> ph(static_cast<size_t>(Wf) * Wf, kNaNf);
  for (int i = 0; i < n; ++i) {std::copy_n(&hs[i * n], n, &ph[(i + R) * Wf + R]);}
  std::vector<long> roff;
  for (auto [a, b] : ring) {roff.push_back(static_cast<long>(a) * Wf + b);}
  std::vector<float> hf = hs;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (!std::isnan(hs[i * n + j])) {continue;}
      const float * c = &ph[(i + R) * Wf + j + R];
      for (long o : roff) {
        if (!std::isnan(c[o])) {
          hf[i * n + j] = c[o];
          break;
        }
      }
    }
  }
  const auto ds = disk(p_.r_step, p_.res);
  step.assign(nn, kNaNf);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::isnan(hf[i * n + j])) {continue;}
      float hi = -std::numeric_limits<float>::infinity(), lo = std::numeric_limits<float>::infinity();
      for (auto [a, b] : ds) {
        if (!in(i + a, j + b)) {continue;}
        const float v = hf[(i + a) * n + j + b];
        if (std::isnan(v)) {continue;}
        hi = std::max(hi, v);
        lo = std::min(lo, v);
      }
      step[i * n + j] = hi - lo;
    }
  }
  if (!hobs) {return;}
  // disk minimum = min over rows of row minima with the disk's half width on that row;
  // rmin[w][k] = min of hf over [j - w, j + w] in row i, built up one width at a time
  const int kg = static_cast<int>(std::round(p_.r_gnd / p_.res));
  std::vector<std::vector<float>> rmin(kg + 1, std::vector<float>(nn, kInf));
  for (size_t k = 0; k < nn; ++k) {if (!std::isnan(hf[k])) {rmin[0][k] = hf[k];}}
  for (int w = 1; w <= kg; ++w) {
    for (int i = 0; i < n; ++i) {
      const float * prev = &rmin[w - 1][i * n];
      const float * row = &rmin[0][i * n];
      float * cur = &rmin[w][i * n];
      for (int j = 0; j < n; ++j) {
        float m = prev[j];
        if (j - w >= 0) {m = std::min(m, row[j - w]);}
        if (j + w < n) {m = std::min(m, row[j + w]);}
        cur[j] = m;
      }
    }
  }
  std::vector<int> half(2 * kg + 1);
  for (int a = -kg; a <= kg; ++a) {
    int w = 0;
    while ((w + 1) * (w + 1) + a * a <= kg * kg) {++w;}
    half[a + kg] = w;
  }
  hobs->assign(nn, kNaNf);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (std::isnan(zmax[i * n + j])) {continue;}
      float lo = kInf;
      for (int a = std::max(-kg, -i); a <= std::min(kg, n - 1 - i); ++a) {
        lo = std::min(lo, rmin[half[a + kg]][(i + a) * n + j]);
      }
      if (std::isfinite(lo)) {(*hobs)[i * n + j] = zmax[i * n + j] - lo;}
    }
  }
}

// Drop-offs along rays from the sensor. A drop edge occludes the lower ground behind it,
// so look for: observed ground -> unobserved gap -> observed ground lower than the near
// side extrapolated across the gap, with a gap no longer than the shadow that drop casts.
void ElevationGrid::drops(Window & w, double sx, double sy, double sz) const
{
  const int n = w.n;
  const double r0 = 0.4, rmax = 10.0, ds = 0.05;
  const int n_ang = 720;
  const int R = static_cast<int>(std::ceil((rmax - r0) / ds));
  const int nb = static_cast<int>(0.3 / ds), nf = static_cast<int>(0.3 / ds), nfit = static_cast<int>(1.0 / ds);
  std::vector<double> rs(R);
  for (int k = 0; k < R; ++k) {rs[k] = r0 + k * ds;}
  w.drop.assign(static_cast<size_t>(n) * n, kNaNf);
  w.shadow.assign(static_cast<size_t>(n) * n, 0);
  std::vector<int> I(R), J(R);
  std::vector<char> ok(R), obs(R), ob(R);
  std::vector<double> zr(R), tmp;
  std::vector<int> near;
  const double astep = 2.0 * M_PI / n_ang;
  for (int k = 0; k < n_ang; ++k) {
    const double ang = k * astep + (-M_PI);
    const double ca = std::cos(ang), sa = std::sin(ang);
    int nobs = 0;
    for (int s = 0; s < R; ++s) {
      I[s] = static_cast<int>((sx + ca * rs[s] - w.wx) / p_.res);
      J[s] = static_cast<int>((sy + sa * rs[s] - w.wy) / p_.res);
      ok[s] = I[s] >= 0 && I[s] < n && J[s] >= 0 && J[s] < n;
      zr[s] = ok[s] ? static_cast<double>(w.zall[I[s] * n + J[s]]) : kNaN;
      obs[s] = !std::isnan(zr[s]);
      const float ho = ok[s] ? w.hobs[I[s] * n + J[s]] : kNaNf;
      ob[s] = !std::isnan(ho) && ho > p_.obs_height;
      nobs += obs[s];
    }
    if (nobs < 10) {continue;}
    for (int a = 1; a < R; ++a) {
      if (!(obs[a - 1] && !obs[a])) {continue;}   // a: first unobserved sample of a gap
      int b = a + 1;                                // b: first observed sample after it
      while (b < R && !obs[b]) {++b;}
      if (b >= R) {break;}
      if ((b - a) * ds < p_.drop_min_gap || a < nb) {continue;}
      bool occl = false;                            // occluded by an object
      for (int s = a - 2; s <= b; ++s) {occl = occl || ob[s];}
      if (occl) {continue;}
      near.clear();
      for (int s = std::max(a - nfit, 0); s < a; ++s) {if (obs[s]) {near.push_back(s);}}
      if (near.size() < 4) {continue;}
      tmp.clear();
      for (int s : near) {if (s >= a - nb) {tmp.push_back(zr[s]);}}
      if (tmp.size() < 3) {continue;}
      const double zn = median(tmp);
      // the near side must be reachable ground, not a wall or hedge top stored as ground
      if (zn - zs_ > std::min(0.15 * rs[a - 1], 0.4) + 0.25) {continue;}
      double mx = 0.0, my = 0.0;
      for (int s : near) {mx += rs[s]; my += zr[s];}
      mx /= near.size();
      my /= near.size();
      double sxy = 0.0, sxx = 0.0;
      for (int s : near) {sxy += (rs[s] - mx) * (zr[s] - my); sxx += (rs[s] - mx) * (rs[s] - mx);}
      const double slope = std::clamp(sxy / sxx, -0.15, 0.15);
      tmp.clear();
      for (int s = b; s < std::min(b + nf, R); ++s) {if (obs[s]) {tmp.push_back(zr[s]);}}
      if (tmp.size() < 3) {continue;}
      const auto [mn, mxz] = std::minmax_element(tmp.begin(), tmp.end());
      if (*mxz - *mn > 0.1) {continue;}             // far side must be a consistent surface
      const double zf = median(tmp);
      const double dz = zn + slope * (rs[b] - rs[a - 1]) - zf;
      if (dz < p_.drop_min) {continue;}
      const double exp_gap = rs[a - 1] * (zn - zf) / std::max(sz - zn, 0.3);
      if ((b - a) * ds > 2.0 * exp_gap + 0.4) {continue;}
      float & e = w.drop[I[a - 1] * n + J[a - 1]];
      e = std::isnan(e) ? static_cast<float>(dz) : std::max(e, static_cast<float>(dz));
      for (int s = a; s < b; ++s) {if (ok[s]) {w.shadow[I[s] * n + J[s]] = 1;}}
    }
  }
}

Window ElevationGrid::window(double t, double x, double y, double yaw) const
{
  Window w;
  const int n = static_cast<int>(2 * p_.half / p_.res);
  const long i0 = static_cast<long>(std::floor((x - p_.half - ox_) / p_.res));
  const long j0 = static_cast<long>(std::floor((y - p_.half - oy_) / p_.res));
  w.n = n;
  w.wx = ox_ + i0 * p_.res;
  w.wy = oy_ + j0 * p_.res;
  const size_t nn = static_cast<size_t>(n) * n;
  w.zall.assign(nn, kNaNf);
  w.zmax.assign(nn, kNaNf);
  std::vector<float> zc(nn, kNaNf);
  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      const Cell * c = find(i0 + a, j0 + b);
      if (!c) {continue;}
      const size_t k = static_cast<size_t>(a) * n + b;
      if (t - c->tmin <= p_.t_exp) {
        w.zall[k] = c->zavg;
        if (c->rng <= p_.r_conf) {zc[k] = c->zavg;}
      }
      if (t - c->tmax <= p_.t_obs) {w.zmax[k] = c->zmax;}
    }
  }
  // the confirmed-cell step is independent of the rest: run it on a second thread
  auto confirmed = std::async(std::launch::async, [&] {features(zc, w.zmax, n, w.step_c, nullptr, false);});
  features(w.zall, w.zmax, n, w.step_all, &w.hobs, true);
  drops(w, x + p_.lidar_x * std::cos(yaw), y + p_.lidar_x * std::sin(yaw), zs_ + p_.lidar_z);
  confirmed.get();
  return w;
}

}  // namespace traversability_grid

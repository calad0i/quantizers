#include "ap_types/ap_float.h"
#include <vector>
namespace qkn_test {
template <int M, int E, int E0> float floatq(float x) {
  ap_float<M, E, E0> a(x);
  return a.to_float();
}

template <int M, int E, int E0>
std::vector<ap_float<M, E, E0>> floatq_vec(std::vector<float> vx) {
  std::vector<ap_float<M, E, E0>> res(vx.size());
  for (int i = 0; i < vx.size(); i++) {
    ap_float<M, E, E0> a(vx[i]);
    res[i] = a.to_float();
  }
  return res;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O>
float fixedq(float x) {
  ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0> a(x);
  return a.to_float();
};

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O>
std::vector<ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0>>
fixedq_vec(std::vector<float> vx) {
  std::vector<ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0>> res(
      vx.size());
  for (int i = 0; i < vx.size(); i++) {
    ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0> a(vx[i]);
    res[i] = a;
  }
  return res;
}
} // namespace qkn_test

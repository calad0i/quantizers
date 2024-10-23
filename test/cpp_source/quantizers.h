#include "ap_types/ap_binary.h"
#include "ap_types/ap_float.h"
#include <vector>
namespace qkn_test {
template <int M, int E, int E0> std::vector<ap_float<M, E, E0>> floatq(std::vector<float> vx) {
    std::vector<ap_float<M, E, E0>> res(vx.size());
    for (int i = 0; i < vx.size(); i++) {
        ap_float<M, E, E0> a(vx[i]);
        res[i] = a.to_float();
    }
    return res;
}

template <int _AP_W, int _AP_I, bool _AP_S, ap_q_mode _AP_Q, ap_o_mode _AP_O>
std::vector<ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0>> fixedq(std::vector<float> vx) {
    std::vector<ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0>> res(vx.size());
    for (int i = 0; i < vx.size(); i++) {
        ap_fixed_base<_AP_W, _AP_I, _AP_S, _AP_Q, _AP_O, 0> a(vx[i]);
        res[i] = a;
    }
    return res;
}

std::vector<ap_binary> binaryq(std::vector<float> vx) {
    std::vector<ap_binary> res(vx.size());
    for (int i = 0; i < vx.size(); i++) {
        ap_binary a(vx[i]);
        res[i] = a;
    }
    return res;
}

auto ternaryq(std::vector<float> vx) { return fixedq<2, 2, true, AP_RND_CONV, AP_SAT_SYM>(vx); }

} // namespace qkn_test

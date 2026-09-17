// The exchange alone: the runtime's comm section (comm.inc, extracted from
// tp_llama.cpp) driven by a loop of exchanges with nothing between them, so
// the per-exchange cost is measured without engine time or rank skew.
// Variants: the committed all-gather with blocking reads; the same with
// non-blocking reads polled round-robin; an all-reduce as reduce-scatter +
// all-gather (half the bytes, two rounds, same rank-ordered float sums).
//
//   mesh_bench --rank r --nranks n --peers ip0,ip1,... [--port 5556] [--iters 1000]
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#define DIE(...) do { fprintf(stderr, "mesh_bench: " __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)
static double now() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#include "comm.inc"

struct bench_comm : comm_t {
    // writes as committed, then non-blocking reads round-robin until every peer's payload is in
    void allgather_poll(const uint8_t * in, uint8_t * out, int len) {
        memcpy(out + (size_t) rank * len, in, len);
        for (int r = 0; r < n; ++r) if (r != rank) write_all(fd[r], in, len);
        std::vector<int> got(n, 0);
        int remaining = n - 1;
        while (remaining > 0) {
            for (int r = 0; r < n; ++r) {
                if (r == rank || got[r] == len) continue;
                ssize_t k = recv(fd[r], out + (size_t) r * len + got[r], len - got[r], MSG_DONTWAIT);
                if (k > 0) { got[r] += (int) k; if (got[r] == len) --remaining; }
                else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) DIE("peer connection lost (poll read)");
            }
        }
    }
    // point-to-point exchange of per-peer payloads: send buf_out[r] to r, receive buf_in[r] from r
    void exchange(const std::vector<std::vector<uint8_t>> & to, std::vector<std::vector<uint8_t>> & from, int len, bool poll) {
        for (int r = 0; r < n; ++r) if (r != rank) write_all(fd[r], to[r].data(), len);
        if (!poll) { for (int r = 0; r < n; ++r) if (r != rank) read_all(fd[r], from[r].data(), len); return; }
        std::vector<int> got(n, 0);
        int remaining = n - 1;
        while (remaining > 0) for (int r = 0; r < n; ++r) {
            if (r == rank || got[r] == len) continue;
            ssize_t k = recv(fd[r], from[r].data() + got[r], len - got[r], MSG_DONTWAIT);
            if (k > 0) { got[r] += (int) k; if (got[r] == len) --remaining; }
            else if (k == 0 || (k < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) DIE("peer connection lost (poll read)");
        }
    }
    // all-reduce as reduce-scatter (each rank sums its quarter in rank order) + all-gather of the quarters
    void allreduce_rs(float * v, int len, bool poll) {
        const int q = len / n;
        std::vector<std::vector<uint8_t>> to(n, std::vector<uint8_t>(q * 4)), from(n, std::vector<uint8_t>(q * 4));
        for (int r = 0; r < n; ++r) if (r != rank) memcpy(to[r].data(), v + r * q, q * 4);
        exchange(to, from, q * 4, poll);
        float * mine = v + rank * q;
        for (int i = 0; i < q; ++i) {
            float s = 0;
            for (int r = 0; r < n; ++r) s = s + (r == rank ? mine[i] : ((const float *) from[r].data())[i]);
            mine[i] = s;
        }
        for (int r = 0; r < n; ++r) if (r != rank) memcpy(to[r].data(), mine, q * 4);
        exchange(to, from, q * 4, poll);
        for (int r = 0; r < n; ++r) if (r != rank) memcpy(v + r * q, from[r].data(), q * 4);
    }
};

static void report(const char * name, int bytes, std::vector<double> & us) {
    std::sort(us.begin(), us.end());
    double mean = 0; for (double x : us) mean += x; mean /= us.size();
    printf("%-44s %6d B  median %7.1f us  mean %7.1f  p90 %7.1f  max %8.1f\n", name, bytes, us[us.size() / 2], mean, us[us.size() * 9 / 10], us.back());
}

int main(int argc, char ** argv) {
    int rank = 0, nranks = 1, port = 5556, iters = 1000;
    std::vector<std::string> ips;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&] { if (i + 1 >= argc) DIE("%s needs a value", a.c_str()); return std::string(argv[++i]); };
        if (a == "--rank") rank = atoi(val().c_str());
        else if (a == "--nranks") nranks = atoi(val().c_str());
        else if (a == "--port") port = atoi(val().c_str());
        else if (a == "--iters") iters = atoi(val().c_str());
        else if (a == "--peers") { std::string p = val(); size_t s = 0; for (;;) { size_t c = p.find(',', s); ips.push_back(p.substr(s, c - s)); if (c == std::string::npos) break; s = c + 1; } }
        else DIE("unknown arg %s", a.c_str());
    }
    if ((int) ips.size() != nranks) DIE("--peers must list %d addresses", nranks);
    bench_comm C;
    C.open(rank, ips, port);
    std::vector<uint8_t> in(65536, (uint8_t) rank), out((size_t) nranks * 65536);
    std::vector<float> v(4096), scratch;
    std::vector<double> us;
    auto run = [&](const char * name, int bytes, auto fn) {
        us.clear();
        for (int i = 0; i < 50; ++i) fn();                       // warm up
        for (int i = 0; i < iters; ++i) { double t0 = now(); fn(); us.push_back(1e6 * (now() - t0)); }
        report(name, bytes, us);
    };
    for (int bytes : {64, 2048, 4096, 8192, 16384}) {
        run("all-gather (committed comm_t)", bytes, [&] { C.allgather(in.data(), out.data(), bytes); });
        run("all-gather, bench variant, polled reads", bytes, [&] { C.allgather_poll(in.data(), out.data(), bytes); });
    }
    run("all-reduce 2048 f32 (committed comm_t)", 8192, [&] { C.allreduce_sum(v.data(), 2048, scratch); });
    run("all-reduce, bench reduce-scatter+all-gather, blocking", 8192, [&] { C.allreduce_rs(v.data(), 2048, false); });
    run("all-reduce, bench reduce-scatter+all-gather, polled", 8192, [&] { C.allreduce_rs(v.data(), 2048, true); });
    return 0;
}

// Calls the pinned upstream NIPP implementation. The extra encrypted support
// output is an explicit top-k adaptation of its threshold-query interface.
#include "cloud.h"
#include "user.h"
#include <tfhe/tfhe_io.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
struct CipherArray {
    LweSample* p;
    int n;
    CipherArray(int n, const TFheGateBootstrappingParameterSet* params)
        : p(new_gate_bootstrapping_ciphertext_array(n, params)), n(n) {}
    ~CipherArray() { delete_gate_bootstrapping_ciphertext_array(n, p); }
    CipherArray(const CipherArray&) = delete;
    CipherArray& operator=(const CipherArray&) = delete;
};

// 云端只能访问密文和 cloud key。不会收到私钥或原始支持数。
static void cloud_query(LweSample* support, LweSample* below, LweSample* above,
                        const std::vector<LweSample*>& encrypted_rows,
                        int columns, int width, const LweSample* query,
                        const LweSample* threshold,
                        const TFheGateBootstrappingCloudKeySet* cloud,
                        const std::vector<int>* public_query = nullptr) {
    for (int j = 0; j < width; ++j) bootsCONSTANT(&support[j], 0, cloud);
    CipherArray membership(1, cloud->params);
    for (const auto* row : encrypted_rows) {
        if (public_query)
            ppfim::somewhat_secure_subset_testing(membership.p, *public_query, row, columns, cloud);
        else
            ppfim::secure_subset_testing(membership.p, query, row, columns, cloud);
        ppfim::secure_count(support, width, membership.p, cloud);
    }
    // 保留上游的 support < threshold 语义，并额外给出严格大于关系。
    ppfim::secure_compare(below, support, threshold, width, cloud);
    ppfim::secure_compare(above, threshold, support, width, cloud);
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4)
            throw std::runtime_error("usage: nipp_upstream input.txt [--check-upstream] [--public-query]");
        bool check = false, public_query = false;
        for (int arg = 2; arg < argc; ++arg) {
            const std::string option(argv[arg]);
            if (option == "--check-upstream") check = true;
            else if (option == "--public-query") public_query = true;
            else throw std::runtime_error("unknown option");
        }
        std::ifstream input(argv[1]);
        int n, d, owners, q, threshold;
        if (!(input >> n >> d >> owners >> q >> threshold) || n < 1 || d < 1 ||
            owners < 1 || q < 1 || threshold < 0 || threshold > n)
            throw std::runtime_error("invalid dimensions/threshold");
        std::vector<int> owner(d), counts(owners, 0);
        for (auto& o : owner) {
            if (!(input >> o) || o < 0 || o >= owners) throw std::runtime_error("invalid owner");
            ++counts[o];
        }
        // A broadcast candidate may not reference every vertical owner.
        // Such owners send no selected item ciphertexts in this query batch.
        std::vector<int> data(static_cast<size_t>(n) * d), queries(static_cast<size_t>(q) * d);
        for (auto* bits : {&data, &queries}) for (auto& b : *bits)
            if (!(input >> b) || (b != 0 && b != 1)) throw std::runtime_error("expected binary input");
        int width = 1;
        while ((uint64_t(1) << width) <= static_cast<uint64_t>(n)) ++width;
        const auto start = Clock::now();
        auto phase = Clock::now();
        std::random_device random;
        uint32_t entropy[16];
        for (auto& x : entropy) x = random();
        tfhe_random_generator_setSeed(entropy, 16);
        auto* params = new_default_gate_bootstrapping_parameters(128);
        auto* key = new_random_gate_bootstrapping_secret_keyset(params);
        const double keygen = elapsed(phase);
        {
            std::vector<std::unique_ptr<CipherArray>> storage;
            std::vector<LweSample*> rows;
            for (int r = 0; r < n; ++r) {
                storage.emplace_back(new CipherArray(d, params));
                rows.push_back(storage.back()->p);
            }
            CipherArray eq(q * d, params), et(width, params);
            CipherArray support(q * width, params), below(q, params), above(q, params);
            std::vector<double> owner_seconds(owners);
            phase = Clock::now();
            // 同一用户行已经对齐，各数据方只加密自己的列，云端拼接密文。
            for (int o = 0; o < owners; ++o) {
                auto os = Clock::now();
                for (int r = 0; r < n; ++r) for (int j = 0; j < d; ++j)
                    if (owner[j] == o) bootsSymEncrypt(&rows[r][j], data[r * d + j], key);
                owner_seconds[o] = elapsed(os);
            }
            if (!public_query || check)
                for (size_t j = 0; j < queries.size(); ++j) bootsSymEncrypt(&eq.p[j], queries[j], key);
            ppfim::ctxt_min_supp_count(et.p, width, threshold, key);
            const double encryption = elapsed(phase);
            phase = Clock::now();
            int completed_queries = 0;
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < q; ++i) {
                std::vector<int> plain;
                if (public_query)
                    plain.assign(queries.begin() + i * d, queries.begin() + (i + 1) * d);
                cloud_query(&support.p[i * width], &below.p[i], &above.p[i], rows,
                            d, width, &eq.p[i * d], et.p, &key->cloud,
                            public_query ? &plain : nullptr);
#pragma omp critical(progress)
                {
                    ++completed_queries;
                    std::cerr << "query " << completed_queries << "/" << q << " completed\n";
                }
            }
            const double cloud = elapsed(phase);
            phase = Clock::now();
            std::vector<int> supports(q), lt(q), gt(q);
            for (int i = 0; i < q; ++i) {
                for (int j = 0; j < width; ++j)
                    supports[i] = (supports[i] << 1) | bootsSymDecrypt(&support.p[i * width + j], key);
                lt[i] = bootsSymDecrypt(&below.p[i], key);
                gt[i] = bootsSymDecrypt(&above.p[i], key);
            }
            const double decrypt = elapsed(phase);
            const double protocol = elapsed(start);
            // 复核原始上游入口，单独计时，不重复计入协议运行时间。
            phase = Clock::now();
            if (check) {
                CipherArray result(1, params);
                for (int i = 0; i < q; ++i) {
                    ppfim::freq_itemset_mining_first(result.p, rows, n, d, &eq.p[i * d], et.p, &key->cloud);
                    if (bootsSymDecrypt(result.p, key) != lt[i])
                        throw std::runtime_error("upstream encrypted-query disagreement");
                    std::vector<int> plain(queries.begin() + i * d, queries.begin() + (i + 1) * d);
                    ppfim::freq_itemset_mining_second(result.p, rows, n, d, plain, et.p, &key->cloud);
                    if (bootsSymDecrypt(result.p, key) != lt[i])
                        throw std::runtime_error("upstream public-query disagreement");
                }
            }
            const double validation = check ? elapsed(phase) : 0.;
            std::ostringstream serialized;
            export_gate_bootstrapping_ciphertext_toStream(serialized, rows[0], params);
            auto vector_json = [](const auto& values) {
                std::cout << '[';
                for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
                std::cout << ']';
            };
            std::cout << std::setprecision(12)
                      << "{\"backend\":\"NIPP-upstream-TFHE-128\",\"native_crypto_measured\":true,"
                      << "\"publication_eligible\":false,\"topk_adaptation\":true,"
                      << "\"public_query\":" << (public_query ? "true" : "false") << ","
                      << "\"upstream_entrypoints_checked\":" << (check ? "true" : "false")
                      << ",\"rows\":" << n << ",\"items\":" << d << ",\"queries\":" << q
                      << ",\"keygen_seconds\":" << keygen << ",\"encrypt_seconds\":" << encryption
                      << ",\"cloud_seconds\":" << cloud << ",\"decrypt_seconds\":" << decrypt
                      << ",\"protocol_seconds\":" << protocol << ",\"validation_seconds\":" << validation
                      << ",\"serialized_ciphertext_bytes\":" << serialized.str().size()
                      << ",\"owner_encrypt_seconds\":";
            vector_json(owner_seconds);
            std::cout << ",\"supports\":"; vector_json(supports);
            std::cout << ",\"below_threshold\":"; vector_json(lt);
            std::cout << ",\"above_threshold\":"; vector_json(gt);
            std::cout << "}\n";
        }
        delete_gate_bootstrapping_secret_keyset(key);
        delete_gate_bootstrapping_parameters(params);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

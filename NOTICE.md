# Sources and notices

This release assembles the PrivFim research implementation. No project-wide open-source license has been selected here.
The repository owner should select a license before granting general reuse rights.

Optional third-party repositories are downloaded only on request and excluded from Git:

- TFHE: https://github.com/tfhe/tfhe, pinned at `9373b5d8a5b022ca5b4e112ae1c2440bc18e160c`.
  Preserve its LICENSE and notices, including FFT component notices.
- NIPP implementation: https://github.com/Airscope/non-interactive-ppfim,
  pinned at `d862f5c0d9dd4fa8cd155abdd8bbaa3cc6e63a81`. The inspected checkout did not contain a root LICENSE.
  This release references and builds that checkout rather than redistributing its source as our own.

UCI datasets retain their own attribution and terms. Official source URLs and hashes are recorded by preparation scripts.
Bank is an attributed historical integer reference export, not newly reconstructed preprocessing.

Research references:

- Zhen Zhao, Lei Lan, Baocang Wang, Jianchang Lai. Verifiable Privacy-Preserving Outsourced Frequent Itemset Mining
  on Vertically Partitioned Databases. Electronics 12(8), 1952, 2023. DOI: 10.3390/electronics12081952.
- Peijia Zheng, Ziyan Cheng, Xianhao Tian, Hongmei Liu, Weiqi Luo, Jiwu Huang.
  Non-interactive privacy-preserving frequent itemset mining over encrypted cloud data.
  IEEE Transactions on Cloud Computing 11(4), 3452–3468, 2023.

Historical plotting projections and manually adjusted paper figures are not packaged as measured experiment results.

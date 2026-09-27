# Third-party components

| component | licence | use |
|---|---|---|
| CatBoost 1.2.10 | Apache-2.0 | primary pair classifier (trained artefact: ours, MIT) |
| LightGBM 4.7.0 | MIT | fallback pair classifier |
| scikit-learn, scipy, numpy, pandas, pyarrow | BSD-3 / Apache-2.0 | data handling, TF-IDF, calibration |
| rapidfuzz | MIT | string similarity |
| sparse_dot_topn | Apache-2.0 | sparse top-k matmul |
| jellyfish | MIT (>=1.0) | phonetic encoding |
| intfloat/multilingual-e5-small (optional, disabled) | MIT | dense retrieval channel |

Neural parameter count of the released system: 0 (classical champion). Optional E5-small: ~118M.
No external business data, APIs, geocoders or hosted LLMs are used anywhere in the pipeline.

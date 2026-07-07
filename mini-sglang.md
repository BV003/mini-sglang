# mini-sglang

## Project Architecture Overview

```
mini-sglang/
├── README.md
├── LICENSE
├── pyproject.toml
├── Dockerfile
├── .gitignore
├── .pre-commit-config.yaml
│
├── assets/
│   └── logo.png
│
├── docs/
│   ├── features.md              # Feature descriptions & CLI args
│   └── structures.md            # System architecture & code organization
│
├── benchmark/
│   ├── offline/
│   │   ├── bench.py
│   │   └── bench_wildchat.py
│   └── online/
│       ├── bench_qwen.py
│       └── bench_simple.py
│
├── tests/
│   ├── core/
│   │   ├── test_cache_allocate.py
│   │   └── test_scheduler.py
│   ├── kernel/
│   │   ├── test_comm.py
│   │   ├── test_index.py
│   │   ├── test_store.py
│   │   └── test_tensor.py
│   └── misc/
│       └── test_serialize.py
│
└── python/minisgl/              # ← ALL source code (~5000 lines)
    │
    ├── __main__.py              # Entry point: `python -m minisgl ...`
    ├── core.py                  # Req, Batch, Context, SamplingParams
    ├── env.py                   # Environment variable config
    │
    ├── server/                  # ── API & process launcher ──
    │   ├── args.py              # CLI argument parsing (ServerArgs)
    │   ├── launch.py            # Spawns all subprocesses
    │   └── api_server.py        # FastAPI server + interactive shell
    │
    ├── message/                 # ── ZMQ message protocol ──
    │   ├── backend.py           # Scheduler ↔ Tokenizer/Detokenizer msgs
    │   ├── frontend.py          # API Server ↔ Tokenizer msgs
    │   ├── tokenizer.py         # Tokenize/Detokenize message types
    │   └── utils.py             # ZMQ async queue wrappers
    │
    ├── tokenizer/               # ── Tokenization workers ──
    │   ├── server.py            # Main worker loop
    │   ├── tokenize.py          # Text → token IDs
    │   └── detokenize.py        # Token IDs → text
    │
    ├── scheduler/               # ── Core scheduling logic ──
    │   ├── config.py            # SchedulerConfig dataclass
    │   ├── scheduler.py         # Main Scheduler class + event loop
    │   ├── prefill.py           # PrefillManager (chunked prefill)
    │   ├── decode.py            # DecodeManager (batching decode reqs)
    │   ├── cache.py             # CacheManager (wraps KV cache)
    │   ├── table.py             # TableManager (page table slots)
    │   ├── io.py                # SchedulerIOMixin (ZMQ send/recv)
    │   └── utils.py
    │
    ├── engine/                  # ── GPU execution ──
    │   ├── config.py            # EngineConfig dataclass
    │   ├── engine.py            # Engine class (model load, forward, sample)
    │   ├── graph.py             # GraphRunner (CUDA Graph capture/replay)
    │   └── sample.py            # Sampler (top-k, top-p, temperature)
    │
    ├── models/                  # ── Model architectures ──
    │   ├── base.py              # BaseLLM (Transformer forward loop)
    │   ├── config.py            # ModelConfig dataclass
    │   ├── llama.py             # Llama 3
    │   ├── qwen2.py             # Qwen 2.5
    │   ├── qwen3.py             # Qwen 3 (dense)
    │   ├── qwen3_moe.py         # Qwen 3 (MoE)
    │   ├── mistral.py           # Mistral
    │   ├── weight.py            # HuggingFace weight loading & sharding
    │   ├── register.py          # Model registry
    │   └── utils.py
    │
    ├── layers/                  # ── Layer building blocks ──
    │   ├── base.py              # BaseLayer (TP-aware)
    │   ├── linear.py            # ColumnParallelLinear, RowParallelLinear
    │   ├── attention.py         # AttentionLayer
    │   ├── embedding.py         # VocabParallelEmbedding
    │   ├── norm.py              # RMSNorm
    │   ├── rotary.py            # RoPE (Rotary Position Embedding)
    │   ├── activation.py        # SiLU, etc.
    │   └── moe.py              # MoE layer (router + experts)
    │
    ├── attention/               # ── Attention kernel backends ──
    │   ├── base.py              # BaseAttnBackend interface
    │   ├── fa.py                # FlashAttention backend
    │   ├── fi.py                # FlashInfer backend
    │   ├── trtllm.py            # TensorRT-LLM backend
    │   └── utils.py
    │
    ├── kvcache/                 # ── KV cache management ──
    │   ├── base.py              # BaseKVCachePool, BaseCacheHandle
    │   ├── mha_pool.py          # MHAKVCache (physical memory pool)
    │   ├── naive_cache.py       # NaiveCacheManager (FIFO)
    │   └── radix_cache.py       # RadixCacheManager (prefix sharing)
    │
    ├── moe/                     # ── Mixture of Experts backends ──
    │   ├── base.py
    │   └── fused.py             # Fused MoE kernel
    │
    ├── distributed/             # ── Tensor Parallelism ──
    │   ├── info.py              # DistributedInfo dataclass
    │   └── impl.py              # all_reduce, all_gather wrappers
    │
    ├── kernel/                  # ── Custom CUDA kernels (TVM-FFI) ──
    │   ├── __main__.py
    │   ├── index.py             # Indexing kernel
    │   ├── store.py             # KV store kernel
    │   ├── tensor.py            # Tensor ops kernel
    │   ├── radix.py             # Radix tree kernel
    │   ├── moe_impl.py          # MoE kernel implementation
    │   ├── pynccl.py            # PyNCCL (custom NCCL wrapper)
    │   ├── utils.py
    │   └── triton/
    │       └── fused_moe.py     # Triton fused MoE kernel
    │
    ├── llm/                     # ── Python API ──
    │   └── llm.py               # LLM class (programmatic interface)
    │
    ├── utils/                   # ── Utilities ──
    │   ├── arch.py              # GPU architecture detection
    │   ├── hf.py                # HuggingFace model download
    │   ├── logger.py            # Logging setup (rank-aware)
    │   ├── misc.py              # div_even, etc.
    │   ├── mp.py                # Multiprocessing helpers
    │   ├── registry.py          # Generic registry pattern
    │   └── torch_utils.py       # torch_dtype, etc.
    │
    └── benchmark/               # ── Internal benchmark utils ──
        ├── client.py
        └── perf.py
```

Dependency order:
```
utils → distributed → kernel → layers → models → attention/kvcache/moe
                                                    ↓
server → message → tokenizer → scheduler → engine → (uses all above)
                                                    ↓
                                              llm (public API)
```


```

```


## Core Data Structures

### SamplingParams — what the user wants 

Temperature, top-k, top-p, max_tokens, ignore_eos. Also has is_greedy the helper.


### 2. Req — one request's state

Lifecycle methods: complete_one() shifts the window forward by 1 token after each decode step.append_host() grows the CPU-side token array.

### 3. Batch — a group of Reqs processed together

Two phases: "prefill" or "decode". Fields (input_ids, positions, out_loc, attn_metadata) are set later by the scheduler and attention backend — not by the constructor. 

### 4. Context — global state
Singleton holding everything the model layers need during forward: page_table, kv_cache, attn_backend, moe_backend, and the current Batch. The forward_batch() context manager ensures only one batch is active at a time. 

## The Engine — Model Loading & Execution

## The Scheduler — Request Lifecycle Management

## KV Cache Management

## Attention Backends

## Model Implementations (Layers & Architectures) 

## Tensor Parallelism & Distributed Communication

## Overlap Scheduling 

## Serving API & Tokenizer
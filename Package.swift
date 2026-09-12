// swift-tools-version: 6.2
import PackageDescription

// mlx-indextts2-swift — Swift-MLX port of IndexTTS-2.5 (multilingual zero-shot cloning TTS with
// native emotion + duration control). Donor: vanch007/mlx-indextts2 (MIT, Python-MLX, `v25`
// profile) — the oracle for every golden in PORTING/goldens-v25 (fp32 CPU). The 2.0 tier
// (non-commercial INDEX_MODEL_LICENSE, MaskGCT/RepCodec + SentencePiece) was removed at the
// 2.5 update (v0.4.0) — this package ships ONE model.
//
// Parity gates live in the `indextts2-gate` CLI lane (NOT XCTest — the SPM test product's
// metallib is unreliable; `swift run` is the doctrine for gates that touch kernels).
// XCTest carries the offline checks (tokenizer/frontend parity, weight-free key contracts, the
// manifest + MAT-1..5 materialization gate, CAN-1..3, INF).
//
// `MLXIndexTTS2TTS` is the engine-facing wrapper (IndexTTS2Configuration + IndexTTS2Package)
// over the `MLXIndexTTS2` core — same split as MLXMossTTS. The core stays MLXToolKit-free.
// Engine contract pinned ≥0.51.0 (SPDXLicense.bilibiliModelUse on the permissive allowlist).
let package = Package(
    name: "mlx-indextts2-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "MLXIndexTTS2", targets: ["MLXIndexTTS2"]),
        .library(name: "MLXIndexTTS2TTS", targets: ["MLXIndexTTS2TTS"]),
        .executable(name: "indextts2-gate", targets: ["indextts2-gate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Shared STFT/mel primitives.
        .package(url: "https://github.com/xocialize/mlx-audio-dsp.git", from: "0.1.0"),
        // Engine contract — ≥0.51.0 for SPDXLicense.bilibiliModelUse (the IndexTTS-2.5
        // weight license, allowlisted); the engine executes materialization (≥0.32.0).
        // 0.54.0 ⊇ contract 1.38.0: the E12 controls plane (TTSControls + typed emotion/targetDuration).
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.54.0"),
    ],
    targets: [
        .target(
            name: "MLXIndexTTS2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXAudioDSP", package: "mlx-audio-dsp"),
            ],
            resources: [.copy("Resources")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "indextts2-gate",
            dependencies: [
                "MLXIndexTTS2",
                "MLXIndexTTS2TTS",
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXAudioDSP", package: "mlx-audio-dsp"),
                // `engine` mode: the consumer path (MLXServeEngine register → prepare → run).
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXIndexTTS2TTS",
            dependencies: [
                "MLXIndexTTS2",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXAudioDSP", package: "mlx-audio-dsp"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXIndexTTS2Tests",
            dependencies: [
                "MLXIndexTTS2",
                "MLXIndexTTS2TTS",
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeConformanceNN", package: "mlx-engine-swift"),  // INF gate
            ],
            resources: [.copy("Resources")]
        ),
    ]
)

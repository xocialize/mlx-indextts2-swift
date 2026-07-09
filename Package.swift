// swift-tools-version: 6.2
import PackageDescription

// mlx-indextts2-swift — Swift-MLX port of IndexTTS2 (emotion + duration-controllable TTS),
// donor: solar2ain/mlx-indextts (MIT, Python-MLX) + our MLX-Python front-end ports
// (w2v-BERT 2.0, MaskGCT RepCodec) in _indextts2-oracle. Ported phase-by-phase per the
// mlx-swift-integration parity doctrine; each phase gates against the Stage-0 goldens.
//
// Parity gates live in the `indextts2-gate` CLI lane (NOT XCTest — the SPM test product's
// metallib is unreliable; `swift run` is the doctrine for gates that touch kernels).
// XCTest carries the offline checks (tokenizer parity, weight-free key contracts, the
// Stage-2 manifest + MAT-1..5 materialization gate).
//
// Stage 2: `MLXIndexTTS2TTS` is the engine-facing wrapper (IndexTTS2Configuration +
// IndexTTS2Package) over the `MLXIndexTTS2` core — same split as MLXVoxCPM2TTS/MLXQwen3TTS.
// The core stays MLXToolKit-free. Engine contract is a local-path dep during WIP
// (pin a tagged mlx-engine-swift ≥0.23.0 — the LicenseRef-Index-Model entry — at publish).
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
        // Shared STFT/mel primitives (local path during WIP; publish before tagging).
        .package(path: "../mlx-audio-dsp"),
        // Engine contract (local path during WIP — needs the unreleased LicenseRef-Index-Model
        // + emotionControl/durationControl specialty entries).
        .package(path: "../../../MLXEngine/mlx-engine-swift"),
        // Native downloader for WeightSourcing auto-materialization.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
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
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXAudioDSP", package: "mlx-audio-dsp"),
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
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXIndexTTS2Tests",
            dependencies: [
                "MLXIndexTTS2",
                "MLXIndexTTS2TTS",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            resources: [.copy("Resources")]
        ),
    ]
)

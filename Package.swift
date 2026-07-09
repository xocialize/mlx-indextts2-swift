// swift-tools-version: 6.2
import PackageDescription

// mlx-indextts2-swift — Swift-MLX port of IndexTTS2 (emotion + duration-controllable TTS),
// donor: solar2ain/mlx-indextts (MIT, Python-MLX) + our MLX-Python front-end ports
// (w2v-BERT 2.0, MaskGCT RepCodec) in _indextts2-oracle. Ported phase-by-phase per the
// mlx-swift-integration parity doctrine; each phase gates against the Stage-0 goldens.
//
// Parity gates live in the `indextts2-gate` CLI lane (NOT XCTest — the SPM test product's
// metallib is unreliable; `swift run` is the doctrine for gates that touch kernels).
// XCTest carries only offline checks (tokenizer parity, weight-free key contracts).
// MLXToolKit (engine contract) arrives at Stage 2.
let package = Package(
    name: "mlx-indextts2-swift",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MLXIndexTTS2", targets: ["MLXIndexTTS2"]),
        .executable(name: "indextts2-gate", targets: ["indextts2-gate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
    ],
    targets: [
        .target(
            name: "MLXIndexTTS2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "indextts2-gate",
            dependencies: ["MLXIndexTTS2"],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXIndexTTS2Tests",
            dependencies: ["MLXIndexTTS2"],
            resources: [.copy("Resources")]
        ),
    ]
)

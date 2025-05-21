// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Swift-TTS",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "Swift-TTS",
            targets: ["Swift-TTS","ESpeakNG"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.2"),
        .package(url: "ttps://github.com/huggingface/swift-transformers.git", from: "0.1.20")
    ],
    targets: [
        .binaryTarget(
            name: "ESpeakNG",
            path: "Swift-TTS/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "Swift-TTS",
            dependencies: [.product(name: "MLX", package: "mlx-swift"),
                           .product(name: "Transformers", package: "swift-transformers"),
                           .product(name: "MLXFFT", package: "mlx-swift"),
                           .product(name: "MLXNN", package: "mlx-swift"),
                           "ESpeakNG"
            ],
            path: "Swift-TTS",
            exclude: ["Preview Content","Assets.xcassets","Swift_TTSApp.swift","Swift_TTS.entitlements"]),
        .testTarget(
            name: "Swift-TTS-Tests",
            dependencies: ["Swift-TTS"],
            path: "Tests"
        ),
    ]
)

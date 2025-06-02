# Creating eSpeak-ng visionOS Framework

This guide helps you create a visionOS-compatible framework for eSpeak-ng to use with Apple Vision Pro.

## Prerequisites

- Xcode 15.0+ with visionOS SDK
- macOS 14.0+ (for visionOS development)
- Command Line Tools installed
- An existing eSpeak-ng Xcode project/scheme

## Method 1: Using Your Existing Build Script (Recommended)

Your existing script is actually the cleanest approach. I've enhanced it to include visionOS support.

### Steps:

1. **Update Package.swift** (Already done)
   - Added `.visionOS(.v1)` to supported platforms

2. **Use the Enhanced Build Script**
   - Use `build_xcframework_with_visionos.sh` which adds visionOS device and simulator builds

3. **Prerequisites for Your eSpeak-ng Project**
   Make sure your eSpeak-ng Xcode project has:
   - A scheme named "espeak-ng" 
   - visionOS as a supported platform in project settings
   - Proper framework target configuration

### Running the Script:

```bash
# From your eSpeak-ng project directory
export PROJECT_DIR=/path/to/your/espeak-ng-project
./build_xcframework_with_visionos.sh
```

## Method 2: Manual Xcode Project Setup

If you need to manually configure your eSpeak-ng Xcode project:

### 1. Add visionOS Platform Support

1. Open your eSpeak-ng Xcode project
2. Select your project in the navigator
3. Go to your framework target
4. In "Deployment Info", add visionOS:
   - Minimum deployment: visionOS 1.0

### 2. Configure Build Settings

Add these build settings for visionOS:
- `SUPPORTED_PLATFORMS = macosx iphoneos iphonesimulator xros xrsimulator`
- `VALID_ARCHS[sdk=xros*] = arm64`
- `VALID_ARCHS[sdk=xrsimulator*] = arm64`

### 3. Update Info.plist

Ensure your framework's Info.plist supports visionOS:

```xml
<key>CFBundleSupportedPlatforms</key>
<array>
    <string>MacOSX</string>
    <string>iPhoneOS</string>
    <string>iPhoneSimulator</string>
    <string>XROS</string>
    <string>XRSimulator</string>
</array>
```

## Method 3: From Source (Advanced)

If you need to build eSpeak-ng from source for visionOS:

### Building eSpeak-ng Library for visionOS

```bash
# Clone eSpeak-ng
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng

# Install build tools
brew install autoconf automake libtool

# Build for visionOS Device
./autogen.sh
export SDKROOT=$(xcrun --sdk xros --show-sdk-path)
export CC=$(xcrun --sdk xros --find clang)
export CFLAGS="-arch arm64 -mxros-version-min=1.0 -isysroot $SDKROOT"
export LDFLAGS="-arch arm64 -mxros-version-min=1.0 -isysroot $SDKROOT"

./configure --host=arm64-apple-xros --prefix=/tmp/espeak-visionos \
    --without-async --without-mbrola --without-sonic --disable-rpath

make -j$(sysctl -n hw.ncpu)
make install

# Build for visionOS Simulator
make clean
export SDKROOT=$(xcrun --sdk xrsimulator --show-sdk-path)
export CC=$(xcrun --sdk xrsimulator --find clang)
export CFLAGS="-arch arm64 -mxros-version-min=1.0 -isysroot $SDKROOT"
export LDFLAGS="-arch arm64 -mxros-version-min=1.0 -isysroot $SDKROOT"

./configure --host=arm64-apple-xros --prefix=/tmp/espeak-visionos-sim \
    --without-async --without-mbrola --without-sonic --disable-rpath

make -j$(sysctl -n hw.ncpu)
make install
```

## Testing Your visionOS Framework

### 1. Create a Simple visionOS App

```swift
import SwiftUI
import ESpeakNG

@main
struct VisionOSTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        VStack {
            Text("eSpeak-ng visionOS Test")
            Button("Test Speech") {
                testSpeech()
            }
        }
        .padding()
    }
    
    func testSpeech() {
        // Test your eSpeak-ng integration here
        // This depends on your specific eSpeak-ng wrapper
    }
}
```

### 2. Add Framework to Project

1. Drag your `ESpeakNG.xcframework` into your visionOS project
2. Ensure it's added to "Frameworks, Libraries, and Embedded Content"
3. Set to "Embed & Sign"

### 3. Build and Test

```bash
# Build for visionOS Simulator
xcodebuild -scheme YourApp -destination "platform=visionOS Simulator,name=Apple Vision Pro"

# Build for visionOS Device (requires signing)
xcodebuild -scheme YourApp -destination "generic/platform=visionOS"
```

## Common Issues and Solutions

### Issue: "No such SDK: xros"
**Solution**: Make sure you have Xcode 15+ with visionOS SDK installed.

### Issue: Framework not linking on visionOS
**Solution**: Check that your framework target includes visionOS in supported platforms.

### Issue: Missing symbols for visionOS
**Solution**: Ensure you've built the eSpeak-ng library specifically for visionOS architectures.

### Issue: "Library not loaded" runtime error
**Solution**: Make sure the framework is embedded in your app bundle, not just linked.

## Framework Structure

Your final xcframework should have this structure:

```
ESpeakNG.xcframework/
├── Info.plist
├── ios-arm64/
│   └── ESpeakNG.framework/
├── ios-arm64-simulator/
│   └── ESpeakNG.framework/
├── macos-arm64_x86_64/
│   └── ESpeakNG.framework/
├── xros-arm64/
│   └── ESpeakNG.framework/
└── xros-arm64-simulator/
    └── ESpeakNG.framework/
```

## Next Steps

1. Run the enhanced build script to create your visionOS-compatible framework
2. Test integration in a simple visionOS app
3. Consider adding visionOS-specific optimizations
4. Update your Swift package or distribution method to include visionOS support

## Additional Resources

- [Apple visionOS Developer Documentation](https://developer.apple.com/visionos/)
- [Creating XCFrameworks](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle)
- [eSpeak-ng Documentation](https://github.com/espeak-ng/espeak-ng/blob/master/docs/index.md)

---

**Note**: This assumes you have an existing eSpeak-ng Xcode project. If you're starting from scratch, you may need to first create an Xcode project that wraps the eSpeak-ng C library, similar to how the iOS version was created. 
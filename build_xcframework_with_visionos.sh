#!/bin/bash
set -euo pipefail

FRAMEWORK_NAME="ESpeakNG" 
SCHEME_NAME="espeak-ng" 
CONFIGURATION="Release"
OUTPUT_DIR="${PROJECT_DIR}/../Frameworks" 

# Derived archive paths
MACOS_DEVICE_ARCHIVE_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}-macos.xcarchive"
IOS_DEVICE_ARCHIVE_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}-ios.xcarchive"
IOS_SIMULATOR_ARCHIVE_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}-simulator.xcarchive"
VISIONOS_DEVICE_ARCHIVE_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}-visionos.xcarchive"
VISIONOS_SIMULATOR_ARCHIVE_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}-visionos-simulator.xcarchive"
XCFRAMEWORK_OUTPUT_PATH="${OUTPUT_DIR}/${FRAMEWORK_NAME}.xcframework"

echo "🧹 Cleaning previous build artifacts..."
rm -rf "${MACOS_DEVICE_ARCHIVE_PATH}" "${IOS_DEVICE_ARCHIVE_PATH}" "${IOS_SIMULATOR_ARCHIVE_PATH}" "${VISIONOS_DEVICE_ARCHIVE_PATH}" "${VISIONOS_SIMULATOR_ARCHIVE_PATH}" "${XCFRAMEWORK_OUTPUT_PATH}"

echo "📦 Archiving for macOS device..."
xcodebuild archive \
  -scheme "${SCHEME_NAME}" \
  -configuration "${CONFIGURATION}" \
  -destination "generic/platform=macOS" \
  -archivePath "${MACOS_DEVICE_ARCHIVE_PATH}" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

echo "📱 Archiving for iOS device..."
xcodebuild archive \
  -scheme "${SCHEME_NAME}" \
  -configuration "${CONFIGURATION}" \
  -destination "generic/platform=iOS" \
  -archivePath "${IOS_DEVICE_ARCHIVE_PATH}" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

echo "📱 Archiving for iOS Simulator..."
xcodebuild archive \
  -scheme "${SCHEME_NAME}" \
  -configuration "${CONFIGURATION}" \
  -destination "generic/platform=iOS Simulator" \
  -archivePath "${IOS_SIMULATOR_ARCHIVE_PATH}" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

echo "🥽 Archiving for visionOS device..."
xcodebuild archive \
  -scheme "${SCHEME_NAME}" \
  -configuration "${CONFIGURATION}" \
  -destination "generic/platform=visionOS" \
  -archivePath "${VISIONOS_DEVICE_ARCHIVE_PATH}" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

echo "🥽 Archiving for visionOS Simulator..."
xcodebuild archive \
  -scheme "${SCHEME_NAME}" \
  -configuration "${CONFIGURATION}" \
  -destination "generic/platform=visionOS Simulator" \
  -archivePath "${VISIONOS_SIMULATOR_ARCHIVE_PATH}" \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES

echo "🔨 Creating XCFramework with visionOS support..."
xcodebuild -create-xcframework \
  -framework "${IOS_DEVICE_ARCHIVE_PATH}/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
  -framework "${IOS_SIMULATOR_ARCHIVE_PATH}/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
  -framework "${MACOS_DEVICE_ARCHIVE_PATH}/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
  -framework "${VISIONOS_DEVICE_ARCHIVE_PATH}/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
  -framework "${VISIONOS_SIMULATOR_ARCHIVE_PATH}/Products/Library/Frameworks/${FRAMEWORK_NAME}.framework" \
  -output "${XCFRAMEWORK_OUTPUT_PATH}"

echo "🧹 Cleaning up archive files..."
rm -rf "${IOS_DEVICE_ARCHIVE_PATH}" "${IOS_SIMULATOR_ARCHIVE_PATH}" "${MACOS_DEVICE_ARCHIVE_PATH}" "${VISIONOS_DEVICE_ARCHIVE_PATH}" "${VISIONOS_SIMULATOR_ARCHIVE_PATH}"

echo "✅ XCFramework with visionOS support successfully created at: ${XCFRAMEWORK_OUTPUT_PATH}"
echo ""
echo "🎯 Supported platforms:"
echo "  • macOS (arm64 + x86_64)"
echo "  • iOS (arm64)"
echo "  • iOS Simulator (arm64)"
echo "  • visionOS (arm64)"
echo "  • visionOS Simulator (arm64)" 
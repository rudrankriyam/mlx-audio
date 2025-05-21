//
//  Kokoro-tts-lib
//

import Foundation
import MLX
import MLXNN

class KokoroWeightLoader {
  private init() {}

  static let resourceName = "kokoro-v1_0"
  static let resourceType = "safetensors"
  static let hubRepoId = "prince-canuma/Kokoro-82M"

  static func loadWeights() -> [String: MLXArray] {
    var weightsFileURL: URL?
    var operationError: Error?
    let semaphore = DispatchSemaphore(value: 0)

    Task {
      defer { semaphore.signal() }
      do {
        weightsFileURL = try await getWeightsFileURL()
      } catch {
        operationError = error
      }
    }

    semaphore.wait()

    if let error = operationError {
      fatalError("Failed to obtain weights file URL: \(error.localizedDescription)")
    }

    guard let finalURL = weightsFileURL else {
      fatalError("Weight file URL could not be determined.")
    }

    print("Proceeding to load and process weights from URL: \(finalURL.path)")
    do {
      return try self.loadAndProcessWeights(from: finalURL)
    } catch {
      fatalError("Failed to load and process weights from \(finalURL.path): \(error.localizedDescription)")
    }
  }

  private static func getWeightsFileURL() async throws -> URL {
    let fileName = "\(resourceName).\(resourceType)"

    if let localPath = Bundle.main.path(forResource: resourceName, ofType: resourceType) {
      print("Found local weights at: \(localPath)")
      return URL(fileURLWithPath: localPath)
    } else {
      print(
        "Local weights file '\(fileName)' not found in bundle. Attempting to download from Hub...")

      let repo = Hub.Repo(id: hubRepoId)

      print("Starting download of \(fileName) from repo \(repo.id)")

      let modelDirectory: URL = try await Hub.snapshot(
        from: repo,
        matching: [fileName],
        progressHandler: { progress in
          let percentage = (progress.fractionCompleted) * 100
          print(String(format: "Download progress for \(fileName): %.2f%%", percentage))
        }
      )

      let downloadedFileUrl = modelDirectory.appendingPathComponent(fileName)

      print("Weights downloaded to directory: \(modelDirectory.path). Expected file at: \(downloadedFileUrl.path)")

      if FileManager.default.fileExists(atPath: downloadedFileUrl.path) {
        print("Successfully verified downloaded file exists at: \(downloadedFileUrl.path)")
        return downloadedFileUrl
      } else {
        var directoryContentsMessage =
          "Contents of download directory '\(modelDirectory.path)' could not be determined or directory is empty."
        if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelDirectory.path), !contents.isEmpty {
          directoryContentsMessage =
            "Contents of download directory '\(modelDirectory.path)\': \(contents.joined(separator: ", "))"
        } else if let contents = try? FileManager.default.contentsOfDirectory(atPath: modelDirectory.path), contents.isEmpty {
          directoryContentsMessage = "Download directory '\(modelDirectory.path)' is empty."
        }

        throw NSError(
          domain: "WeightLoaderError", code: 1001,
          userInfo: [
            NSLocalizedDescriptionKey:
              "Downloaded weight file '\(fileName)' not found at expected location: \(downloadedFileUrl.path). \(directoryContentsMessage)"
          ])
      }
    }
  }

  private static func loadAndProcessWeights(from url: URL) throws -> [String: MLXArray] {
    let weights = try MLX.loadArrays(url: url)
    var sanitizedWeights: [String: MLXArray] = [:]

    for (key, value) in weights {
      if key.hasPrefix("bert") {
        if key.contains("position_ids") {
          continue
        }
        sanitizedWeights[key] = value
      } else if key.hasPrefix("predictor") {
        if key.contains("F0_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("N_proj.weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      } else if key.hasPrefix("text_encoder") {
        if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      } else if key.hasPrefix("decoder") {
        if key.contains("noise_convs"), key.hasSuffix(".weight") {
          sanitizedWeights[key] = value.transposed(0, 2, 1)
        } else if key.contains("weight_v") {
          if checkArrayShape(arr: value) {
            sanitizedWeights[key] = value
          } else {
            sanitizedWeights[key] = value.transposed(0, 2, 1)
          }
        } else {
          sanitizedWeights[key] = value
        }
      }
    }
    return sanitizedWeights
  }

  private static func checkArrayShape(arr: MLXArray) -> Bool {
    guard arr.shape.count != 3 else { return false }

    let outChannels = arr.shape[0]
    let kH = arr.shape[1]
    let kW = arr.shape[2]

    return (outChannels >= kH) && (outChannels >= kW) && (kH == kW)
  }
}

//
//  HubApi.swift
//
//
//  Created by Pedro Cuenca on 20231230.
//

import CryptoKit
import Foundation
import Network
import os

public struct HubApi {
    var downloadBase: URL
    var hfToken: String?
    var endpoint: String
    var useBackgroundSession: Bool
    var useOfflineMode: Bool?

    private let networkMonitor = NetworkMonitor()
    public typealias RepoType = Hub.RepoType
    public typealias Repo = Hub.Repo

    public init(downloadBase: URL? = nil, hfToken: String? = nil, endpoint: String = "https://huggingface.co", useBackgroundSession: Bool = false, useOfflineMode: Bool? = nil) {
        self.hfToken = hfToken ?? Self.hfTokenFromEnv()
        if let downloadBase {
            self.downloadBase = downloadBase
        } else {
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            self.downloadBase = documents.appending(component: "huggingface")
        }
        self.endpoint = endpoint
        self.useBackgroundSession = useBackgroundSession
        self.useOfflineMode = useOfflineMode
        NetworkMonitor.shared.startMonitoring()
    }

    let sha256Pattern = "^[0-9a-f]{64}$"
    let commitHashPattern = "^[0-9a-f]{40}$"

    public static let shared = HubApi()

    private static let logger = Logger()
}

private extension HubApi {
    static func hfTokenFromEnv() -> String? {
        let possibleTokens = [
            { ProcessInfo.processInfo.environment["HF_TOKEN"] },
            { ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"] },
            {
                ProcessInfo.processInfo.environment["HF_TOKEN_PATH"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath),
                        encoding: .utf8
                    )
                }
            },
            {
                ProcessInfo.processInfo.environment["HF_HOME"].flatMap {
                    try? String(
                        contentsOf: URL(filePath: NSString(string: $0).expandingTildeInPath).appending(path: "token"),
                        encoding: .utf8
                    )
                }
            },
            { try? String(contentsOf: .homeDirectory.appendingPathComponent(".cache/huggingface/token"), encoding: .utf8) },
            { try? String(contentsOf: .homeDirectory.appendingPathComponent(".huggingface/token"), encoding: .utf8) },
        ]
        return possibleTokens
            .lazy
            .compactMap { $0() }
            .filter { !$0.isEmpty }
            .first
    }
}

/// File retrieval
public extension HubApi {
    /// Model data for parsed filenames
    struct Sibling: Codable {
        let rfilename: String
    }

    struct SiblingsResponse: Codable {
        let siblings: [Sibling]
    }

    /// Throws error if the response code is not 20X
    func httpGet(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        if let hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw Hub.HubClientError.unexpectedError
            }

            switch httpResponse.statusCode {
            case 200..<300:
                return (data, httpResponse)
            case 401, 403:
                throw Hub.HubClientError.authorizationRequired
            case 404:
                throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
            default:
                throw Hub.HubClientError.httpStatusCode(httpResponse.statusCode)
            }
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            throw Hub.HubClientError.downloadError(error.localizedDescription)
        }
    }

    /// Throws error if page does not exist or is not accessible.
    /// Allows relative redirects but ignores absolute ones for LFS files.
    func httpHead(for url: URL) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        if let hfToken {
            request.setValue("Bearer \(hfToken)", forHTTPHeaderField: "Authorization")
        }
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")

        let redirectDelegate = RedirectDelegate()
        let session = URLSession(configuration: .default, delegate: redirectDelegate, delegateQueue: nil)

        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Hub.HubClientError.unexpectedError }

        switch response.statusCode {
        case 200..<400: break // Allow redirects to pass through to the redirect delegate
        case 400..<500: throw Hub.HubClientError.authorizationRequired
        default: throw Hub.HubClientError.httpStatusCode(response.statusCode)
        }

        return (data, response)
    }

    func getFilenames(from repo: Repo, matching globs: [String] = []) async throws -> [String] {
        // Read repo info and only parse "siblings"
        let url = URL(string: "\(endpoint)/api/\(repo.type)/\(repo.id)")!
        let (data, _) = try await httpGet(for: url)
        let response = try JSONDecoder().decode(SiblingsResponse.self, from: data)
        let filenames = response.siblings.map { $0.rfilename }
        guard globs.count > 0 else { return filenames }

        var selected: Set<String> = []
        for glob in globs {
            selected = selected.union(filenames.matching(glob: glob))
        }
        return Array(selected)
    }

    func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        try await getFilenames(from: Repo(id: repoId), matching: globs)
    }

    func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await getFilenames(from: repo, matching: [glob])
    }

    func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await getFilenames(from: Repo(id: repoId), matching: [glob])
    }
}

/// Additional Errors
public extension HubApi {
    enum EnvironmentError: LocalizedError {
        case invalidMetadataError(String)
        case offlineModeError(String)
        case fileIntegrityError(String)
        case fileWriteError(String)

        public var errorDescription: String? {
            switch self {
            case let .invalidMetadataError(message):
                String(localized: "Invalid metadata: \(message)")
            case let .offlineModeError(message):
                String(localized: "Offline mode error: \(message)")
            case let .fileIntegrityError(message):
                String(localized: "File integrity check failed: \(message)")
            case let .fileWriteError(message):
                String(localized: "Failed to write file: \(message)")
            }
        }
    }
}

/// Configuration loading helpers
public extension HubApi {
    /// Assumes the file has already been downloaded.
    /// `filename` is relative to the download base.
    func configuration(from filename: String, in repo: Repo) throws -> Config {
        let fileURL = localRepoLocation(repo).appending(path: filename)
        return try configuration(fileURL: fileURL)
    }

    /// Assumes the file is already present at local url.
    /// `fileURL` is a complete local file path for the given model
    func configuration(fileURL: URL) throws -> Config {
        let data = try Data(contentsOf: fileURL)
        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Whoami
public extension HubApi {
    func whoami() async throws -> Config {
        guard hfToken != nil else { throw Hub.HubClientError.authorizationRequired }

        let url = URL(string: "\(endpoint)/api/whoami-v2")!
        let (data, _) = try await httpGet(for: url)

        let parsed = try JSONSerialization.jsonObject(with: data, options: [])
        guard let dictionary = parsed as? [NSString: Any] else { throw Hub.HubClientError.parse }
        return Config(dictionary)
    }
}

/// Snaphsot download
public extension HubApi {
    func localRepoLocation(_ repo: Repo) -> URL {
        downloadBase.appending(component: repo.type.rawValue).appending(component: repo.id)
    }

    /// Reads metadata about a file in the local directory related to a download process.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L263
    ///
    /// - Parameters:
    ///   - localDir: The local directory where metadata files are downloaded.
    ///   - filePath: The path of the file for which metadata is being read.
    /// - Throws: An `EnvironmentError.invalidMetadataError` if the metadata file is invalid and cannot be removed.
    /// - Returns: A `LocalDownloadFileMetadata` object if the metadata file exists and is valid, or `nil` if the file is missing or invalid.
    func readDownloadMetadata(metadataPath: URL) throws -> LocalDownloadFileMetadata? {
        if FileManager.default.fileExists(atPath: metadataPath.path) {
            do {
                let contents = try String(contentsOf: metadataPath, encoding: .utf8)
                let lines = contents.components(separatedBy: .newlines)

                guard lines.count >= 3 else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Metadata file is missing required fields"))
                }

                let commitHash = lines[0].trimmingCharacters(in: .whitespacesAndNewlines)
                let etag = lines[1].trimmingCharacters(in: .whitespacesAndNewlines)

                guard let timestamp = Double(lines[2].trimmingCharacters(in: .whitespacesAndNewlines)) else {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Invalid timestamp format"))
                }

                let timestampDate = Date(timeIntervalSince1970: timestamp)
                let filename = metadataPath.lastPathComponent.replacingOccurrences(of: ".metadata", with: "")

                return LocalDownloadFileMetadata(commitHash: commitHash, etag: etag, filename: filename, timestamp: timestampDate)
            } catch let error as EnvironmentError {
                do {
                    HubApi.logger.warning("Invalid metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            } catch {
                do {
                    HubApi.logger.warning("Error reading metadata file \(metadataPath): \(error.localizedDescription). Removing it from disk and continuing.")
                    try FileManager.default.removeItem(at: metadataPath)
                } catch {
                    throw EnvironmentError.invalidMetadataError(String(localized: "Could not remove corrupted metadata file: \(error.localizedDescription)"))
                }
                return nil
            }
        }

        // metadata file does not exist
        return nil
    }

    func isValidHash(hash: String, pattern: String) -> Bool {
        let regex = try? NSRegularExpression(pattern: pattern)
        let range = NSRange(location: 0, length: hash.utf16.count)
        return regex?.firstMatch(in: hash, options: [], range: range) != nil
    }

    func computeFileHash(file url: URL) throws -> String {
        // Open file for reading
        guard let fileHandle = try? FileHandle(forReadingFrom: url) else {
            throw Hub.HubClientError.fileNotFound(url.lastPathComponent)
        }

        defer {
            try? fileHandle.close()
        }

        var hasher = SHA256()
        let chunkSize = 1024 * 1024 // 1MB chunks

        while autoreleasepool(invoking: {
            let nextChunk = try? fileHandle.read(upToCount: chunkSize)

            guard let nextChunk,
                  !nextChunk.isEmpty
            else {
                return false
            }

            hasher.update(data: nextChunk)

            return true
        }) { }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/_local_folder.py#L391
    func writeDownloadMetadata(commitHash: String, etag: String, metadataPath: URL) throws {
        let metadataContent = "\(commitHash)\n\(etag)\n\(Date().timeIntervalSince1970)\n"
        do {
            try FileManager.default.createDirectory(at: metadataPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            try metadataContent.write(to: metadataPath, atomically: true, encoding: .utf8)
        } catch {
            throw EnvironmentError.fileWriteError(String(localized: "Failed to write metadata to \(metadataPath.path): \(error.localizedDescription)"))
        }
    }

    struct HubFileDownloader {
        let hub: HubApi
        let repo: Repo
        let repoDestination: URL
        let repoMetadataDestination: URL
        let relativeFilename: String
        let hfToken: String?
        let endpoint: String?
        let backgroundSession: Bool

        var source: URL {
            // https://huggingface.co/coreml-projects/Llama-2-7b-chat-coreml/resolve/main/tokenizer.json?download=true
            var url = URL(string: endpoint ?? "https://huggingface.co")!
            if repo.type != .models {
                url = url.appending(component: repo.type.rawValue)
            }
            url = url.appending(path: repo.id)
            url = url.appending(path: "resolve/main") // TODO: revisions
            url = url.appending(path: relativeFilename)
            return url
        }

        var destination: URL {
            repoDestination.appending(path: relativeFilename)
        }

        var metadataDestination: URL {
            repoMetadataDestination.appending(path: relativeFilename + ".metadata")
        }

        var downloaded: Bool {
            FileManager.default.fileExists(atPath: destination.path)
        }

        /// We're using incomplete destination to prepare cache destination because incomplete files include lfs + non-lfs files (vs only lfs for metadata files)
        func prepareCacheDestination(_ incompleteDestination: URL) throws {
            let directoryURL = incompleteDestination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
            if !FileManager.default.fileExists(atPath: incompleteDestination.path) {
                try "".write(to: incompleteDestination, atomically: true, encoding: .utf8)
            }
        }

        /// Note we go from Combine in Downloader to callback-based progress reporting
        /// We'll probably need to support Combine as well to play well with Swift UI
        /// (See for example PipelineLoader in swift-coreml-diffusers)
        @discardableResult
        func download(progressHandler: @escaping (Double) -> Void) async throws -> URL {
            let localMetadata = try hub.readDownloadMetadata(metadataPath: metadataDestination)
            let remoteMetadata = try await hub.getFileMetadata(url: source)

            let localCommitHash = localMetadata?.commitHash ?? ""
            let remoteCommitHash = remoteMetadata.commitHash ?? ""

            // Local file exists + metadata exists + commit_hash matches => return file
            if hub.isValidHash(hash: remoteCommitHash, pattern: hub.commitHashPattern), downloaded, localMetadata != nil, localCommitHash == remoteCommitHash {
                return destination
            }

            // From now on, etag, commit_hash, url and size are not empty
            guard let remoteCommitHash = remoteMetadata.commitHash,
                  let remoteEtag = remoteMetadata.etag,
                  let remoteSize = remoteMetadata.size,
                  remoteMetadata.location != ""
            else {
                throw EnvironmentError.invalidMetadataError("File metadata must have been retrieved from server")
            }

            // Local file exists => check if it's up-to-date
            if downloaded {
                // etag matches => update metadata and return file
                if localMetadata?.etag == remoteEtag {
                    try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                    return destination
                }

                // etag is a sha256
                // => means it's an LFS file (large)
                // => let's compute local hash and compare
                // => if match, update metadata and return file
                if hub.isValidHash(hash: remoteEtag, pattern: hub.sha256Pattern) {
                    let fileHash = try hub.computeFileHash(file: destination)
                    if fileHash == remoteEtag {
                        try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)
                        return destination
                    }
                }
            }

            // Otherwise, let's download the file!
            let incompleteDestination = repoMetadataDestination.appending(path: relativeFilename + ".\(remoteEtag).incomplete")
            try prepareCacheDestination(incompleteDestination)

            let downloader = Downloader(
                from: source,
                to: destination,
                incompleteDestination: incompleteDestination,
                using: hfToken,
                inBackground: backgroundSession,
                expectedSize: remoteSize
            )

            return try await withTaskCancellationHandler {
                let downloadSubscriber = downloader.downloadState.sink { state in
                    switch state {
                    case let .downloading(progress):
                        progressHandler(progress)
                    case .completed, .failed, .notStarted:
                        break
                    }
                }
                do {
                    _ = try withExtendedLifetime(downloadSubscriber) {
                        try downloader.waitUntilDone()
                    }

                    try hub.writeDownloadMetadata(commitHash: remoteCommitHash, etag: remoteEtag, metadataPath: metadataDestination)

                    return destination
                } catch {
                    // If download fails, leave the incomplete file in place for future resume
                    throw error
                }
            } onCancel: {
                downloader.cancel()
            }
        }
    }

    @discardableResult
    func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        let repoDestination = localRepoLocation(repo)
        let repoMetadataDestination = repoDestination
            .appendingPathComponent(".cache")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("download")

        if useOfflineMode ?? NetworkMonitor.shared.shouldUseOfflineMode() {
            if !FileManager.default.fileExists(atPath: repoDestination.path) {
                throw EnvironmentError.offlineModeError(String(localized: "Repository not available locally"))
            }

            let fileUrls = try FileManager.default.getFileUrls(at: repoDestination)
            if fileUrls.isEmpty {
                throw EnvironmentError.offlineModeError(String(localized: "No files available locally for this repository"))
            }

            for fileUrl in fileUrls {
                let metadataPath = URL(fileURLWithPath: fileUrl.path.replacingOccurrences(
                    of: repoDestination.path,
                    with: repoMetadataDestination.path
                ) + ".metadata")

                let localMetadata = try readDownloadMetadata(metadataPath: metadataPath)

                guard let localMetadata else {
                    throw EnvironmentError.offlineModeError(String(localized: "Metadata not available for \(fileUrl.lastPathComponent)"))
                }
                let localEtag = localMetadata.etag

                // LFS file so check file integrity
                if isValidHash(hash: localEtag, pattern: sha256Pattern) {
                    let fileHash = try computeFileHash(file: fileUrl)
                    if fileHash != localEtag {
                        throw EnvironmentError.fileIntegrityError(String(localized: "Hash mismatch for \(fileUrl.lastPathComponent)"))
                    }
                }
            }

            return repoDestination
        }

        let filenames = try await getFilenames(from: repo, matching: globs)
        let progress = Progress(totalUnitCount: Int64(filenames.count))
        for filename in filenames {
            let fileProgress = Progress(totalUnitCount: 100, parent: progress, pendingUnitCount: 1)
            let downloader = HubFileDownloader(
                hub: self,
                repo: repo,
                repoDestination: repoDestination,
                repoMetadataDestination: repoMetadataDestination,
                relativeFilename: filename,
                hfToken: hfToken,
                endpoint: endpoint,
                backgroundSession: useBackgroundSession
            )
            try await downloader.download { fractionDownloaded in
                fileProgress.completedUnitCount = Int64(100 * fractionDownloaded)
                progressHandler(progress)
            }
            fileProgress.completedUnitCount = 100
        }
        progressHandler(progress)
        return repoDestination
    }

    @discardableResult
    func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: repo, matching: [glob], progressHandler: progressHandler)
    }

    @discardableResult
    func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await snapshot(from: Repo(id: repoId), matching: [glob], progressHandler: progressHandler)
    }
}

/// Metadata
public extension HubApi {
    /// Data structure containing information about a file versioned on the Hub
    struct FileMetadata {
        /// The commit hash related to the file
        public let commitHash: String?

        /// Etag of the file on the server
        public let etag: String?

        /// Location where to download the file. Can be a Hub url or not (CDN).
        public let location: String

        /// Size of the file. In case of an LFS file, contains the size of the actual LFS file, not the pointer.
        public let size: Int?
    }

    /// Metadata about a file in the local directory related to a download process
    struct LocalDownloadFileMetadata {
        /// Commit hash of the file in the repo
        public let commitHash: String

        /// ETag of the file in the repo. Used to check if the file has changed.
        /// For LFS files, this is the sha256 of the file. For regular files, it corresponds to the git hash.
        public let etag: String

        /// Path of the file in the repo
        public let filename: String

        /// The timestamp of when the metadata was saved i.e. when the metadata was accurate
        public let timestamp: Date
    }

    private func normalizeEtag(_ etag: String?) -> String? {
        guard let etag else { return nil }
        return etag.trimmingPrefix("W/").trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    }

    func getFileMetadata(url: URL) async throws -> FileMetadata {
        let (_, response) = try await httpHead(for: url)
        let location = response.statusCode == 302 ? response.value(forHTTPHeaderField: "Location") : response.url?.absoluteString

        return FileMetadata(
            commitHash: response.value(forHTTPHeaderField: "X-Repo-Commit"),
            etag: normalizeEtag(
                (response.value(forHTTPHeaderField: "X-Linked-Etag")) ?? (response.value(forHTTPHeaderField: "Etag"))
            ),
            location: location ?? url.absoluteString,
            size: Int(response.value(forHTTPHeaderField: "X-Linked-Size") ?? response.value(forHTTPHeaderField: "Content-Length") ?? "")
        )
    }

    func getFileMetadata(from repo: Repo, matching globs: [String] = []) async throws -> [FileMetadata] {
        let files = try await getFilenames(from: repo, matching: globs)
        let url = URL(string: "\(endpoint)/\(repo.id)/resolve/main")! // TODO: revisions
        var selectedMetadata: [FileMetadata] = []
        for file in files {
            let fileURL = url.appending(path: file)
            try await selectedMetadata.append(getFileMetadata(url: fileURL))
        }
        return selectedMetadata
    }

    func getFileMetadata(from repoId: String, matching globs: [String] = []) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), matching: globs)
    }

    func getFileMetadata(from repo: Repo, matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: repo, matching: [glob])
    }

    func getFileMetadata(from repoId: String, matching glob: String) async throws -> [FileMetadata] {
        try await getFileMetadata(from: Repo(id: repoId), matching: [glob])
    }
}

/// Network monitor helper class to help decide whether to use offline mode
private extension HubApi {
    private final class NetworkMonitor {
        private var monitor: NWPathMonitor
        private var queue: DispatchQueue

        private(set) var isConnected: Bool = false
        private(set) var isExpensive: Bool = false
        private(set) var isConstrained: Bool = false

        static let shared = NetworkMonitor()

        init() {
            monitor = NWPathMonitor()
            queue = DispatchQueue(label: "HubApi.NetworkMonitor")
            startMonitoring()
        }

        func startMonitoring() {
            monitor.pathUpdateHandler = { [weak self] path in
                guard let self else { return }

                isConnected = path.status == .satisfied
                isExpensive = path.isExpensive
                isConstrained = path.isConstrained
            }

            monitor.start(queue: queue)
        }

        func stopMonitoring() {
            monitor.cancel()
        }

        func shouldUseOfflineMode() -> Bool {
            !isConnected || isExpensive || isConstrained
        }

        deinit {
            stopMonitoring()
        }
    }
}

/// Stateless wrappers that use `HubApi` instances
public extension Hub {
    static func getFilenames(from repo: Hub.Repo, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: globs)
    }

    static func getFilenames(from repoId: String, matching globs: [String] = []) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: globs)
    }

    static func getFilenames(from repo: Repo, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: repo, matching: glob)
    }

    static func getFilenames(from repoId: String, matching glob: String) async throws -> [String] {
        try await HubApi.shared.getFilenames(from: Repo(id: repoId), matching: glob)
    }

    static func snapshot(from repo: Repo, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching globs: [String] = [], progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: globs, progressHandler: progressHandler)
    }

    static func snapshot(from repo: Repo, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: repo, matching: glob, progressHandler: progressHandler)
    }

    static func snapshot(from repoId: String, matching glob: String, progressHandler: @escaping (Progress) -> Void = { _ in }) async throws -> URL {
        try await HubApi.shared.snapshot(from: Repo(id: repoId), matching: glob, progressHandler: progressHandler)
    }

    static func whoami(token: String) async throws -> Config {
        try await HubApi(hfToken: token).whoami()
    }

    static func getFileMetadata(fileURL: URL) async throws -> HubApi.FileMetadata {
        try await HubApi.shared.getFileMetadata(url: fileURL)
    }

    static func getFileMetadata(from repo: Repo, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: repo, matching: globs)
    }

    static func getFileMetadata(from repoId: String, matching globs: [String] = []) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: globs)
    }

    static func getFileMetadata(from repo: Repo, matching glob: String) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: repo, matching: [glob])
    }

    static func getFileMetadata(from repoId: String, matching glob: String) async throws -> [HubApi.FileMetadata] {
        try await HubApi.shared.getFileMetadata(from: Repo(id: repoId), matching: [glob])
    }
}

public extension [String] {
    func matching(glob: String) -> [String] {
        filter { fnmatch(glob, $0, 0) == 0 }
    }
}

public extension FileManager {
    func getFileUrls(at directoryUrl: URL) throws -> [URL] {
        var fileUrls = [URL]()

        // Get all contents including subdirectories
        guard let enumerator = FileManager.default.enumerator(
            at: directoryUrl,
            includingPropertiesForKeys: [.isRegularFileKey, .isHiddenKey],
            options: [.skipsHiddenFiles]
        ) else {
            return fileUrls
        }

        for case let fileURL as URL in enumerator {
            do {
                let resourceValues = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .isHiddenKey])
                if resourceValues.isRegularFile == true, resourceValues.isHidden != true {
                    fileUrls.append(fileURL)
                }
            } catch {
                throw error
            }
        }

        return fileUrls
    }
}

/// Only allow relative redirects and reject others
/// Reference: https://github.com/huggingface/huggingface_hub/blob/b2c9a148d465b43ab90fab6e4ebcbbf5a9df27d4/src/huggingface_hub/file_download.py#L258
private class RedirectDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        // Check if it's a redirect status code (300-399)
        if (300...399).contains(response.statusCode) {
            // Get the Location header
            if let locationString = response.value(forHTTPHeaderField: "Location"),
               let locationUrl = URL(string: locationString)
            {
                // Check if it's a relative redirect (no host component)
                if locationUrl.host == nil {
                    // For relative redirects, construct the new URL using the original request's base
                    if let originalUrl = task.originalRequest?.url,
                       var components = URLComponents(url: originalUrl, resolvingAgainstBaseURL: true)
                    {
                        // Update the path component with the relative path
                        components.path = locationUrl.path
                        components.query = locationUrl.query

                        // Create new request with the resolved URL
                        if let resolvedUrl = components.url {
                            var newRequest = URLRequest(url: resolvedUrl)
                            // Copy headers from original request
                            task.originalRequest?.allHTTPHeaderFields?.forEach { key, value in
                                newRequest.setValue(value, forHTTPHeaderField: key)
                            }
                            newRequest.setValue(resolvedUrl.absoluteString, forHTTPHeaderField: "Location")
                            completionHandler(newRequest)
                            return
                        }
                    }
                }
            }
        }

        // For all other cases (non-redirects or absolute redirects), prevent redirect
        completionHandler(nil)
    }
}

//
//  Downloader.swift
//
//  Adapted from https://github.com/huggingface/swift-coreml-diffusers/blob/d041577b9f5e201baa3465bc60bc5d0a1cf7ed7f/Diffusion/Common/Downloader.swift
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Combine
import Foundation

class Downloader: NSObject, ObservableObject {
    private(set) var destination: URL

    private let chunkSize = 10 * 1024 * 1024 // 10MB

    enum DownloadState {
        case notStarted
        case downloading(Double)
        case completed(URL)
        case failed(Error)
    }

    enum DownloadError: Error {
        case invalidDownloadLocation
        case unexpectedError
        case tempFileNotFound
    }

    private(set) lazy var downloadState: CurrentValueSubject<DownloadState, Never> = CurrentValueSubject(.notStarted)
    private var stateSubscriber: Cancellable?

    private(set) var tempFilePath: URL
    private(set) var expectedSize: Int?
    private(set) var downloadedSize: Int = 0

    var session: URLSession? = nil
    var downloadTask: Task<Void, Error>? = nil

    init(
        from url: URL,
        to destination: URL,
        incompleteDestination: URL,
        using authToken: String? = nil,
        inBackground: Bool = false,
        headers: [String: String]? = nil,
        expectedSize: Int? = nil,
        timeout: TimeInterval = 10,
        numRetries: Int = 5
    ) {
        self.destination = destination
        self.expectedSize = expectedSize

        // Create incomplete file path based on destination
        tempFilePath = incompleteDestination

        // If resume size wasn't specified, check for an existing incomplete file
        let resumeSize = Self.incompleteFileSize(at: incompleteDestination)

        super.init()
        let sessionIdentifier = "swift-transformers.hub.downloader"

        var config = URLSessionConfiguration.default
        if inBackground {
            config = URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
            config.isDiscretionary = false
            config.sessionSendsLaunchEvents = true
        }

        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        setUpDownload(from: url, with: authToken, resumeSize: resumeSize, headers: headers, expectedSize: expectedSize, timeout: timeout, numRetries: numRetries)
    }

    /// Check if an incomplete file exists for the destination and returns its size
    /// - Parameter destination: The destination URL for the download
    /// - Returns: Size of the incomplete file if it exists, otherwise 0
    static func incompleteFileSize(at incompletePath: URL) -> Int {
        if FileManager.default.fileExists(atPath: incompletePath.path) {
            if let attributes = try? FileManager.default.attributesOfItem(atPath: incompletePath.path), let fileSize = attributes[.size] as? Int {
                return fileSize
            }
        }

        return 0
    }

    /// Sets up and initiates a file download operation
    ///
    /// - Parameters:
    ///   - url: Source URL to download from
    ///   - authToken: Bearer token for authentication with Hugging Face
    ///   - resumeSize: Number of bytes already downloaded for resuming interrupted downloads
    ///   - headers: Additional HTTP headers to include in the request
    ///   - expectedSize: Expected file size in bytes for validation
    ///   - timeout: Time interval before the request times out
    ///   - numRetries: Number of retry attempts for failed downloads
    private func setUpDownload(
        from url: URL,
        with authToken: String?,
        resumeSize: Int,
        headers: [String: String]?,
        expectedSize: Int?,
        timeout: TimeInterval,
        numRetries: Int
    ) {
        session?.getAllTasks { tasks in
            // If there's an existing pending background task with the same URL, let it proceed.
            if let existing = tasks.filter({ $0.originalRequest?.url == url }).first {
                switch existing.state {
                case .running:
                    return
                case .suspended:
                    existing.resume()
                    return
                case .canceling, .completed:
                    existing.cancel()
                @unknown default:
                    existing.cancel()
                }
            }

            self.downloadTask = Task {
                do {
                    // Set up the request with appropriate headers
                    var request = URLRequest(url: url)
                    var requestHeaders = headers ?? [:]

                    if let authToken {
                        requestHeaders["Authorization"] = "Bearer \(authToken)"
                    }

                    self.downloadedSize = resumeSize

                    // Set Range header if we're resuming
                    if resumeSize > 0 {
                        requestHeaders["Range"] = "bytes=\(resumeSize)-"

                        // Calculate and show initial progress
                        if let expectedSize, expectedSize > 0 {
                            let initialProgress = Double(resumeSize) / Double(expectedSize)
                            self.downloadState.value = .downloading(initialProgress)
                        } else {
                            self.downloadState.value = .downloading(0)
                        }
                    } else {
                        self.downloadState.value = .downloading(0)
                    }

                    request.timeoutInterval = timeout
                    request.allHTTPHeaderFields = requestHeaders

                    // Open the incomplete file for writing
                    let tempFile = try FileHandle(forWritingTo: self.tempFilePath)

                    // If resuming, seek to end of file
                    if resumeSize > 0 {
                        try tempFile.seekToEnd()
                    }

                    try await self.httpGet(request: request, tempFile: tempFile, resumeSize: self.downloadedSize, numRetries: numRetries, expectedSize: expectedSize)

                    // Clean up and move the completed download to its final destination
                    tempFile.closeFile()

                    try Task.checkCancellation()
                    try FileManager.default.moveDownloadedFile(from: self.tempFilePath, to: self.destination)
                    self.downloadState.value = .completed(self.destination)
                } catch {
                    self.downloadState.value = .failed(error)
                }
            }
        }
    }

    /// Downloads a file from given URL using chunked transfer and handles retries.
    ///
    /// Reference: https://github.com/huggingface/huggingface_hub/blob/418a6ffce7881f5c571b2362ed1c23ef8e4d7d20/src/huggingface_hub/file_download.py#L306
    ///
    /// - Parameters:
    ///   - request: The URLRequest for the file to download
    ///   - resumeSize: The number of bytes already downloaded. If set to 0 (default), the whole file is download. If set to a positive number, the download will resume at the given position
    ///   - numRetries: The number of retry attempts remaining for failed downloads
    ///   - expectedSize: The expected size of the file to download. If set, the download will raise an error if the size of the received content is different from the expected one.
    /// - Throws: `DownloadError.unexpectedError` if the response is invalid or file size mismatch occurs
    ///           `URLError` if the download fails after all retries are exhausted
    private func httpGet(
        request: URLRequest,
        tempFile: FileHandle,
        resumeSize: Int,
        numRetries: Int,
        expectedSize: Int?
    ) async throws {
        guard let session else {
            throw DownloadError.unexpectedError
        }

        // Create a new request with Range header for resuming
        var newRequest = request
        if resumeSize > 0 {
            newRequest.setValue("bytes=\(resumeSize)-", forHTTPHeaderField: "Range")
        }

        // Start the download and get the byte stream
        let (asyncBytes, response) = try await session.bytes(for: newRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadError.unexpectedError
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw DownloadError.unexpectedError
        }

        // Create a buffer to collect bytes before writing to disk
        var buffer = Data(capacity: chunkSize)

        var newNumRetries = numRetries
        do {
            for try await byte in asyncBytes {
                buffer.append(byte)
                // When buffer is full, write to disk
                if buffer.count == chunkSize {
                    if !buffer.isEmpty { // Filter out keep-alive chunks
                        try tempFile.write(contentsOf: buffer)
                        buffer.removeAll(keepingCapacity: true)
                        downloadedSize += chunkSize
                        newNumRetries = 5
                        guard let expectedSize else { continue }
                        let progress = expectedSize != 0 ? Double(downloadedSize) / Double(expectedSize) : 0
                        downloadState.value = .downloading(progress)
                    }
                }
            }

            if !buffer.isEmpty {
                try tempFile.write(contentsOf: buffer)
                downloadedSize += buffer.count
                buffer.removeAll(keepingCapacity: true)
                newNumRetries = 5
            }
        } catch let error as URLError {
            if newNumRetries <= 0 {
                throw error
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)

            let config = URLSessionConfiguration.default
            self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)

            try await httpGet(
                request: request,
                tempFile: tempFile,
                resumeSize: self.downloadedSize,
                numRetries: newNumRetries - 1,
                expectedSize: expectedSize
            )
        }

        // Verify the downloaded file size matches the expected size
        let actualSize = try tempFile.seekToEnd()
        if let expectedSize, expectedSize != actualSize {
            throw DownloadError.unexpectedError
        }
    }

    @discardableResult
    func waitUntilDone() throws -> URL {
        // It's either this, or stream the bytes ourselves (add to a buffer, save to disk, etc; boring and finicky)
        let semaphore = DispatchSemaphore(value: 0)
        stateSubscriber = downloadState.sink { state in
            switch state {
            case .completed: semaphore.signal()
            case .failed: semaphore.signal()
            default: break
            }
        }
        semaphore.wait()

        switch downloadState.value {
        case let .completed(url): return url
        case let .failed(error): throw error
        default: throw DownloadError.unexpectedError
        }
    }

    func cancel() {
        session?.invalidateAndCancel()
        downloadTask?.cancel()
        downloadState.value = .failed(URLError(.cancelled))
    }
}

extension Downloader: URLSessionDownloadDelegate {
    func urlSession(_: URLSession, downloadTask: URLSessionDownloadTask, didWriteData _: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        downloadState.value = .downloading(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }

    func urlSession(_: URLSession, downloadTask _: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            // If the downloaded file already exists on the filesystem, overwrite it
            try FileManager.default.moveDownloadedFile(from: location, to: destination)
            downloadState.value = .completed(destination)
        } catch {
            downloadState.value = .failed(error)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            downloadState.value = .failed(error)
//        } else if let response = task.response as? HTTPURLResponse {
//            print("HTTP response status code: \(response.statusCode)")
//            let headers = response.allHeaderFields
//            print("HTTP response headers: \(headers)")
        }
    }
}

extension FileManager {
    func moveDownloadedFile(from srcURL: URL, to dstURL: URL) throws {
        if fileExists(atPath: dstURL.path()) {
            try removeItem(at: dstURL)
        }

        let directoryURL = dstURL.deletingLastPathComponent()
        try createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)

        try moveItem(at: srcURL, to: dstURL)
    }
}

//
//  Hub.swift
//
//
//  Created by Pedro Cuenca on 18/5/23.
//

import Foundation

public struct Hub { }

public extension Hub {
    enum HubClientError: LocalizedError {
        case authorizationRequired
        case httpStatusCode(Int)
        case parse
        case unexpectedError
        case downloadError(String)
        case fileNotFound(String)
        case networkError(URLError)
        case resourceNotFound(String)
        case configurationMissing(String)
        case fileSystemError(Error)
        case parseError(String)

        public var errorDescription: String? {
            switch self {
            case .authorizationRequired:
                String(localized: "Authentication required. Please provide a valid Hugging Face token.")
            case let .httpStatusCode(code):
                String(localized: "HTTP error with status code: \(code)")
            case .parse:
                String(localized: "Failed to parse server response.")
            case .unexpectedError:
                String(localized: "An unexpected error occurred.")
            case let .downloadError(message):
                String(localized: "Download failed: \(message)")
            case let .fileNotFound(filename):
                String(localized: "File not found: \(filename)")
            case let .networkError(error):
                String(localized: "Network error: \(error.localizedDescription)")
            case let .resourceNotFound(resource):
                String(localized: "Resource not found: \(resource)")
            case let .configurationMissing(file):
                String(localized: "Required configuration file missing: \(file)")
            case let .fileSystemError(error):
                String(localized: "File system error: \(error.localizedDescription)")
            case let .parseError(message):
                String(localized: "Parse error: \(message)")
            }
        }
    }

    enum RepoType: String, Codable {
        case models
        case datasets
        case spaces
    }

    struct Repo: Codable {
        public let id: String
        public let type: RepoType

        public init(id: String, type: RepoType = .models) {
            self.id = id
            self.type = type
        }
    }
}

// MARK: - Configuration files with dynamic lookup

@dynamicMemberLookup
public struct Config {
    public private(set) var dictionary: [NSString: Any]

    public init(_ dictionary: [NSString: Any]) {
        self.dictionary = dictionary
    }

    func camelCase(_ string: String) -> String {
        string
            .split(separator: "_")
            .enumerated()
            .map { $0.offset == 0 ? $0.element.lowercased() : $0.element.capitalized }
            .joined()
    }

    func uncamelCase(_ string: String) -> String {
        let scalars = string.unicodeScalars
        var result = ""

        var previousCharacterIsLowercase = false
        for scalar in scalars {
            if CharacterSet.uppercaseLetters.contains(scalar) {
                if previousCharacterIsLowercase {
                    result += "_"
                }
                let lowercaseChar = Character(scalar).lowercased()
                result += lowercaseChar
                previousCharacterIsLowercase = false
            } else {
                result += String(scalar)
                previousCharacterIsLowercase = true
            }
        }

        return result
    }

    public subscript(dynamicMember member: String) -> Config? {
        let key = (dictionary[member as NSString] != nil ? member : uncamelCase(member)) as NSString
        if let value = dictionary[key] as? [NSString: Any] {
            return Config(value)
        } else if let value = dictionary[key] {
            return Config(["value": value])
        }
        return nil
    }

    public var value: Any? {
        dictionary["value"]
    }

    public var intValue: Int? { value as? Int }
    public var boolValue: Bool? { value as? Bool }
    public var stringValue: String? { value as? String }

    /// Instead of doing this we could provide custom classes and decode to them
    public var arrayValue: [Config]? {
        guard let list = value as? [Any] else { return nil }
        return list.map { Config($0 as! [NSString: Any]) }
    }

    /// Tuple of token identifier and string value
    public var tokenValue: (UInt, String)? {
        guard let value = value as? [Any] else {
            return nil
        }
        guard let stringValue = value.first as? String, let intValue = value.dropFirst().first as? UInt else {
            return nil
        }
        return (intValue, stringValue)
    }
}

public class LanguageModelConfigurationFromHub {
    struct Configurations {
        var modelConfig: Config
        var tokenizerConfig: Config?
        var tokenizerData: Config
    }

    private var configPromise: Task<Configurations, Error>?

    public init(
        modelName: String,
        hubApi: HubApi = .shared
    ) {
        configPromise = Task.init {
            try await self.loadConfig(modelName: modelName, hubApi: hubApi)
        }
    }

    public init(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) {
        configPromise = Task {
            try await self.loadConfig(modelFolder: modelFolder, hubApi: hubApi)
        }
    }

    public var modelConfig: Config {
        get async throws {
            try await configPromise!.value.modelConfig
        }
    }

    public var tokenizerConfig: Config? {
        get async throws {
            if let hubConfig = try await configPromise!.value.tokenizerConfig {
                // Try to guess the class if it's not present and the modelType is
                if let _ = hubConfig.tokenizerClass?.stringValue { return hubConfig }
                guard let modelType = try await modelType else { return hubConfig }

                // If the config exists but doesn't contain a tokenizerClass, use a fallback config if we have it
                if let fallbackConfig = Self.fallbackTokenizerConfig(for: modelType) {
                    let configuration = fallbackConfig.dictionary.merging(hubConfig.dictionary, uniquingKeysWith: { current, _ in current })
                    return Config(configuration)
                }

                // Guess by capitalizing
                var configuration = hubConfig.dictionary
                configuration["tokenizer_class"] = "\(modelType.capitalized)Tokenizer"
                return Config(configuration)
            }

            // Fallback tokenizer config, if available
            guard let modelType = try await modelType else { return nil }
            return Self.fallbackTokenizerConfig(for: modelType)
        }
    }

    public var tokenizerData: Config {
        get async throws {
            try await configPromise!.value.tokenizerData
        }
    }

    public var modelType: String? {
        get async throws {
            try await modelConfig.modelType?.stringValue
        }
    }

    func loadConfig(
        modelName: String,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        let filesToDownload = ["config.json", "tokenizer_config.json", "chat_template.json", "tokenizer.json"]
        let repo = Hub.Repo(id: modelName)

        do {
            let downloadedModelFolder = try await hubApi.snapshot(from: repo, matching: filesToDownload)
            return try await loadConfig(modelFolder: downloadedModelFolder, hubApi: hubApi)
        } catch {
            // Convert generic errors to more specific ones
            if let urlError = error as? URLError {
                switch urlError.code {
                case .notConnectedToInternet, .networkConnectionLost:
                    throw Hub.HubClientError.networkError(urlError)
                case .resourceUnavailable:
                    throw Hub.HubClientError.resourceNotFound(modelName)
                default:
                    throw Hub.HubClientError.networkError(urlError)
                }
            } else {
                throw error
            }
        }
    }

    func loadConfig(
        modelFolder: URL,
        hubApi: HubApi = .shared
    ) async throws -> Configurations {
        do {
            // Load required configurations
            let modelConfigURL = modelFolder.appending(path: "config.json")
            guard FileManager.default.fileExists(atPath: modelConfigURL.path) else {
                throw Hub.HubClientError.configurationMissing("config.json")
            }

            let modelConfig = try hubApi.configuration(fileURL: modelConfigURL)

            let tokenizerDataURL = modelFolder.appending(path: "tokenizer.json")
            guard FileManager.default.fileExists(atPath: tokenizerDataURL.path) else {
                throw Hub.HubClientError.configurationMissing("tokenizer.json")
            }

            let tokenizerData = try hubApi.configuration(fileURL: tokenizerDataURL)

            // Load tokenizer config (optional)
            var tokenizerConfig: Config? = nil
            let tokenizerConfigURL = modelFolder.appending(path: "tokenizer_config.json")
            if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
                tokenizerConfig = try hubApi.configuration(fileURL: tokenizerConfigURL)
            }

            // Check for chat template and merge if available
            let chatTemplateURL = modelFolder.appending(path: "chat_template.json")
            if FileManager.default.fileExists(atPath: chatTemplateURL.path),
               let chatTemplateConfig = try? hubApi.configuration(fileURL: chatTemplateURL),
               let chatTemplate = chatTemplateConfig.chatTemplate?.stringValue
            {
                // Create or update tokenizer config with chat template
                if var configDict = tokenizerConfig?.dictionary {
                    configDict["chat_template"] = chatTemplate
                    tokenizerConfig = Config(configDict)
                } else {
                    tokenizerConfig = Config(["chat_template": chatTemplate])
                }
            }

            return Configurations(
                modelConfig: modelConfig,
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } catch let error as Hub.HubClientError {
            throw error
        } catch {
            if let nsError = error as NSError? {
                if nsError.domain == NSCocoaErrorDomain, nsError.code == NSFileReadNoSuchFileError {
                    throw Hub.HubClientError.fileSystemError(error)
                } else if nsError.domain == "NSJSONSerialization" {
                    throw Hub.HubClientError.parseError("Invalid JSON format: \(nsError.localizedDescription)")
                }
            }
            throw Hub.HubClientError.fileSystemError(error)
        }
    }

    static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            let parsed = try JSONSerialization.jsonObject(with: data, options: [])
            guard let dictionary = parsed as? [NSString: Any] else {
                throw Hub.HubClientError.parseError("Failed to parse fallback tokenizer config")
            }
            return Config(dictionary)
        } catch let error as Hub.HubClientError {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        } catch {
            print("Error loading fallback tokenizer config: \(error.localizedDescription)")
            return nil
        }
    }
}

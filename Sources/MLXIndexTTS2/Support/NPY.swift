// NPY.swift — minimal .npy reader for parity goldens (fp32 / fp16 / int32 / int64, C-order).
// Promoted into the main target so the gate CLI and any future fixtures share it.

import Foundation
import MLX

public enum NPY {

    public enum NPYError: Error, CustomStringConvertible {
        case badMagic, badHeader, unsupportedDescr(String), fortranOrder
        public var description: String {
            switch self {
            case .badMagic: return "not an NPY file"
            case .badHeader: return "unparseable NPY header"
            case .unsupportedDescr(let d): return "unsupported dtype descr \(d)"
            case .fortranOrder: return "fortran-order arrays unsupported"
            }
        }
    }

    /// Load a .npy file as an MLXArray (little-endian, C-order).
    public static func load(_ url: URL) throws -> MLXArray {
        let data = try Data(contentsOf: url)
        guard data.count > 10, data.prefix(6) == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw NPYError.badMagic
        }
        let major = data[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
            headerStart = 12
        }
        guard let header = String(data: data.subdata(in: headerStart..<headerStart + headerLen),
                                  encoding: .ascii) else { throw NPYError.badHeader }

        guard !header.contains("'fortran_order': True") else { throw NPYError.fortranOrder }

        guard let descrRange = header.range(of: #"'descr':\s*'([^']+)'"#, options: .regularExpression),
              let shapeRange = header.range(of: #"'shape':\s*\(([^)]*)\)"#, options: .regularExpression)
        else { throw NPYError.badHeader }

        let descr = String(header[descrRange]).components(separatedBy: "'").dropLast().last ?? ""
        let shapeBody = String(header[shapeRange])
            .components(separatedBy: "(").last!.components(separatedBy: ")").first!
        let shape = shapeBody.split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)

        let payload = data.subdata(in: (headerStart + headerLen)..<data.count)
        let finalShape = shape.isEmpty ? [1] : shape

        switch descr {
        case "<f4":
            let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Float32.self).prefix(count)) }
            return MLXArray(values, finalShape)
        case "<f2":
            let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Float16.self).prefix(count)) }
            return MLXArray(values.map { Float32($0) }, finalShape).asType(.float16)
        case "<i4":
            let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Int32.self).prefix(count)) }
            return MLXArray(values, finalShape)
        case "<i8":
            let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Int64.self).prefix(count)) }
            return MLXArray(values.map { Int32(clamping: $0) }, finalShape)
        default:
            throw NPYError.unsupportedDescr(descr)
        }
    }
}

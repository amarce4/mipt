#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mipt_io
{
    constexpr std::uint32_t RHO_FILE_VERSION = 2;
    constexpr std::uint32_t RHO_ENDIAN_MARKER = 0x01020304u;
    constexpr char RHO_FILE_MAGIC[8] = {'M', 'I', 'P', 'T', 'R', 'H', 'O', '2'};

    struct DensityFileMetadata
    {
        std::uint32_t spatial_dimension = 0; // 1 or 2
        std::uint32_t total_qubits = 0;
        std::uint32_t kept_qubits = 0;
        std::uint32_t matrix_dimension = 0;   // 2^kept_qubits
        std::uint32_t periods = 0;
        std::uint32_t realizations = 0;
        std::uint32_t resolution = 0;
        std::uint32_t grid_x = 0;             // total qubits for 1D; x-side for 2D
        std::uint32_t grid_y = 0;             // 1 for 1D; y-side for 2D
        std::uint32_t subsystem_count = 0;
        std::uint64_t record_count = 0;       // simulation trajectories, not RDM count
        double p_min = 0.0;
        double p_max = 0.0;

        // Flattened subsystem list. For subsystem s and local position k:
        // subsystem_qubits[s * kept_qubits + k] is the original CUDA-Q qubit.
        // Reduced-basis bit k corresponds to this listed qubit.
        std::vector<std::uint32_t> subsystem_qubits;
    };

    struct DensityRecord
    {
        std::uint32_t p_index = 0;
        std::uint32_t realization = 0;
        double p = 0.0;

        // Payload is ordered by subsystem, then row-major complex matrix:
        // offset = s * matrix_value_count(metadata) + 2 * (row * D + col)
        // offset+0 = real; offset+1 = imaginary.
        std::vector<double> rho_ri;
    };

    inline std::size_t matrix_value_count(const DensityFileMetadata &metadata)
    {
        const std::size_t d = static_cast<std::size_t>(metadata.matrix_dimension);
        return 2u * d * d;
    }

    inline std::size_t payload_value_count(const DensityFileMetadata &metadata)
    {
        return static_cast<std::size_t>(metadata.subsystem_count) *
               matrix_value_count(metadata);
    }

    inline void validate_metadata(const DensityFileMetadata &metadata)
    {
        if (metadata.kept_qubits == 0 || metadata.kept_qubits >= 31)
        {
            throw std::invalid_argument("Invalid kept-qubit count in density-matrix metadata.");
        }

        const std::uint32_t expected_dim = std::uint32_t{1} << metadata.kept_qubits;
        if (metadata.matrix_dimension != expected_dim)
        {
            throw std::invalid_argument("Density-matrix dimension does not match kept-qubit count.");
        }

        if (metadata.total_qubits < metadata.kept_qubits)
        {
            throw std::invalid_argument("Kept subsystem is larger than the simulated system.");
        }

        if (metadata.subsystem_count == 0)
        {
            throw std::invalid_argument("Density-matrix file contains no subsystems.");
        }

        const std::size_t expected_indices =
            static_cast<std::size_t>(metadata.subsystem_count) *
            static_cast<std::size_t>(metadata.kept_qubits);
        if (metadata.subsystem_qubits.size() != expected_indices)
        {
            throw std::invalid_argument("Subsystem-qubit metadata length is inconsistent.");
        }

        for (std::uint32_t s = 0; s < metadata.subsystem_count; ++s)
        {
            for (std::uint32_t i = 0; i < metadata.kept_qubits; ++i)
            {
                const std::uint32_t qi =
                    metadata.subsystem_qubits[s * metadata.kept_qubits + i];
                if (qi >= metadata.total_qubits)
                {
                    throw std::invalid_argument("Subsystem qubit index is out of range.");
                }
                for (std::uint32_t j = 0; j < i; ++j)
                {
                    if (qi == metadata.subsystem_qubits[s * metadata.kept_qubits + j])
                    {
                        throw std::invalid_argument("A retained subsystem contains duplicate qubits.");
                    }
                }
            }
        }
    }

    template <typename T>
    inline void write_scalar(std::ostream &stream, const T &value)
    {
        stream.write(reinterpret_cast<const char *>(&value), sizeof(T));
        if (!stream)
        {
            throw std::runtime_error("Failed while writing density-matrix file.");
        }
    }

    template <typename T>
    inline void read_scalar(std::istream &stream, T &value)
    {
        stream.read(reinterpret_cast<char *>(&value), sizeof(T));
        if (!stream)
        {
            throw std::runtime_error("Failed while reading density-matrix file.");
        }
    }

    class DensityMatrixWriter
    {
      public:
        DensityMatrixWriter(const std::string &path, const DensityFileMetadata &metadata)
            : metadata_(metadata),
              expected_values_(payload_value_count(metadata)),
              stream_(path, std::ios::binary | std::ios::trunc)
        {
            validate_metadata(metadata_);
            if (!stream_)
            {
                throw std::runtime_error("Could not create density-matrix file: " + path);
            }

            stream_.write(RHO_FILE_MAGIC, sizeof(RHO_FILE_MAGIC));
            write_scalar(stream_, RHO_ENDIAN_MARKER);
            write_scalar(stream_, RHO_FILE_VERSION);
            write_scalar(stream_, metadata_.spatial_dimension);
            write_scalar(stream_, metadata_.total_qubits);
            write_scalar(stream_, metadata_.kept_qubits);
            write_scalar(stream_, metadata_.matrix_dimension);
            write_scalar(stream_, metadata_.periods);
            write_scalar(stream_, metadata_.realizations);
            write_scalar(stream_, metadata_.resolution);
            write_scalar(stream_, metadata_.grid_x);
            write_scalar(stream_, metadata_.grid_y);
            write_scalar(stream_, metadata_.subsystem_count);
            write_scalar(stream_, metadata_.record_count);
            write_scalar(stream_, metadata_.p_min);
            write_scalar(stream_, metadata_.p_max);
            stream_.write(
                reinterpret_cast<const char *>(metadata_.subsystem_qubits.data()),
                static_cast<std::streamsize>(metadata_.subsystem_qubits.size() *
                                             sizeof(std::uint32_t)));
            if (!stream_)
            {
                throw std::runtime_error("Failed while writing subsystem-qubit metadata.");
            }
        }

        void write_record(std::uint32_t p_index,
                          std::uint32_t realization,
                          double p,
                          const double *rho_ri,
                          std::size_t count)
        {
            if (rho_ri == nullptr || count != expected_values_)
            {
                throw std::invalid_argument("Incorrect density-matrix record size.");
            }

            if (records_written_ >= metadata_.record_count)
            {
                throw std::runtime_error("Attempted to write too many density-matrix records.");
            }

            write_scalar(stream_, p_index);
            write_scalar(stream_, realization);
            write_scalar(stream_, p);
            stream_.write(reinterpret_cast<const char *>(rho_ri),
                          static_cast<std::streamsize>(count * sizeof(double)));
            if (!stream_)
            {
                throw std::runtime_error("Failed while writing density-matrix payload.");
            }
            ++records_written_;
        }

        void close()
        {
            if (closed_)
            {
                return;
            }
            if (records_written_ != metadata_.record_count)
            {
                throw std::runtime_error("Density-matrix file closed before all records were written.");
            }
            stream_.flush();
            if (!stream_)
            {
                throw std::runtime_error("Failed while flushing density-matrix file.");
            }
            stream_.close();
            closed_ = true;
        }

        ~DensityMatrixWriter()
        {
            if (!closed_)
            {
                stream_.close();
            }
        }

      private:
        DensityFileMetadata metadata_;
        std::size_t expected_values_ = 0;
        std::uint64_t records_written_ = 0;
        bool closed_ = false;
        std::ofstream stream_;
    };

    class DensityMatrixReader
    {
      public:
        explicit DensityMatrixReader(const std::string &path)
            : stream_(path, std::ios::binary)
        {
            if (!stream_)
            {
                throw std::runtime_error("Could not open density-matrix file: " + path);
            }

            char magic[sizeof(RHO_FILE_MAGIC)]{};
            stream_.read(magic, sizeof(magic));
            if (!stream_)
            {
                throw std::runtime_error("Could not read density-matrix file header.");
            }
            for (std::size_t i = 0; i < sizeof(RHO_FILE_MAGIC); ++i)
            {
                if (magic[i] != RHO_FILE_MAGIC[i])
                {
                    throw std::runtime_error("Not a version-2 MIPT density-matrix file.");
                }
            }

            std::uint32_t endian_marker = 0;
            std::uint32_t version = 0;
            read_scalar(stream_, endian_marker);
            read_scalar(stream_, version);
            if (endian_marker != RHO_ENDIAN_MARKER)
            {
                throw std::runtime_error("Density-matrix file endianness is not supported on this host.");
            }
            if (version != RHO_FILE_VERSION)
            {
                throw std::runtime_error("Unsupported density-matrix file version.");
            }

            read_scalar(stream_, metadata_.spatial_dimension);
            read_scalar(stream_, metadata_.total_qubits);
            read_scalar(stream_, metadata_.kept_qubits);
            read_scalar(stream_, metadata_.matrix_dimension);
            read_scalar(stream_, metadata_.periods);
            read_scalar(stream_, metadata_.realizations);
            read_scalar(stream_, metadata_.resolution);
            read_scalar(stream_, metadata_.grid_x);
            read_scalar(stream_, metadata_.grid_y);
            read_scalar(stream_, metadata_.subsystem_count);
            read_scalar(stream_, metadata_.record_count);
            read_scalar(stream_, metadata_.p_min);
            read_scalar(stream_, metadata_.p_max);

            const std::size_t index_count =
                static_cast<std::size_t>(metadata_.subsystem_count) *
                static_cast<std::size_t>(metadata_.kept_qubits);
            metadata_.subsystem_qubits.resize(index_count);
            stream_.read(reinterpret_cast<char *>(metadata_.subsystem_qubits.data()),
                         static_cast<std::streamsize>(index_count * sizeof(std::uint32_t)));
            if (!stream_)
            {
                throw std::runtime_error("Density-matrix subsystem metadata is incomplete.");
            }

            validate_metadata(metadata_);
            expected_values_ = payload_value_count(metadata_);
        }

        const DensityFileMetadata &metadata() const
        {
            return metadata_;
        }

        bool read_record(DensityRecord &record)
        {
            if (records_read_ >= metadata_.record_count)
            {
                return false;
            }

            read_scalar(stream_, record.p_index);
            read_scalar(stream_, record.realization);
            read_scalar(stream_, record.p);
            record.rho_ri.resize(expected_values_);
            stream_.read(reinterpret_cast<char *>(record.rho_ri.data()),
                         static_cast<std::streamsize>(expected_values_ * sizeof(double)));
            if (!stream_)
            {
                throw std::runtime_error("Density-matrix record payload is incomplete.");
            }

            ++records_read_;
            return true;
        }

      private:
        DensityFileMetadata metadata_;
        std::size_t expected_values_ = 0;
        std::uint64_t records_read_ = 0;
        std::ifstream stream_;
    };
} // namespace mipt_io

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class CommMapper:
    """Communication mapping helpers with batch sharding among two worker groups that require fixed rank pairing in communications.

    For example, env and rollout should always use the same rank pair for communications.
    """

    @staticmethod
    def build_channel_key(src_rank: int, dst_rank: int, extra: str) -> str:
        """Build a canonical point-to-point channel key."""
        return f"{src_rank}_{dst_rank}_{extra}"

    @staticmethod
    def parse_weights(weights, world_size: int) -> list[float] | None:
        """Parse optional rank weights used for uneven batch routing."""
        if weights is None:
            return None
        if isinstance(weights, str):
            value = weights.strip().strip("'").strip('"')
            if not value:
                return None
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            weights = [item.strip() for item in value.split(",") if item.strip()]
        else:
            weights = list(weights)

        assert len(weights) == world_size, (
            f"Expected {world_size} routing weights, got {len(weights)}: {weights}."
        )
        parsed = [float(weight) for weight in weights]
        assert all(weight > 0 for weight in parsed), (
            f"Routing weights must be positive, got {parsed}."
        )
        return parsed

    @staticmethod
    def partition_batch_size(
        batch_size: int,
        world_size: int,
        weights=None,
        align_to: int = 1,
    ) -> list[int]:
        """Partition a batch among ranks, optionally using weighted proportions."""
        assert batch_size > 0, "batch_size must be positive."
        assert world_size > 0, "world_size must be positive."
        assert align_to > 0, "align_to must be positive."
        assert batch_size % align_to == 0, (
            f"batch_size ({batch_size}) must be divisible by align_to ({align_to})."
        )

        parsed_weights = CommMapper.parse_weights(weights, world_size)
        if parsed_weights is None:
            assert batch_size % world_size == 0, (
                f"batch_size ({batch_size}) must be divisible by world_size ({world_size})."
            )
            return [batch_size // world_size for _ in range(world_size)]

        unit_count = batch_size // align_to
        assert unit_count >= world_size, (
            f"Not enough routing units ({unit_count}) for {world_size} ranks."
        )
        total_weight = sum(parsed_weights)
        raw_units = [weight / total_weight * unit_count for weight in parsed_weights]
        units = [max(1, int(raw)) for raw in raw_units]

        while sum(units) > unit_count:
            candidates = [
                idx for idx, value in enumerate(units) if value > 1
            ]
            assert candidates, (
                f"Unable to shrink weighted routing units {units} to {unit_count}."
            )
            idx = min(candidates, key=lambda i: (raw_units[i] - int(raw_units[i]), raw_units[i]))
            units[idx] -= 1

        while sum(units) < unit_count:
            idx = max(
                range(world_size),
                key=lambda i: (raw_units[i] - int(raw_units[i]), raw_units[i]),
            )
            units[idx] += 1
            raw_units[idx] = int(raw_units[idx])

        return [unit * align_to for unit in units]

    @staticmethod
    def _get_dst_ranks_from_sizes(
        src_sizes: list[int],
        dst_sizes: list[int],
        src_rank: int,
    ) -> list[tuple[int, int]]:
        assert 0 <= src_rank < len(src_sizes), (
            f"src_rank ({src_rank}) must be in [0, {len(src_sizes)})."
        )
        assert sum(src_sizes) == sum(dst_sizes), (
            f"Source sizes {src_sizes} and destination sizes {dst_sizes} "
            "must sum to the same batch size."
        )

        src_begin = sum(src_sizes[:src_rank])
        src_end = src_begin + src_sizes[src_rank]
        dst_ranks_and_sizes: list[tuple[int, int]] = []
        dst_begin = 0
        for dst_rank, dst_size in enumerate(dst_sizes):
            dst_end = dst_begin + dst_size
            overlap_begin = max(src_begin, dst_begin)
            overlap_end = min(src_end, dst_end)
            if overlap_begin < overlap_end:
                dst_ranks_and_sizes.append((dst_rank, overlap_end - overlap_begin))
            dst_begin = dst_end
        return dst_ranks_and_sizes

    @staticmethod
    def _get_src_ranks_from_sizes(
        src_sizes: list[int],
        dst_sizes: list[int],
        dst_rank: int,
    ) -> list[tuple[int, int]]:
        assert 0 <= dst_rank < len(dst_sizes), (
            f"dst_rank ({dst_rank}) must be in [0, {len(dst_sizes)})."
        )
        src_ranks_and_sizes: list[tuple[int, int]] = []
        for src_rank in range(len(src_sizes)):
            for mapped_dst_rank, size in CommMapper._get_dst_ranks_from_sizes(
                src_sizes, dst_sizes, src_rank
            ):
                if mapped_dst_rank == dst_rank:
                    src_ranks_and_sizes.append((src_rank, size))

        expected_size = dst_sizes[dst_rank]
        actual_size = sum(size for _, size in src_ranks_and_sizes)
        assert actual_size == expected_size, (
            f"Expected receive size {expected_size} for destination rank {dst_rank}, "
            f"got {actual_size} from mappings {src_ranks_and_sizes}."
        )
        return src_ranks_and_sizes

    @staticmethod
    def get_dst_ranks_by_partition(
        batch_size: int,
        src_world_size: int,
        dst_world_size: int,
        src_rank: int,
        src_weights=None,
        dst_weights=None,
        src_align_to: int = 1,
        dst_align_to: int = 1,
    ) -> list[tuple[int, int]]:
        src_sizes = CommMapper.partition_batch_size(
            batch_size, src_world_size, src_weights, src_align_to
        )
        dst_sizes = CommMapper.partition_batch_size(
            batch_size, dst_world_size, dst_weights, dst_align_to
        )
        return CommMapper._get_dst_ranks_from_sizes(src_sizes, dst_sizes, src_rank)

    @staticmethod
    def get_src_ranks_by_partition(
        batch_size: int,
        src_world_size: int,
        dst_world_size: int,
        dst_rank: int,
        src_weights=None,
        dst_weights=None,
        src_align_to: int = 1,
        dst_align_to: int = 1,
    ) -> list[tuple[int, int]]:
        src_sizes = CommMapper.partition_batch_size(
            batch_size, src_world_size, src_weights, src_align_to
        )
        dst_sizes = CommMapper.partition_batch_size(
            batch_size, dst_world_size, dst_weights, dst_align_to
        )
        return CommMapper._get_src_ranks_from_sizes(src_sizes, dst_sizes, dst_rank)

    @staticmethod
    def get_dst_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, src_rank: int
    ) -> list[tuple[int, int]]:
        """Compute destination ranks and transfer sizes for one source rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= src_rank < src_world_size, (
            f"src_rank ({src_rank}) must be in [0, {src_world_size})."
        )

        batch_size_per_src_rank = batch_size // src_world_size
        batch_size_per_dst_rank = batch_size // dst_world_size

        dst_ranks_and_sizes: list[tuple[int, int]] = []
        batch_begin = src_rank * batch_size_per_src_rank
        batch_end = (src_rank + 1) * batch_size_per_src_rank
        while batch_begin < batch_end:
            dst_rank = batch_begin // batch_size_per_dst_rank
            dst_batch_begin = dst_rank * batch_size_per_dst_rank
            dst_remaining = batch_size_per_dst_rank - (batch_begin - dst_batch_begin)
            src_remaining = batch_end - batch_begin
            dst_size = min(dst_remaining, src_remaining)
            dst_ranks_and_sizes.append((dst_rank, dst_size))
            batch_begin += dst_size
        return dst_ranks_and_sizes

    @staticmethod
    def get_src_ranks(
        batch_size: int, src_world_size: int, dst_world_size: int, dst_rank: int
    ) -> list[tuple[int, int]]:
        """Compute source ranks/sizes for one destination rank."""
        assert batch_size % src_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by src_world_size ({src_world_size})."
        )
        assert batch_size % dst_world_size == 0, (
            f"batch_size ({batch_size}) must be divisible by dst_world_size ({dst_world_size})."
        )
        assert 0 <= dst_rank < dst_world_size, (
            f"dst_rank ({dst_rank}) must be in [0, {dst_world_size})."
        )

        src_ranks_and_sizes: list[tuple[int, int]] = []
        for src_rank in range(src_world_size):
            dst_ranks_and_sizes = CommMapper.get_dst_ranks(
                batch_size=batch_size,
                src_world_size=src_world_size,
                dst_world_size=dst_world_size,
                src_rank=src_rank,
            )
            for mapped_dst_rank, size in dst_ranks_and_sizes:
                if mapped_dst_rank == dst_rank:
                    src_ranks_and_sizes.append((src_rank, size))

        expected_size = batch_size // dst_world_size
        actual_size = sum(size for _, size in src_ranks_and_sizes)
        assert actual_size == expected_size, (
            f"Expected receive size {expected_size} for destination rank {dst_rank}, "
            f"got {actual_size} from mappings {src_ranks_and_sizes}."
        )
        return src_ranks_and_sizes

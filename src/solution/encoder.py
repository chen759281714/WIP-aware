"""Encoding utilities for operation-priority OS and fixed-order MS.

OS is an explicit operation-priority sequence. Each gene is a unique
``(job_id, op_idx)`` tuple, operations of the same job remain in technological
order, and a gene's position is its static priority rather than its actual
start position in the decoded schedule.

MS remains a one-dimensional list expanded in fixed ``(job, op)`` order.
"""

import os
import random
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class Encoder:
    """Generate and validate OS/MS chromosomes for one scheduling instance."""

    def __init__(self, operations, rng=None):
        self.operations = operations
        self.rng = rng if rng is not None else random
        self.ms_index_order = self.build_ms_index_order()

    def build_ms_index_order(self):
        """Return the fixed MS expansion order."""
        order = []
        for job_id, ops in self.operations.items():
            for op_idx in range(len(ops)):
                order.append((job_id, op_idx))
        return order

    def build_ms_map(self, ms_list):
        """Convert an MS list into ``(job, op_idx) -> machine_id``."""
        if len(ms_list) != len(self.ms_index_order):
            raise ValueError("MS_list length does not match the operation count")

        ms_map = {}
        for idx, (job, op_idx) in enumerate(self.ms_index_order):
            machine = ms_list[idx]
            legal_machines = self.operations[job][op_idx]["machines"].keys()
            if machine not in legal_machines:
                raise ValueError(
                    f"Illegal machine selection: {(job, op_idx)} cannot use {machine}"
                )
            ms_map[(job, op_idx)] = machine
        return ms_map

    def generate_random_os(self):
        """Generate a random precedence-preserving operation-priority OS.

        A repeated-job list is shuffled first. The kth occurrence of each job
        is then mapped to ``(job, k)``. This retains the old random interleaving
        distribution while making every operation explicit and unique.
        """
        repeated_jobs = []
        for job_id, ops in self.operations.items():
            repeated_jobs.extend([job_id] * len(ops))
        self.rng.shuffle(repeated_jobs)

        occurrences = {job_id: 0 for job_id in self.operations}
        os_seq = []
        for job_id in repeated_jobs:
            op_idx = occurrences[job_id]
            os_seq.append((job_id, op_idx))
            occurrences[job_id] += 1
        return os_seq

    def validate_os(self, os_seq):
        """Validate explicit genes, coverage, uniqueness, and precedence."""
        if not isinstance(os_seq, list):
            raise ValueError("OS must be a list")

        expected_length = len(self.ms_index_order)
        if len(os_seq) != expected_length:
            raise ValueError(
                f"Invalid OS length: expected {expected_length}, got {len(os_seq)}"
            )

        seen = set()
        for position, gene in enumerate(os_seq):
            if not isinstance(gene, tuple) or len(gene) != 2:
                raise ValueError(
                    f"OS gene {position} must be a (job_id, op_idx) tuple: {gene!r}"
                )

            job, op_idx = gene
            if job not in self.operations:
                raise ValueError(f"OS gene {position} has an unknown job: {job!r}")
            if not isinstance(op_idx, int) or isinstance(op_idx, bool):
                raise ValueError(f"OS gene {position} has a non-integer op_idx: {gene!r}")
            if op_idx < 0 or op_idx >= len(self.operations[job]):
                raise ValueError(f"OS gene {position} is not a valid operation: {gene!r}")
            if gene in seen:
                raise ValueError(f"OS contains a duplicate operation: {gene!r}")
            seen.add(gene)

        missing = [key for key in self.ms_index_order if key not in seen]
        if missing:
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(f"OS is missing operations: {missing[:10]}{suffix}")

        next_expected = {job: 0 for job in self.operations}
        for position, (job, op_idx) in enumerate(os_seq):
            expected_op_idx = next_expected[job]
            if op_idx != expected_op_idx:
                raise ValueError(
                    f"OS violates precedence for {job} at position {position}: "
                    f"got {(job, op_idx)!r}, expected {(job, expected_op_idx)!r}"
                )
            next_expected[job] += 1

        return True

    def build_priority_rank(self, os_seq):
        """Validate OS and return the static rank of every operation."""
        self.validate_os(os_seq)
        return {op_key: position for position, op_key in enumerate(os_seq)}

    def generate_random_ms(self):
        """Generate a random legal machine selection list."""
        ms_list = []
        for job, op_idx in self.ms_index_order:
            machines = list(self.operations[job][op_idx]["machines"].keys())
            ms_list.append(self.rng.choice(machines))
        return ms_list

    def get_total_operations(self):
        """Return the total operation count."""
        return len(self.ms_index_order)

    def print_ms_index_order(self):
        """Print the MS expansion order for debugging."""
        print("MS index order:")
        for idx, item in enumerate(self.ms_index_order):
            print(idx, "->", item)

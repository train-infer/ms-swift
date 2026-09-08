#!/usr/bin/env python3
"""将原始 JSONL 切成细分片，由多节点动态领取任务并合并完整 cached dataset。"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


DEFAULT_CONFIG_NAME = "cached_dataset_config.example.toml"


def log_progress(message: str) -> None:
    """输出毫秒级进度日志。"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} {message}", flush=True)


def parse_bool(value) -> bool:
    """解析布尔配置值。"""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid bool value: {value}")


def bool_arg(value: bool) -> str:
    """转换命令行布尔值。"""
    return "true" if value else "false"


def shard_name(shard_id: int) -> str:
    """生成分片名称。"""
    return f"shard_{shard_id:04d}"


def write_json(path: Path, data: Dict) -> None:
    """写入 JSON 文件。"""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    """流式计算文件摘要，避免一次性读取大文件。"""
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raise_open_file_limit(required: int) -> None:
    """尽量抬升进程可同时打开的文件数上限，不足则报错退出。"""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= required:
        return
    new_soft = required if hard == resource.RLIM_INFINITY else min(hard, required)
    if new_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        soft = new_soft
    if soft < required:
        raise SystemExit(f"open file limit too low: need>={required}, current={soft}")


@dataclass
class BuildConfig:
    """保存主控和 worker 共享配置。"""

    env_bin: Path
    swift_bin: Path
    python_bin: Path
    ms_swift_repo: Path
    model: Path
    ip_file: Path
    total_shards: int
    dataset_num_proc: int
    data_seed: int
    truncation_strategy: str
    template: str
    agent_template: str
    loss_scale: str
    role_loss_config: Optional[Path]
    materialize_role_weights: bool
    max_length: int
    dataset_shuffle: bool
    split_dataset_ratio: float
    load_from_cache_file: bool
    to_cached_dataset: bool
    copy_parallel: int
    save_num_proc: int
    auto_merge: bool
    stale_task_seconds: int
    heartbeat_seconds: int
    progress_interval_seconds: int
    max_worker_rounds: int
    split_buffer_bytes: int
    split_num_proc: int
    split_chunk_bytes: int
    split_worker_buffer_bytes: int
    merge_group_size: int
    local_work_root: Path
    allowed_output_root: Path
    work_root: Path
    raw_shards_dir: Path
    cached_shards_dir: Path
    final_out_dir: Path
    log_dir: Path
    datasets: List[Path]
    cuda_fallback_ld_path: str
    script_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    config_path: Path = field(default_factory=lambda: Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME))
    run_id: str = field(default_factory=lambda: f"r{time.strftime('%m%d%H%M%S')}")

    @classmethod
    def from_file(cls, config_path: Path) -> "BuildConfig":
        """从 TOML 配置创建构建配置。"""
        config_path = config_path.expanduser()
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        else:
            config_path = config_path.resolve()
        if not config_path.is_file():
            raise SystemExit(f"missing config file: {config_path}")
        data = tomllib.loads(config_path.read_text())

        runtime = data.get("runtime", {})
        output = data.get("output", {})
        build = data.get("build", {})
        export = data.get("export", {})
        dataset = data.get("dataset", {})
        run_id = os.environ.get("RUN_ID", f"r{time.strftime('%m%d%H%M%S')}")
        env_bin = Path(os.environ.get("ENV_BIN", runtime["env_bin"]))
        default_repo = Path(__file__).resolve().parents[3]
        ms_swift_repo = Path(os.environ.get("MS_SWIFT_REPO", runtime.get("ms_swift_repo", default_repo)))
        role_loss_config_value = os.environ.get("ROLE_LOSS_CONFIG", export.get("role_loss_config"))
        role_loss_config = Path(role_loss_config_value) if role_loss_config_value else None
        if role_loss_config is not None and not role_loss_config.is_absolute():
            role_loss_config = (config_path.parent / role_loss_config).resolve()
        work_root = Path(os.environ.get("WORK_ROOT", output["work_root"]))
        raw_shards_dir = Path(os.environ.get("RAW_SHARDS_DIR", output["raw_shards_dir"]))
        cached_shards_dir = Path(os.environ.get("CACHED_SHARDS_DIR", output["cached_shards_dir"]))
        log_dir = Path(os.environ.get("LOG_DIR", output.get("log_dir", str(work_root / "logs" / run_id))))
        base = Path(dataset.get("base", ""))
        files = dataset["files"]
        datasets = [Path(item) if Path(item).is_absolute() else base / item for item in files]
        if "DATASETS" in os.environ:
            datasets = [Path(p) for p in os.environ["DATASETS"].split(os.pathsep) if p]

        return cls(
            env_bin=env_bin,
            swift_bin=Path(os.environ.get("SWIFT_BIN", runtime.get("swift_bin", str(env_bin / "swift")))),
            python_bin=Path(os.environ.get("PYTHON_BIN", runtime.get("python_bin", str(env_bin / "python")))),
            ms_swift_repo=ms_swift_repo,
            model=Path(os.environ.get("MODEL", data["model"])),
            ip_file=Path(os.environ.get("IP_FILE", data["ip_file"])),
            total_shards=int(os.environ.get("TOTAL_SHARDS", build["total_shards"])),
            dataset_num_proc=int(os.environ.get("DATASET_NUM_PROC", export["dataset_num_proc"])),
            data_seed=int(os.environ.get("DATA_SEED", export["data_seed"])),
            truncation_strategy=os.environ.get("TRUNCATION_STRATEGY", export["truncation_strategy"]),
            template=os.environ.get("TEMPLATE", export["template"]),
            agent_template=os.environ.get("AGENT_TEMPLATE", export["agent_template"]),
            loss_scale=os.environ.get("LOSS_SCALE", export["loss_scale"]),
            role_loss_config=role_loss_config,
            materialize_role_weights=parse_bool(
                os.environ.get("MATERIALIZE_ROLE_WEIGHTS", export.get("materialize_role_weights", False))),
            max_length=int(os.environ.get("MAX_LENGTH", export["max_length"])),
            dataset_shuffle=parse_bool(os.environ.get("DATASET_SHUFFLE", export["dataset_shuffle"])),
            split_dataset_ratio=float(os.environ.get("SPLIT_DATASET_RATIO", export["split_dataset_ratio"])),
            load_from_cache_file=parse_bool(os.environ.get("LOAD_FROM_CACHE_FILE", export["load_from_cache_file"])),
            to_cached_dataset=parse_bool(os.environ.get("TO_CACHED_DATASET", export["to_cached_dataset"])),
            copy_parallel=int(os.environ.get("COPY_PARALLEL", build["copy_parallel"])),
            save_num_proc=int(os.environ.get("SAVE_NUM_PROC", build["save_num_proc"])),
            auto_merge=parse_bool(os.environ.get("AUTO_MERGE", build["auto_merge"])),
            stale_task_seconds=int(os.environ.get("STALE_TASK_SECONDS", build["stale_task_seconds"])),
            heartbeat_seconds=int(os.environ.get("HEARTBEAT_SECONDS", build["heartbeat_seconds"])),
            progress_interval_seconds=int(os.environ.get("PROGRESS_INTERVAL_SECONDS", build["progress_interval_seconds"])),
            max_worker_rounds=int(os.environ.get("MAX_WORKER_ROUNDS", build["max_worker_rounds"])),
            split_buffer_bytes=int(os.environ.get("SPLIT_BUFFER_BYTES", build["split_buffer_bytes"])),
            split_num_proc=int(os.environ.get("SPLIT_NUM_PROC", build.get("split_num_proc", 1))),
            split_chunk_bytes=int(os.environ.get("SPLIT_CHUNK_BYTES", build.get("split_chunk_bytes", 8 << 30))),
            split_worker_buffer_bytes=int(os.environ.get(
                "SPLIT_WORKER_BUFFER_BYTES", build.get("split_worker_buffer_bytes", 64 << 20)
            )),
            merge_group_size=int(os.environ.get("MERGE_GROUP_SIZE", build["merge_group_size"])),
            local_work_root=Path(os.environ.get("LOCAL_WORK_ROOT", output["local_work_root"])),
            allowed_output_root=Path(os.environ.get("ALLOWED_OUTPUT_ROOT", output["allowed_output_root"])),
            work_root=work_root,
            raw_shards_dir=raw_shards_dir,
            cached_shards_dir=cached_shards_dir,
            final_out_dir=Path(os.environ.get("FINAL_OUT_DIR", output["final_out_dir"])),
            log_dir=log_dir,
            datasets=datasets,
            cuda_fallback_ld_path=os.environ.get("CUDA_FALLBACK_LD_PATH", runtime["cuda_fallback_ld_path"]),
            config_path=config_path,
            run_id=run_id,
        )

    @cached_property
    def swift_commit(self) -> str:
        """返回本次构建绑定的 ms-swift Git 提交。"""
        try:
            return subprocess.check_output(
                ["git", "-C", str(self.ms_swift_repo), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"failed to resolve ms-swift commit: {self.ms_swift_repo}") from exc

    @cached_property
    def build_metadata(self) -> Dict:
        """生成决定 cached dataset 语义的稳定元数据。"""
        role_config_sha256 = sha256_file(self.role_loss_config) if self.role_loss_config else None
        source_files = [
            "swift/loss_scale/role.py",
            "swift/loss_scale/mapping.py",
            "swift/template/base.py",
            "swift/template/template_inputs.py",
            "swift/template/utils.py",
            "swift/arguments/base_args/template_args.py",
            "scripts/utils/inject_role_loss_scale.py",
            "scripts/utils/build_cached_dataset/distributed_cached_dataset.py",
        ]
        source_hashes = {
            name: sha256_file(self.ms_swift_repo / name)
            for name in source_files
        }
        model_files = {}
        for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
            path = self.model / name
            if path.is_file():
                model_files[name] = sha256_file(path)
        raw_manifest = self.raw_shards_dir / "manifest.json"
        return {
            "ms_swift_repo": str(self.ms_swift_repo.resolve()),
            "ms_swift_commit": self.swift_commit,
            "ms_swift_source_hashes": source_hashes,
            "model": str(self.model.resolve()),
            "model_files": model_files,
            "template": self.template,
            "agent_template": self.agent_template,
            "loss_scale": self.loss_scale,
            "role_loss_config": str(self.role_loss_config.resolve()) if self.role_loss_config else None,
            "role_loss_config_sha256": role_config_sha256,
            "materialize_role_weights": self.materialize_role_weights,
            "max_length": self.max_length,
            "truncation_strategy": self.truncation_strategy,
            "raw_manifest_sha256": sha256_file(raw_manifest) if raw_manifest.is_file() else None,
        }

    @cached_property
    def build_fingerprint(self) -> str:
        """返回 cached shard 复用检查使用的构建指纹。"""
        payload = json.dumps(self.build_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def cached_shard_is_ready(self, path: Path) -> bool:
        """仅复用由完全相同源码和配置生成的 cached shard。"""
        if not (path / "_SUCCESS").exists():
            return False
        manifest_path = path / "build_manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"cached shard缺少构建指纹，拒绝复用：{path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("build_fingerprint") != self.build_fingerprint:
            raise SystemExit(f"cached shard构建指纹不一致，拒绝复用：{path}")
        return True

    def validate(self) -> None:
        """校验输入、输出、源码和 role 权重配置。"""
        for binary in (self.swift_bin, self.python_bin):
            if not os.access(binary, os.X_OK):
                raise SystemExit(f"missing executable: {binary}")
        if not (self.ms_swift_repo / "swift" / "__init__.py").is_file():
            raise SystemExit(f"invalid ms-swift repository: {self.ms_swift_repo}")
        if not (self.ms_swift_repo / "scripts" / "utils" / "inject_role_loss_scale.py").is_file():
            raise SystemExit(f"missing role loss injector in repository: {self.ms_swift_repo}")
        import_env = os.environ.copy()
        current_pythonpath = import_env.get("PYTHONPATH", "")
        import_env["PYTHONPATH"] = str(self.ms_swift_repo) + (
            os.pathsep + current_pythonpath if current_pythonpath else "")
        try:
            imported_swift = subprocess.check_output(
                [self.python_bin, "-c", "import swift; print(swift.__file__)"], env=import_env, text=True).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"failed to import swift from repository: {self.ms_swift_repo}") from exc
        if not Path(imported_swift).resolve().is_relative_to(self.ms_swift_repo.resolve()):
            raise SystemExit(
                f"swift import does not resolve to configured repository: {imported_swift} != {self.ms_swift_repo}")
        if not self.model.is_dir():
            raise SystemExit(f"missing model: {self.model}")
        role_enabled = self.loss_scale.split("+", 1)[0] == "role"
        if role_enabled and (self.role_loss_config is None or not self.role_loss_config.is_file()):
            raise SystemExit("role loss_scale requires an existing role_loss_config")
        if self.role_loss_config is not None and not role_enabled:
            raise SystemExit("role_loss_config is only valid when loss_scale starts with role")
        if self.materialize_role_weights and not role_enabled:
            raise SystemExit("materialize_role_weights requires role loss_scale")
        if not self.to_cached_dataset:
            raise SystemExit("distributed cached dataset builder requires to_cached_dataset=true")
        allow_root = self.allowed_output_root.expanduser().resolve()
        if allow_root == Path("/") or len(allow_root.parts) < 3:
            raise SystemExit(f"unsafe allowed_output_root: {allow_root}")
        for path_item in (self.work_root, self.raw_shards_dir, self.cached_shards_dir, self.final_out_dir):
            if not path_item.expanduser().resolve().is_relative_to(allow_root):
                raise SystemExit(f"output path must be under {allow_root}: {path_item}")
        if min(self.split_num_proc, self.split_chunk_bytes, self.split_worker_buffer_bytes) <= 0:
            raise SystemExit("split_num_proc, split_chunk_bytes and split_worker_buffer_bytes must be positive")
        if self.truncation_strategy not in {"delete", "left", "right", "split"}:
            raise SystemExit(f"invalid truncation strategy: {self.truncation_strategy}")
        if self.progress_interval_seconds <= 0:
            raise SystemExit(f"invalid progress interval: {self.progress_interval_seconds}")
        if str(self.local_work_root) in {"", "/"}:
            raise SystemExit(f"unsafe LOCAL_WORK_ROOT: {self.local_work_root}")
        if not self.datasets:
            raise SystemExit("empty dataset list")
        for dataset in self.datasets:
            if not dataset.is_file():
                raise SystemExit(f"missing dataset: {dataset}")

    def runtime_env_for_worker(self) -> Dict[str, str]:
        """生成远端 worker 运行时环境变量。"""
        pythonpath = os.environ.get("PYTHONPATH", "")
        keys = {
            "RUN_ID": self.run_id,
            "LOG_DIR": str(self.log_dir),
            "MS_SWIFT_REPO": str(self.ms_swift_repo),
            "PYTHONPATH": str(self.ms_swift_repo) + (os.pathsep + pythonpath if pythonpath else ""),
        }
        if self.role_loss_config is not None:
            keys["ROLE_LOSS_CONFIG"] = str(self.role_loss_config)
        keys["MATERIALIZE_ROLE_WEIGHTS"] = bool_arg(self.materialize_role_weights)
        # Keep this list in sync with environment overrides in BuildConfig.from_file().
        # These names are forwarded to remote workers so master/worker see the same config.
        override_names = (
            "ENV_BIN", "SWIFT_BIN", "PYTHON_BIN", "CUDA_FALLBACK_LD_PATH", "MS_SWIFT_REPO",
            "MODEL", "IP_FILE", "TOTAL_SHARDS", "DATASET_NUM_PROC",
            "DATA_SEED", "TRUNCATION_STRATEGY", "TEMPLATE",
            "AGENT_TEMPLATE", "LOSS_SCALE", "ROLE_LOSS_CONFIG", "MATERIALIZE_ROLE_WEIGHTS",
            "MAX_LENGTH", "DATASET_SHUFFLE",
            "SPLIT_DATASET_RATIO", "LOAD_FROM_CACHE_FILE", "TO_CACHED_DATASET",
            "COPY_PARALLEL", "SAVE_NUM_PROC", "AUTO_MERGE", "STALE_TASK_SECONDS",
            "HEARTBEAT_SECONDS", "PROGRESS_INTERVAL_SECONDS", "MAX_WORKER_ROUNDS", "SPLIT_BUFFER_BYTES",
            "SPLIT_NUM_PROC", "SPLIT_CHUNK_BYTES", "SPLIT_WORKER_BUFFER_BYTES",
            "MERGE_GROUP_SIZE", "WORK_ROOT", "RAW_SHARDS_DIR",
            "CACHED_SHARDS_DIR", "FINAL_OUT_DIR", "LOCAL_WORK_ROOT", "ALLOWED_OUTPUT_ROOT", "DATASETS",
        )
        for name in override_names:
            if name in os.environ:
                keys[name] = os.environ[name]
        return keys


class TimingRecorder:
    """统一记录控制端和单 worker 的阶段耗时，便于定位长尾。"""

    def __init__(self, config: BuildConfig, role: str):
        self.config = config
        self.role = role
        self.host = socket.gethostname()
        self.worker_index = os.environ.get("WORKER_INDEX")
        self.worker_ip = os.environ.get("WORKER_IP")
        timing_dir = config.log_dir / "timing"
        if role == "worker":
            safe_ip = (self.worker_ip or self.host).replace(".", "_")
            index = self.worker_index or "local"
            filename = f"worker_{index}_{safe_ip}.jsonl"
        else:
            filename = "controller.jsonl"
        self.path = timing_dir / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields) -> None:
        """写入单行 JSON 事件，并向关联日志输出简明时间记录。"""
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.config.run_id,
            "role": self.role,
            "host": self.host,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        if self.worker_index is not None:
            record["worker_index"] = self.worker_index
        if self.worker_ip is not None:
            record["worker_ip"] = self.worker_ip
        with self.path.open("a", encoding="utf-8") as writer:
            writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        details = " ".join(
            f"{key}={record[key]}"
            for key in ("phase", "shard", "round", "group_index", "elapsed_sec")
            if key in record
        )
        log_progress(f"timing event={event}" + (f" {details}" if details else ""))

    @contextmanager
    def phase(self, phase: str, **fields) -> Iterator[None]:
        """记录一个阶段的开始、结束或失败事件及单调时钟耗时。"""
        started = time.monotonic()
        self.event("phase_start", phase=phase, **fields)
        try:
            yield
        except BaseException as exc:
            self.event(
                "phase_failed",
                phase=phase,
                elapsed_sec=round(time.monotonic() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
                **fields,
            )
            raise
        else:
            self.event(
                "phase_end",
                phase=phase,
                elapsed_sec=round(time.monotonic() - started, 3),
                **fields,
            )

    def write_summary(self) -> None:
        """汇总全部 worker 的完成事件，输出主要阶段的长尾记录。"""
        completed = []
        for path in sorted(self.path.parent.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "phase_end" and "elapsed_sec" in record:
                    completed.append(record)
        by_phase: Dict[str, List[Dict]] = {}
        for record in completed:
            by_phase.setdefault(record["phase"], []).append(record)
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.config.run_id,
            "phase_summary": {
                phase: {
                    "count": len(records),
                    "total_sec": round(sum(float(record["elapsed_sec"]) for record in records), 3),
                    "avg_sec": round(
                        sum(float(record["elapsed_sec"]) for record in records) / len(records),
                        3,
                    ),
                    "max_sec": round(max(float(record["elapsed_sec"]) for record in records), 3),
                }
                for phase, records in sorted(by_phase.items())
            },
            "slowest_completed_phases": sorted(
                completed,
                key=lambda item: float(item["elapsed_sec"]),
                reverse=True,
            )[:50],
            "slowest_by_phase": {
                phase: sorted(
                    records,
                    key=lambda item: float(item["elapsed_sec"]),
                    reverse=True,
                )[:10]
                for phase, records in sorted(by_phase.items())
            },
        }
        write_json(self.path.parent / "summary.json", summary)
        log_progress(
            f"timing summary: completed_phases={len(completed)} "
            f"output={self.path.parent / 'summary.json'}"
        )


@dataclass(frozen=True)
class SplitChunk:
    """一个以完整 JSONL 行为边界的输入区间。"""

    source: Path
    start: int
    end: int


@dataclass(frozen=True)
class SplitChunkStats:
    """第一遍扫描得到的块布局信息。"""

    row_count: int
    bytes_by_remainder: Tuple[int, ...]
    digest: bytes


@dataclass(frozen=True)
class SplitChunkLayout:
    """第二遍写入使用的全局行号与文件偏移。"""

    chunk: SplitChunk
    row_start: int
    offsets_by_remainder: Tuple[int, ...]


def iter_chunk_lines(chunk: SplitChunk) -> Iterator[bytes]:
    """读取一个换行对齐的块，并保持旧实现的补换行语义。"""
    with chunk.source.open("rb") as reader:
        reader.seek(chunk.start)
        remaining = chunk.end - chunk.start
        while remaining:
            line = reader.readline(remaining)
            if not line:
                raise RuntimeError(f"unexpected EOF in chunk: {chunk.source}:{chunk.end}")
            remaining -= len(line)
            if line and not line.endswith(b"\n"):
                if chunk.end < os.fstat(reader.fileno()).st_size:
                    raise RuntimeError(f"chunk boundary splits a line: {chunk.source}:{chunk.end}")
                line += b"\n"
            yield line


def scan_split_chunk(args: Tuple[SplitChunk, int]) -> SplitChunkStats:
    """统计块内各行余数对应的最终字节数。"""
    chunk, total_shards = args
    byte_counts = [0] * total_shards
    digest = hashlib.sha256()
    row_count = 0
    for line in iter_chunk_lines(chunk):
        byte_counts[row_count % total_shards] += len(line)
        digest.update(line)
        row_count += 1
    return SplitChunkStats(row_count, tuple(byte_counts), digest.digest())


def pwrite_all(fd: int, data: bytearray, offset: int) -> int:
    """将缓冲区完整写入指定文件偏移。"""
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.pwrite(fd, view[written:], offset + written)
        if count <= 0:
            raise OSError("pwrite returned no progress")
        written += count
    return written


def write_split_chunk(
    args: Tuple[SplitChunkLayout, Path, int, int]
) -> Tuple[int, Tuple[int, ...], bytes]:
    """把一个块直接写入所有最终分片的互斥区间。"""
    layout, output_dir, total_shards, buffer_budget = args
    buffer_limit = max(1, buffer_budget // total_shards)
    buffers = [bytearray() for _ in range(total_shards)]
    positions = list(layout.offsets_by_remainder)
    byte_counts = [0] * total_shards
    digest = hashlib.sha256()
    fds = []
    row_count = 0
    try:
        for shard_id in range(total_shards):
            fds.append(os.open(output_dir / f"{shard_name(shard_id)}.jsonl", os.O_WRONLY))
        for line in iter_chunk_lines(layout.chunk):
            remainder = row_count % total_shards
            byte_counts[remainder] += len(line)
            digest.update(line)
            buffer = buffers[remainder]
            buffer.extend(line)
            if len(buffer) >= buffer_limit:
                shard_id = (layout.row_start + remainder) % total_shards
                count = pwrite_all(fds[shard_id], buffer, positions[remainder])
                positions[remainder] += count
                buffer.clear()
            row_count += 1
        for remainder, buffer in enumerate(buffers):
            if not buffer:
                continue
            shard_id = (layout.row_start + remainder) % total_shards
            count = pwrite_all(fds[shard_id], buffer, positions[remainder])
            positions[remainder] += count
    finally:
        for fd in fds:
            os.close(fd)
    return row_count, tuple(byte_counts), digest.digest()


class RawShardSplitter:
    """将原始 JSONL 切成分片。"""

    def __init__(self, config: BuildConfig, timing: TimingRecorder):
        """保存分片配置和阶段计时器。"""
        self.config = config
        self.timing = timing

    def ensure_split(self) -> None:
        """确保原始分片已就绪。"""
        if self._is_ready():
            self.timing.event("phase_skipped", phase="raw_split", reason="already_ready")
            print(f"reuse raw shards: {self.config.raw_shards_dir}", flush=True)
            return
        if self.config.raw_shards_dir.exists():
            raise SystemExit(f"raw shards dir exists but is not valid: {self.config.raw_shards_dir}")
        with self.timing.phase("raw_split", total_shards=self.config.total_shards):
            self._split()

    def _is_ready(self) -> bool:
        """检查原始分片是否完整。"""
        success = self.config.raw_shards_dir / "_SUCCESS"
        manifest = self.config.raw_shards_dir / "manifest.json"
        if not success.exists() or not manifest.exists():
            return False
        data = json.loads(manifest.read_text())
        if data.get("num_shards") != self.config.total_shards:
            raise SystemExit(f"raw shard count mismatch in manifest: {manifest}")
        expected_sources = [{
            "path": str(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        } for path in self.config.datasets]
        if data.get("source_files") != expected_sources:
            raise SystemExit(f"raw shard source fingerprint mismatch: {manifest}")
        return all((self.config.raw_shards_dir / f"{shard_name(i)}.jsonl").is_file() for i in range(self.config.total_shards))

    def _split(self) -> None:
        """按配置选择串行或并行分片。"""
        if self.config.split_num_proc == 1:
            self._split_serial()
        else:
            self._split_parallel()

    def _split_serial(self) -> None:
        """按行轮询切分原始数据。"""
        out_dir = self.config.raw_shards_dir
        tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp.{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        self._ensure_file_limit()
        buffers = [bytearray() for _ in range(self.config.total_shards)]
        counts = [0] * self.config.total_shards
        total = 0
        last_progress_at = time.monotonic()
        try:
            from contextlib import ExitStack

            with ExitStack() as stack:
                writers = [
                    stack.enter_context((tmp_dir / f"{shard_name(shard_id)}.jsonl").open("wb"))
                    for shard_id in range(self.config.total_shards)
                ]
                for source in self.config.datasets:
                    with source.open("rb") as reader:
                        for line in reader:
                            shard_id = total % self.config.total_shards
                            if line and not line.endswith(b"\n"):
                                line += b"\n"
                            buffers[shard_id].extend(line)
                            if len(buffers[shard_id]) >= self.config.split_buffer_bytes:
                                self._flush_buffer(writers[shard_id], buffers[shard_id])
                            counts[shard_id] += 1
                            total += 1
                            now = time.monotonic()
                            if now - last_progress_at >= self.config.progress_interval_seconds:
                                log_progress(f"split progress: rows={total}")
                                last_progress_at = now
                for shard_id, buffer in enumerate(buffers):
                    self._flush_buffer(writers[shard_id], buffer)
            self._publish(tmp_dir, out_dir, total, counts)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        print(f"created {self.config.total_shards} raw shards, total rows={total}, output={out_dir}", flush=True)

    def _split_parallel(self) -> None:
        """两遍并行扫描，并把各块写入预计算的互斥区间。"""
        out_dir = self.config.raw_shards_dir
        tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp.{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        try:
            self._ensure_file_limit()
            signatures = {
                source: (source.stat().st_size, source.stat().st_mtime_ns)
                for source in self.config.datasets
            }
            chunks = self._build_chunks()
            with self.timing.phase("raw_split_layout", chunks=len(chunks)):
                stats = self._parallel_map(
                    scan_split_chunk,
                    [(chunk, self.config.total_shards) for chunk in chunks],
                    "layout",
                )
            if any(
                (source.stat().st_size, source.stat().st_mtime_ns) != signature
                for source, signature in signatures.items()
            ):
                raise RuntimeError("source dataset changed during split")
            layouts, counts, shard_sizes = self._build_layouts(chunks, stats)
            for shard_id, size in enumerate(shard_sizes):
                with (tmp_dir / f"{shard_name(shard_id)}.jsonl").open("wb") as writer:
                    writer.truncate(size)
            with self.timing.phase("raw_split_write", chunks=len(chunks)):
                written = self._parallel_map(
                    write_split_chunk,
                    [
                        (layout, tmp_dir, self.config.total_shards, self.config.split_worker_buffer_bytes)
                        for layout in layouts
                    ],
                    "write",
                )
            for index, (rows, byte_counts, digest) in enumerate(written):
                expected = stats[index]
                if (
                    rows != expected.row_count
                    or byte_counts != expected.bytes_by_remainder
                    or digest != expected.digest
                ):
                    raise RuntimeError(f"parallel split write mismatch in chunk {index}")
            if any(
                (source.stat().st_size, source.stat().st_mtime_ns) != signature
                for source, signature in signatures.items()
            ):
                raise RuntimeError("source dataset changed during split")
            total = sum(item.row_count for item in stats)
            self._publish(tmp_dir, out_dir, total, counts)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        print(f"created {self.config.total_shards} raw shards, total rows={total}, output={out_dir}", flush=True)

    def _build_chunks(self) -> List[SplitChunk]:
        """把输入文件切成换行对齐的固定大小区间。"""
        chunks = []
        chunk_bytes = self.config.split_chunk_bytes
        for source in self.config.datasets:
            size = source.stat().st_size
            start = 0
            with source.open("rb") as reader:
                while start < size:
                    end = min(start + chunk_bytes, size)
                    if end < size:
                        reader.seek(end)
                        reader.readline()
                        end = reader.tell()
                    chunks.append(SplitChunk(source, start, end))
                    start = end
        return chunks

    def _build_layouts(
        self,
        chunks: List[SplitChunk],
        stats: List[SplitChunkStats],
    ) -> Tuple[List[SplitChunkLayout], List[int], List[int]]:
        """按旧有全局轮询规则计算每块的输出偏移。"""
        total_shards = self.config.total_shards
        shard_offsets = [0] * total_shards
        row_counts = [0] * total_shards
        layouts = []
        row_start = 0
        for chunk, chunk_stats in zip(chunks, stats):
            offsets = [0] * total_shards
            shift = row_start % total_shards
            for remainder, byte_count in enumerate(chunk_stats.bytes_by_remainder):
                shard_id = (shift + remainder) % total_shards
                offsets[remainder] = shard_offsets[shard_id]
                shard_offsets[shard_id] += byte_count
                row_counts[shard_id] += (
                    chunk_stats.row_count + total_shards - 1 - remainder
                ) // total_shards
            layouts.append(SplitChunkLayout(chunk, row_start, tuple(offsets)))
            row_start += chunk_stats.row_count
        return layouts, row_counts, shard_offsets

    def _parallel_map(self, worker, tasks: List, label: str) -> List:
        """并行执行任务，同时按输入顺序返回结果。"""
        results = [None] * len(tasks)
        last_progress_at = time.monotonic()
        with ProcessPoolExecutor(max_workers=self.config.split_num_proc) as executor:
            futures = {
                executor.submit(worker, task): index
                for index, task in enumerate(tasks)
            }
            try:
                for completed, future in enumerate(as_completed(futures), 1):
                    results[futures[future]] = future.result()
                    now = time.monotonic()
                    if now - last_progress_at >= self.config.progress_interval_seconds:
                        log_progress(f"split {label} progress: chunks={completed}/{len(tasks)}")
                        last_progress_at = now
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    def _publish(
        self,
        tmp_dir: Path,
        out_dir: Path,
        total: int,
        counts: List[int],
    ) -> None:
        """写入完成标记并原子发布分片目录。"""
        write_json(tmp_dir / "manifest.json", {
            "num_shards": self.config.total_shards,
            "total_rows": total,
            "rows_per_shard": counts,
            "sources": [str(p) for p in self.config.datasets],
            "source_files": [{
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            } for path in self.config.datasets],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        (tmp_dir / "_SUCCESS").write_text("ok\n")
        os.rename(tmp_dir, out_dir)

    def _ensure_file_limit(self) -> None:
        """确保可常开所有分片文件。"""
        process_pool_fds = 2 * self.config.split_num_proc
        raise_open_file_limit(self.config.total_shards + process_pool_fds + 128)

    @staticmethod
    def _flush_buffer(writer, buffer: bytearray) -> None:
        """将分片缓冲写入常开文件。"""
        if not buffer:
            return
        writer.write(buffer)
        buffer.clear()


class ShardTaskQueue:
    """管理分片任务队列。"""

    def __init__(self, config: BuildConfig):
        """初始化队列目录。"""
        self.config = config
        self.root = config.work_root / "task_queue" / config.run_id
        self.pending_dir = self.root / "pending"
        self.running_dir = self.root / "running"
        self.done_dir = self.root / "done"
        self.failed_dir = self.root / "failed"

    def prepare(self) -> None:
        """创建待处理任务。"""
        for path in (self.pending_dir, self.running_dir, self.done_dir, self.failed_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.requeue_stale_tasks()
        for shard_id in range(self.config.total_shards):
            name = shard_name(shard_id)
            if self.config.cached_shard_is_ready(self.config.cached_shards_dir / name):
                (self.done_dir / name).write_text("ok\n")
                continue
            has_task = (
                (self.pending_dir / name).exists()
                or (self.done_dir / name).exists()
                or any(self.running_dir.glob(f"{name}.*.json"))
            )
            if not has_task:
                (self.pending_dir / name).write_text(json.dumps({"shard_id": shard_id}) + "\n")

    def claim(self) -> Optional[Tuple[int, Path]]:
        """原子领取一个任务。"""
        self.requeue_stale_tasks()
        for task_path in sorted(self.pending_dir.iterdir()):
            name = task_path.name
            running_path = self.running_dir / f"{name}.{socket.gethostname()}.{os.getpid()}.json"
            try:
                os.rename(task_path, running_path)
            except FileNotFoundError:
                continue
            shard_id = int(name.split("_")[1])
            self.touch_heartbeat(running_path)
            return shard_id, running_path
        return None

    def touch_heartbeat(self, running_path: Path) -> None:
        """更新任务心跳。"""
        info = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "updated_at": time.time(),
        }
        running_path.write_text(json.dumps(info) + "\n")

    def complete(self, running_path: Path, shard_id: int) -> None:
        """标记任务完成。"""
        (self.done_dir / shard_name(shard_id)).write_text("ok\n")
        running_path.unlink(missing_ok=True)

    def fail(self, running_path: Path, shard_id: int) -> None:
        """标记任务失败。"""
        failed_path = self.failed_dir / f"{shard_name(shard_id)}.{int(time.time())}.json"
        if running_path.exists():
            os.rename(running_path, failed_path)

    def missing_shards(self) -> List[str]:
        """列出未完成分片。"""
        return [name for name in (shard_name(i) for i in range(self.config.total_shards))
                if not self.config.cached_shard_is_ready(self.config.cached_shards_dir / name)]

    def requeue_stale_tasks(self) -> None:
        """回收超时任务，容忍其他 worker 已推进快照中的任务状态。"""
        now = time.time()
        for running_path in list(self.running_dir.glob("shard_*.json")):
            try:
                try:
                    data = json.loads(running_path.read_text())
                    updated_at = float(data.get("updated_at", running_path.stat().st_mtime))
                except Exception:
                    updated_at = running_path.stat().st_mtime
                if now - updated_at < self.config.stale_task_seconds:
                    continue
                name = running_path.name.split(".", 1)[0]
                if self.config.cached_shard_is_ready(self.config.cached_shards_dir / name):
                    running_path.unlink(missing_ok=True)
                    (self.done_dir / name).write_text("ok\n")
                else:
                    os.replace(running_path, self.pending_dir / name)
            except FileNotFoundError:
                # glob() 快照后文件可能被其他 worker 完成、回收或移动。
                continue


class CachedShardBuilder:
    """构建单个 cached shard。"""

    def __init__(self, config: BuildConfig, timing: TimingRecorder):
        """保存构建配置和阶段计时器。"""
        self.config = config
        self.timing = timing

    def build(self, shard_id: int, heartbeat=None) -> None:
        """物化单分片role权重，调用指定源码分支导出cached shard。"""
        name = shard_name(shard_id)
        input_jsonl = self.config.raw_shards_dir / f"{name}.jsonl"
        shared_out = self.config.cached_shards_dir / name
        if self.config.cached_shard_is_ready(shared_out):
            print(f"cached shard already exists: {shared_out}", flush=True)
            return
        if not input_jsonl.is_file():
            raise FileNotFoundError(f"missing shard input: {input_jsonl}")
        self.config.cached_shards_dir.mkdir(parents=True, exist_ok=True)
        local_root = self.config.local_work_root / self.config.run_id
        local_out = local_root / f"o{name[-4:]}"
        local_cache = local_root / f"c{name[-4:]}"
        local_tmp = local_root / f"t{name[-4:]}"
        weighted_jsonl = local_root / f"w{name[-4:]}.jsonl"
        for path in (local_cache, local_tmp):
            path.mkdir(parents=True, exist_ok=True)
        if local_out.exists():
            self._safe_rmtree(local_out, self.config.local_work_root)
        weighted_jsonl.unlink(missing_ok=True)
        if shared_out.exists():
            self._safe_rmtree(shared_out, self.config.cached_shards_dir)
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        env.update({
            "PYTHONPATH": str(self.config.ms_swift_repo) + (os.pathsep + pythonpath if pythonpath else ""),
            "LD_LIBRARY_PATH": f"{self.config.cuda_fallback_ld_path}:{env.get('LD_LIBRARY_PATH', '')}",
            "MODELSCOPE_CACHE": env.get("MODELSCOPE_CACHE", str(local_cache / "ms")),
            "HF_DATASETS_CACHE": str(local_cache / "ds"),
            "HF_HOME": str(local_cache / "hf"),
            "TMPDIR": str(local_tmp),
            "TOKENIZERS_PARALLELISM": "false",
        })
        for key in ("MODELSCOPE_CACHE", "HF_DATASETS_CACHE", "HF_HOME", "TMPDIR"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)

        export_input = input_jsonl
        timing_fields = {
            "shard": name,
            "raw_bytes": input_jsonl.stat().st_size,
            "build_fingerprint": self.config.build_fingerprint,
        }
        try:
            if self.config.materialize_role_weights:
                repo_path = str(self.config.ms_swift_repo)
                if repo_path not in sys.path:
                    sys.path.insert(0, repo_path)
                from swift.loss_scale.role import inject_role_loss_scale_jsonl

                with self.timing.phase("shard_role_inject", **timing_fields):
                    inject_role_loss_scale_jsonl(
                        str(input_jsonl), str(self.config.role_loss_config), str(weighted_jsonl))
                    if heartbeat:
                        heartbeat()
                export_input = weighted_jsonl

            command = [
                str(self.config.swift_bin), "export",
                "--model", str(self.config.model),
                "--dataset", str(export_input),
                "--template", self.config.template,
                "--agent_template", self.config.agent_template,
                "--loss_scale", self.config.loss_scale,
                "--max_length", str(self.config.max_length),
                "--truncation_strategy", self.config.truncation_strategy,
                "--dataset_num_proc", str(self.config.dataset_num_proc),
                "--dataset_shuffle", bool_arg(self.config.dataset_shuffle),
                "--data_seed", str(self.config.data_seed),
                "--split_dataset_ratio", str(self.config.split_dataset_ratio),
                "--load_from_cache_file", bool_arg(self.config.load_from_cache_file),
                "--to_cached_dataset", bool_arg(self.config.to_cached_dataset),
                "--output_dir", str(local_out),
            ]
            if self.config.role_loss_config is not None:
                command.extend(["--role_loss_config", str(self.config.role_loss_config)])
            with self.timing.phase("shard_export", **timing_fields):
                self._run_with_heartbeat(command, env, heartbeat)
            self._copy_to_shared(local_out, shared_out, timing_fields, heartbeat)
            log_progress(f"cached shard ready: {shared_out}")
        finally:
            weighted_jsonl.unlink(missing_ok=True)

    def _run_with_heartbeat(self, command: List[str], env: Dict[str, str], heartbeat) -> None:
        """运行命令并维护心跳。"""
        print("run sh: `" + " ".join(shlex.quote(x) for x in command) + "`", flush=True)
        process = subprocess.Popen(command, env=env)
        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                return
            if heartbeat:
                heartbeat()
            time.sleep(self.config.heartbeat_seconds)

    def _copy_to_shared(
        self,
        local_out: Path,
        shared_out: Path,
        timing_fields: Dict[str, object],
        heartbeat=None,
    ) -> None:
        """记录复制槽位等待与共享盘复制耗时后发布结果。"""
        with self.timing.phase("shard_copy_wait", **timing_fields):
            lock_dir = self._acquire_copy_slot(heartbeat)
        try:
            with self.timing.phase("shard_copy", **timing_fields):
                if heartbeat:
                    heartbeat()
                tmp_shared = self.config.cached_shards_dir / f"{shared_out.name}.tmp.{socket.gethostname()}.{os.getpid()}"
                shutil.rmtree(tmp_shared, ignore_errors=True)
                shutil.copytree(local_out, tmp_shared)
                write_json(tmp_shared / "build_manifest.json", {
                    "build_fingerprint": self.config.build_fingerprint,
                    "build_metadata": self.config.build_metadata,
                    "shard": shared_out.name,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                if heartbeat:
                    heartbeat()
                (tmp_shared / "_SUCCESS").write_text("ok\n")
                os.rename(tmp_shared, shared_out)
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)

    def _acquire_copy_slot(self, heartbeat=None) -> Path:
        """获取共享复制槽位。"""
        lock_root = self.config.work_root / "copy_locks"
        lock_root.mkdir(parents=True, exist_ok=True)
        while True:
            if heartbeat:
                heartbeat()
            for slot in range(self.config.copy_parallel):
                lock = lock_root / f"slot_{slot}.lock"
                try:
                    lock.mkdir()
                    (lock / "owner").write_text(f"{socket.gethostname()} {os.getpid()}\n")
                    return lock
                except FileExistsError:
                    continue
            time.sleep(10)

    @staticmethod
    def _safe_rmtree(path: Path, root: Path) -> None:
        """安全删除指定目录。"""
        if not path.expanduser().resolve().is_relative_to(root.expanduser().resolve()):
            raise SystemExit(f"refuse to remove unsafe path: {path}")
        shutil.rmtree(path)


class WorkerNode:
    """消费分片任务的节点进程。"""

    def __init__(self, config: BuildConfig):
        """初始化 worker 组件。"""
        self.config = config
        self.timing = TimingRecorder(config, role="worker")
        self.queue = ShardTaskQueue(config)
        self.builder = CachedShardBuilder(config, self.timing)

    def run(self) -> None:
        """循环消费分片任务。"""
        self.config.validate()
        while True:
            claimed = self.queue.claim()
            if not claimed:
                log_progress(f"{socket.gethostname()}: no pending shard")
                return
            shard_id, running_path = claimed
            name = shard_name(shard_id)
            try:
                with self.timing.phase("shard_total", shard=name):
                    self.builder.build(shard_id, heartbeat=lambda: self.queue.touch_heartbeat(running_path))
                self.queue.complete(running_path, shard_id)
            except Exception as exc:
                self.queue.fail(running_path, shard_id)
                log_progress(f"{socket.gethostname()}: failed {name}: {exc}")
                raise


class RemoteWorkerLauncher:
    """启动远端 worker。"""

    def __init__(self, config: BuildConfig):
        """保存启动配置。"""
        self.config = config

    def read_hosts(self) -> List[str]:
        """读取候选节点列表。"""
        if not self.config.ip_file.is_file():
            raise SystemExit(f"missing ip file: {self.config.ip_file}")
        hosts = []
        for line in self.config.ip_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                hosts.append(line)
        if not hosts:
            raise SystemExit("empty host list")
        return hosts

    def launch_and_wait(self) -> bool:
        """启动并等待远端 worker。"""
        hosts = self.read_hosts()
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        processes = []
        remote_env = self.config.runtime_env_for_worker()
        config_arg = shlex.quote(str(self.config.config_path))
        for index, host in enumerate(hosts):
            safe_host = host.replace(".", "_")
            log_file = self.config.log_dir / f"worker_{index:03d}_{safe_host}.log"
            worker_env = {
                **remote_env,
                "WORKER_INDEX": f"{index:03d}",
                "WORKER_IP": host,
            }
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in worker_env.items())
            command = (
                f"cd {shlex.quote(str(self.config.script_dir))} && "
                f"{env_prefix} {shlex.quote(str(self.config.python_bin))} "
                f"./{Path(__file__).name} worker --config {config_arg}"
            )
            log_progress(f"launch worker {index:03d} on {host}, log={log_file}")
            log_handle = log_file.open("wb")
            process = subprocess.Popen([
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "ConnectionAttempts=1",
                host,
                command,
            ], stdout=log_handle, stderr=subprocess.STDOUT)
            processes.append((process, log_handle, host))
        failed = False
        remaining = set(range(len(processes)))
        last_progress_at = 0.0
        while remaining:
            now = time.monotonic()
            if now - last_progress_at >= self.config.progress_interval_seconds:
                queue_root = self.config.work_root / "task_queue" / self.config.run_id
                counts = {
                    name: sum(1 for _ in (queue_root / name).glob("*")) if (queue_root / name).is_dir() else 0
                    for name in ("pending", "running", "done", "failed")
                }
                cached_success = sum(1 for _ in self.config.cached_shards_dir.glob("shard_*/_SUCCESS"))
                log_progress(
                    "progress: "
                    f"done={counts['done']} running={counts['running']} pending={counts['pending']} "
                    f"failed={counts['failed']} cached_success={cached_success}/{self.config.total_shards}"
                )
                last_progress_at = now
            for index in list(remaining):
                process, log_handle, host = processes[index]
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                if code == 255:
                    log_progress(f"worker unavailable on {host}: ssh exit=255; skipped")
                elif code != 0:
                    failed = True
                    log_progress(f"worker failed on {host}: exit={code}")
                remaining.remove(index)
            if remaining:
                time.sleep(5)
        return not failed


class CachedDatasetMerger:
    """合并 cached shards。"""

    def __init__(self, config: BuildConfig, timing: TimingRecorder):
        """保存合并配置和阶段计时器。"""
        self.config = config
        self.timing = timing

    def merge(self) -> None:
        """一次性拼接全部 cached shards 并单次落盘，避免中间分组重复写盘。"""
        out_dir = self.config.final_out_dir
        missing = [shard_name(i) for i in range(self.config.total_shards)
                   if not self.config.cached_shard_is_ready(self.config.cached_shards_dir / shard_name(i))]
        if missing:
            raise SystemExit(f"missing cached shards: {' '.join(missing)}")
        if (out_dir / "_SUCCESS").exists():
            manifest_path = out_dir / "manifest.json"
            if not manifest_path.is_file():
                raise SystemExit(f"cached dataset缺少构建指纹，拒绝复用：{out_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("build_fingerprint") != self.config.build_fingerprint:
                raise SystemExit(f"cached dataset构建指纹不一致，拒绝复用：{out_dir}")
            self.timing.event("phase_skipped", phase="cached_merge", reason="already_ready")
            log_progress(f"cached dataset already exists: {out_dir}")
            return
        if out_dir.exists():
            raise SystemExit(f"output dir already exists: {out_dir}")

        from datasets import concatenate_datasets, load_from_disk
        shard_dirs = [self.config.cached_shards_dir / shard_name(i) / "train"
                      for i in range(self.config.total_shards)]
        # 一次性 concat 需同时映射全部分片的 arrow 文件，按实际文件数抬升 fd 上限。
        arrow_files = sum(len(list(shard_dir.glob("*.arrow"))) for shard_dir in shard_dirs)
        raise_open_file_limit(arrow_files + 64)
        # 仅当 save_num_proc>1 时传 num_proc；否则不传，走 datasets 单进程分支，
        # 避开其保存多进程硬编码 spawn 时无法重载脚本式主程序的问题。
        save_kwargs = {"num_proc": self.config.save_num_proc} if self.config.save_num_proc > 1 else {}
        tmp_dir = out_dir.with_name(f"{out_dir.name}.tmp.{os.getpid()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            with self.timing.phase("merge_final", shard_count=len(shard_dirs), arrow_files=arrow_files):
                final_dataset = concatenate_datasets([load_from_disk(str(shard_dir)) for shard_dir in shard_dirs])
                final_dataset.save_to_disk(str(tmp_dir / "train"), **save_kwargs)
                write_json(tmp_dir / "manifest.json", {
                    "num_shards": self.config.total_shards,
                    "num_rows": len(final_dataset),
                    "shards_dir": str(self.config.cached_shards_dir),
                    "save_num_proc": self.config.save_num_proc,
                    "build_fingerprint": self.config.build_fingerprint,
                    "build_metadata": self.config.build_metadata,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                (tmp_dir / "_SUCCESS").write_text("ok\n")
                os.rename(tmp_dir, out_dir)
            log_progress(f"merged cached dataset: {out_dir / 'train'} rows={len(final_dataset)}")
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise


class DistributedBuildRunner:
    """编排完整构建流程。"""

    def __init__(self, config: BuildConfig):
        """初始化主控组件。"""
        self.config = config
        self.queue = ShardTaskQueue(config)
        self.timing = TimingRecorder(config, role="controller")

    def run(self) -> None:
        """执行分片、构建和合并流程。"""
        self.config.validate()
        self.config.work_root.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        log_progress(f"run_id: {self.config.run_id}")
        log_progress(f"ms_swift_repo: {self.config.ms_swift_repo}")
        log_progress(f"ms_swift_commit: {self.config.swift_commit}")
        log_progress(f"total_shards: {self.config.total_shards}")
        log_progress(f"raw_shards: {self.config.raw_shards_dir}")
        log_progress(f"cached_shards: {self.config.cached_shards_dir}")
        log_progress(f"final_out: {self.config.final_out_dir}")
        try:
            RawShardSplitter(self.config, self.timing).ensure_split()
            log_progress(f"build_fingerprint: {self.config.build_fingerprint}")
            missing = []
            worker_failed = False
            for round_index in range(1, self.config.max_worker_rounds + 1):
                log_progress(f"worker round {round_index}/{self.config.max_worker_rounds}")
                with self.timing.phase("worker_round", round=round_index):
                    self.queue.prepare()
                    worker_failed = not RemoteWorkerLauncher(self.config).launch_and_wait()
                missing = self.queue.missing_shards()
                if not missing:
                    break
                log_progress(f"missing cached shards after round {round_index}: {' '.join(missing)}")
                if not worker_failed:
                    break
            if missing:
                raise SystemExit(f"missing cached shards: {' '.join(missing)}")
            log_progress(f"all cached shards ready: {self.config.cached_shards_dir}")
            if self.config.auto_merge:
                with self.timing.phase("cached_merge"):
                    CachedDatasetMerger(self.config, self.timing).merge()
            log_progress(f"done. logs: {self.config.log_dir}")
        finally:
            self.timing.write_summary()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    default_config = Path(__file__).resolve().with_name(DEFAULT_CONFIG_NAME)
    parser = argparse.ArgumentParser(description="使用动态任务队列构建 cached dataset。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="启动主控并拉起远端 worker")
    run_parser.add_argument("--config", default=str(default_config), help="cached dataset 的 TOML 配置路径")
    split_parser = subparsers.add_parser("split", help="仅生成原始 JSONL 分片")
    split_parser.add_argument("--config", default=str(default_config), help="cached dataset 的 TOML 配置路径")
    worker_parser = subparsers.add_parser("worker", help="在单个节点消费分片任务")
    worker_parser.add_argument("--config", default=str(default_config), help="cached dataset 的 TOML 配置路径")
    return parser.parse_args()


def main() -> None:
    """执行命令入口。"""
    args = parse_args()
    if args.command == "run":
        config = BuildConfig.from_file(Path(args.config))
        DistributedBuildRunner(config).run()
    elif args.command == "split":
        config = BuildConfig.from_file(Path(args.config))
        config.validate()
        config.work_root.mkdir(parents=True, exist_ok=True)
        timing = TimingRecorder(config, role="controller")
        try:
            RawShardSplitter(config, timing).ensure_split()
        finally:
            timing.write_summary()
    elif args.command == "worker":
        config = BuildConfig.from_file(Path(args.config))
        WorkerNode(config).run()


if __name__ == "__main__":
    main()

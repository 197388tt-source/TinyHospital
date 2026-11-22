"""Atomic per-case results and committed memory snapshots for restart."""
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class ExperimentProgress:
    def __init__(self, directory, stage, config, resume, *, allow_code_change=False):
        if allow_code_change and not resume:
            raise ValueError('接受代码变更需要 resume')
        self.directory = Path(directory)
        self.path = self.directory / stage / 'progress.json'
        self.memory_paths = [self.directory / n for n in
                             ('golden_experience.json', 'reflection_memory.json')]
        self.lock = (self.directory / '.experiment.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                if not resume:
                    raise ValueError('已有实验进度，请加 --resume 续跑，或使用新的实验目录')
                self.state = json.loads(self.path.read_text())
                code_changed = False
                if self.state['config'] != config:
                    previous = self.state['config']
                    without_code = lambda value: {k: v for k, v in value.items() if k != 'code_sha256'}
                    if (not allow_code_change or without_code(previous) != without_code(config)
                            or not isinstance(previous.get('code_sha256'), dict)
                            or not isinstance(config.get('code_sha256'), dict)):
                        raise ValueError('数据、代码或配置与原实验不同，不能续跑；仅代码变更可使用 --resume --allow-code-change')
                    change = {'timestamp': datetime.now(timezone.utc).isoformat(),
                              'previous_code_sha256': previous['code_sha256'],
                              'new_code_sha256': config['code_sha256'],
                              'completed_case_ids': list(self.state['completed_case_ids'])}
                    self.state = {**self.state, 'config': config,
                                  'code_changes': self.state.get('code_changes', []) + [change]}
                    code_changed = True
                if stage == 'train':
                    # An interrupted case may have written memory before its result committed.
                    # Restore the last completed case, then rebuild/reuse the semantic index.
                    for path, rows in zip(self.memory_paths, self.state['memory']):
                        atomic_json(path, rows)
                elif self.read_memory() != self.state['memory']:
                    raise ValueError('测试经验库已变化，不能在同一实验进度上续跑')
                if code_changed:
                    atomic_json(self.path, self.state)
                    print('[续跑] 已记录代码版本变更；保留此前病例结果和经验。', flush=True)
            else:
                if resume:
                    raise ValueError('没有 progress.json，首次运行请去掉 --resume')
                self.state = {'config': config, 'completed_case_ids': [],
                              'results': [], 'memory': self.read_memory()}
                atomic_json(self.path, self.state)
        except BaseException:
            self.close()
            raise

    def read_memory(self):
        return [json.loads(path.read_text()) for path in self.memory_paths]

    def commit(self, outcome):
        case_id = outcome['case_id']
        if case_id in self.state['completed_case_ids']:
            raise ValueError('病例已完成，禁止重复提交')
        state = {**self.state,
                 'completed_case_ids': self.state['completed_case_ids'] + [case_id],
                 'results': self.state['results'] + [outcome],
                 'memory': self.read_memory()}
        atomic_json(self.path, state)
        self.state = state
        print(f'[进度已保存] 病例 {case_id}；累计 {len(state["results"])} 例', flush=True)

    def close(self):
        self.lock.close()

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { checkOptions, MEMORY_ROWS, MEMORY_RULES, recommend as recommendWith, resolveMemoryRows } from './memory-picker.mjs';

const recommend = (choice) => recommendWith(MEMORY_RULES, choice);

const choice = (overrides) => ({ os: 'linux', gpu: 'nvidia', series: 'ada', vram: '16', format: 'full', ...overrides });

describe('recommend', () => {
  it('sends a Windows NVIDIA card short on VRAM to sysmem fallback, FP8 Storage and a smaller build', () => {
    assert.deepEqual(recommend(choice({ os: 'windows', vram: '8' })), [
      'sysmem-fallback',
      'low-vram-mode',
      'fp8-storage',
      'quantized-build',
      'tiled-decode',
      'still-oom',
      'page-file',
    ]);
  });

  it('offers FP8 Compute for fp8 checkpoints only from RTX 40 on, or on AMD where the probe decides', () => {
    assert.ok(recommend(choice({ format: 'fp8', series: 'ada' })).includes('fp8-compute'));
    assert.ok(!recommend(choice({ format: 'fp8', series: 'ampere' })).includes('fp8-compute'));
    assert.ok(recommend(choice({ format: 'fp8', gpu: 'amd' })).includes('fp8-compute'));
    assert.ok(!recommend(choice({ format: 'fp8', gpu: 'intel' })).includes('fp8-compute'));
  });

  it('offers int8 Compute for int8 builds on NVIDIA from RTX 30 on', () => {
    assert.ok(recommend(choice({ format: 'int8', series: 'ampere' })).includes('int8-compute'));
    assert.ok(!recommend(choice({ format: 'int8', series: 'pre-ampere' })).includes('int8-compute'));
    assert.ok(!recommend(choice({ format: 'int8', gpu: 'amd' })).includes('int8-compute'));
  });

  it('keeps VRAM settings away from Apple silicon and the CPU', () => {
    for (const gpu of ['apple', 'none']) {
      const ids = recommend(choice({ gpu, os: gpu === 'apple' ? 'macos' : 'linux' }));
      assert.equal(ids[0], 'system-memory');
      for (const id of ['low-vram-mode', 'fp8-storage', 'tiled-decode', 'still-oom']) assert.ok(!ids.includes(id), id);
    }
  });

  it('presents NF4 as working only where bitsandbytes is installed and runs: NVIDIA and AMD', () => {
    for (const gpu of ['apple', 'intel', 'none']) {
      const ids = recommend(choice({ gpu, format: 'nf4' }));
      assert.ok(ids.includes('nf4-support') && !ids.includes('already-quantized'), gpu);
    }
    for (const gpu of ['nvidia', 'amd']) {
      const ids = recommend(choice({ gpu, format: 'nf4' }));
      assert.ok(!ids.includes('nf4-support') && ids.includes('already-quantized'), gpu);
    }
  });

  it('gives every int8 setup a card: int8 Compute where it runs, unpacked layers elsewhere', () => {
    const cases = [
      [{ series: 'ampere' }, 'int8-compute'],
      [{ series: 'pre-ampere' }, 'int8-unpacked'],
      [{ gpu: 'amd' }, 'int8-unpacked'],
      [{ os: 'windows', gpu: 'arm64' }, 'int8-unpacked'],
      [{ os: 'macos', gpu: 'apple' }, 'int8-unpacked'],
    ];
    for (const [overrides, expected] of cases) {
      const ids = recommend(choice({ format: 'int8', ...overrides }));
      assert.ok(ids.includes(expected), JSON.stringify(overrides));
      assert.ok(!ids.includes(expected === 'int8-compute' ? 'int8-unpacked' : 'int8-compute'));
    }
  });

  it('points to tiled decoding up to 16 GB, where SD 1.5 and SDXL decodes are still not tiled automatically', () => {
    assert.ok(recommend(choice({ vram: '16' })).includes('tiled-decode'));
    assert.ok(!recommend(choice({ vram: '24' })).includes('tiled-decode'));
  });

  it('tells MXFP8 owners to switch FP8 Storage off instead of offering it', () => {
    const ids = recommend(choice({ format: 'mxfp8' }));
    assert.ok(ids.includes('mxfp8-off'));
    assert.ok(!ids.includes('fp8-storage'));
  });

  it('treats Windows on ARM64 as the NVIDIA RTX Spark GPU it runs on', () => {
    const ids = recommend(choice({ os: 'windows', gpu: 'arm64', format: 'fp8' }));
    assert.ok(ids.includes('low-vram-mode') && ids.includes('fp8-storage-installed') && ids.includes('fp8-compute'));
    assert.ok(!ids.includes('sysmem-fallback'));
  });
});

describe('resolveMemoryRows', () => {
  it('keeps valid stored choices and replaces junk with each row default', () => {
    assert.deepEqual(resolveMemoryRows(MEMORY_ROWS, { series: 'ampere', vram: '24', format: 'gguf' }), {
      series: 'ampere',
      vram: '24',
      format: 'gguf',
    });
    assert.deepEqual(resolveMemoryRows(MEMORY_ROWS, { vram: '7', format: 1 }), { series: 'ada', vram: '8', format: 'full' });
    assert.deepEqual(resolveMemoryRows(MEMORY_ROWS, 'junk'), { series: 'ada', vram: '8', format: 'full' });
  });
});

describe('checkOptions', () => {
  const rendered = (ids) => ids.map((id) => `<article data-memory-option="${id}"></article>`).join('');

  it('accepts a page that renders every option once', () => {
    assert.deepEqual(checkOptions(rendered(MEMORY_RULES.map((rule) => rule.id))), { missing: [], unknown: [] });
  });

  it('names forgotten, misspelled and repeated options', () => {
    const ids = MEMORY_RULES.map((rule) => rule.id).filter((id) => id !== 'page-file');
    assert.deepEqual(checkOptions(rendered([...ids, 'pagefile', 'still-oom'])), {
      missing: ['page-file'],
      unknown: ['pagefile', 'still-oom'],
    });
  });
});

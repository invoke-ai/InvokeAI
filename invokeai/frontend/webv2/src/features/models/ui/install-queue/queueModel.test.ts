import type { ModelInstallJob } from '@features/models/core/types';

import { describe, expect, it } from 'vitest';

import {
  buildInstallQueueRows,
  canRestartFailedParts,
  describeEta,
  getInstallJobDisplayName,
  getInstallJobSourceLabel,
  getInstallJobSourceString,
  getInstallRowStatus,
  getProblemDownloadParts,
  resolveInstallProgress,
  summarizeInstallQueue,
} from './queueModel';

const job = (overrides: Partial<ModelInstallJob> & { id: number }): ModelInstallJob => ({
  source: `https://example/${overrides.id}.safetensors`,
  status: 'waiting',
  ...overrides,
});

describe('install row status', () => {
  it('folds backend statuses into row statuses', () => {
    expect(getInstallRowStatus(job({ id: 1, status: 'downloads_done' }))).toBe('installing');
    expect(getInstallRowStatus(job({ id: 1, status: 'running' }))).toBe('installing');
    expect(getInstallRowStatus(job({ id: 1, status: 'waiting' }))).toBe('queued');
    expect(getInstallRowStatus(job({ id: 1, status: 'completed' }))).toBe('installed');
  });

  it('recognises credential failures from the reason, message, or traceback', () => {
    expect(
      getInstallRowStatus(job({ error: 'boom', error_reason: 'DuplicateModelException', id: 1, status: 'error' }))
    ).toBe('failed');
    expect(getInstallRowStatus(job({ error: 'Unauthorized', error_reason: 'HTTPError', id: 1, status: 'error' }))).toBe(
      'unauthorized'
    );
    expect(
      getInstallRowStatus(
        job({
          error: "'org/repo' not found. See trace for details.",
          error_reason: 'UnknownMetadataException',
          error_traceback: 'huggingface_hub.errors.GatedRepoError: 401 Client Error',
          id: 1,
          status: 'error',
        })
      )
    ).toBe('unauthorized');
    // Path segments and file names must not read as credential failures.
    expect(
      getInstallRowStatus(job({ error: 'No such file: /models/gated/model-401.safetensors', id: 1, status: 'error' }))
    ).toBe('failed');
  });
});

describe('problem download parts', () => {
  const parts = [
    {
      bytes: 10,
      resume_message: 'Resume refused by server.',
      resume_required: true,
      source: 'https://x/a.bin',
      status: 'paused',
      total_bytes: 100,
    },
    { bytes: 0, source: 'https://x/b.bin', status: 'error', total_bytes: 100 },
    { bytes: 100, source: 'https://x/c.bin', status: 'completed', total_bytes: 100 },
  ];

  it('surfaces resume-required and errored parts with their restartable source URL', () => {
    expect(getProblemDownloadParts(job({ download_parts: parts, id: 1, status: 'paused' }))).toEqual([
      {
        fileName: 'a.bin',
        key: 'https://x/a.bin',
        message: 'Resume refused by server.',
        resumeRequired: true,
        url: 'https://x/a.bin',
      },
      { fileName: 'b.bin', key: 'https://x/b.bin', message: null, resumeRequired: false, url: 'https://x/b.bin' },
    ]);
  });

  it('only offers a part restart for live jobs', () => {
    expect(canRestartFailedParts(job({ download_parts: parts, id: 1, status: 'error' }))).toBe(true);
    expect(canRestartFailedParts(job({ download_parts: parts, id: 1, status: 'cancelled' }))).toBe(false);
    expect(canRestartFailedParts(job({ id: 1, status: 'error' }))).toBe(false);
  });
});

describe('install job labels', () => {
  it('names hf subfolder installs by file and labels them by repo', () => {
    const hf = job({
      id: 1,
      source: { repo_id: 'InvokeAI/flux_dev', subfolder: 'transformer/base/flux1-dev.safetensors', type: 'hf' },
    });

    expect(getInstallJobDisplayName(hf)).toBe('flux1-dev.safetensors');
    expect(getInstallJobSourceLabel(hf)).toBe('InvokeAI/flux_dev :: transformer/base/flux1-dev.safetensors');
  });

  it('prefers the configured name and falls back to the last path segment', () => {
    expect(getInstallJobDisplayName(job({ config_in: { name: 'Elegance' }, id: 1 }))).toBe('Elegance');
    expect(
      getInstallJobDisplayName(job({ id: 1, source: { path: '/mnt/models/lora/a.safetensors', type: 'local' } }))
    ).toBe('a.safetensors');
  });

  it('rebuilds the install endpoint source string for resubmission', () => {
    expect(
      getInstallJobSourceString(
        job({ id: 1, source: { repo_id: 'org/repo', subfolder: 'vae', type: 'hf', variant: 'fp16' } })
      )
    ).toBe('org/repo:fp16::vae');
    expect(
      getInstallJobSourceString(
        job({ id: 1, source: { provider_id: 'civitai', provider_model_id: '42', type: 'external' } })
      )
    ).toBe('external://civitai/42');
    expect(getInstallJobSourceString(job({ id: 1, source: { type: 'url', url: 'https://x/y.gguf' } }))).toBe(
      'https://x/y.gguf'
    );
  });
});

describe('install queue rows', () => {
  const jobs = [
    job({ id: 1, status: 'completed' }),
    job({ id: 2, status: 'waiting' }),
    job({ id: 3, status: 'downloading' }),
    job({ error: 'Unauthorized', error_reason: 'HTTPError', id: 4, status: 'error' }),
    job({ id: 5, status: 'waiting' }),
    job({ id: 6, status: 'cancelled' }),
    job({ id: 7, status: 'paused' }),
    job({ id: 8, status: 'running' }),
  ];

  it('orders downloads, installs, queued in run order, paused, attention, installed, then cancelled', () => {
    const rows = buildInstallQueueRows(jobs, new Set());

    expect(rows.map((row) => row.job.id)).toEqual([3, 8, 2, 5, 7, 4, 1, 6]);
    expect(rows.find((row) => row.job.id === 5)?.queuePosition).toBe(2);
    expect(rows.find((row) => row.job.id === 3)?.queuePosition).toBeNull();
  });

  it('hides dismissed jobs and summarises the rest', () => {
    const rows = buildInstallQueueRows(jobs, new Set([6]));

    expect(rows.some((row) => row.job.id === 6)).toBe(false);
    expect(summarizeInstallQueue(rows)).toEqual({
      attention: 1,
      downloading: 1,
      installed: 1,
      installing: 1,
      paused: 1,
      queued: 2,
      settled: 2,
    });
  });
});

describe('install progress', () => {
  it('prefers live bytes and derives an ETA from the smoothed rate', () => {
    const progress = resolveInstallProgress(job({ bytes: 10, id: 1, total_bytes: 100 }), {
      bytes: 40_000_000,
      bytesPerSecond: 20_000_000,
      totalBytes: 100_000_000,
    });

    expect(progress).toEqual({
      bytes: 40_000_000,
      bytesPerSecond: 20_000_000,
      etaSeconds: 3,
      ratio: 0.4,
      totalBytes: 100_000_000,
    });
  });

  it('falls back to the REST snapshot without a rate', () => {
    expect(resolveInstallProgress(job({ bytes: 50, id: 1, total_bytes: 200 }), null)).toEqual({
      bytes: 50,
      bytesPerSecond: null,
      etaSeconds: null,
      ratio: 0.25,
      totalBytes: 200,
    });
    expect(resolveInstallProgress(job({ id: 1 }), null).ratio).toBeNull();
  });

  it('withholds the ETA while the rate is below the floor', () => {
    const progress = resolveInstallProgress(job({ id: 1 }), { bytes: 10, bytesPerSecond: 12, totalBytes: 10_000_000 });

    expect(progress.etaSeconds).toBeNull();
  });

  it('buckets ETAs coarsely', () => {
    expect(describeEta(3)).toEqual({ count: 5, unit: 'seconds' });
    expect(describeEta(61)).toEqual({ count: 2, unit: 'minutes' });
    expect(describeEta(7200)).toEqual({ count: 2, unit: 'hours' });
  });
});

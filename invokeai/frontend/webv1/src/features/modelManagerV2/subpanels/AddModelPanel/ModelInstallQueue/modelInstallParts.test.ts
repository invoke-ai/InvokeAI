import type { ModelInstallJob } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { canResumeDownload, describeDownloadPartError, getProblemDownloadParts } from './modelInstallParts';

type DownloadPart = NonNullable<ModelInstallJob['download_parts']>[number];

const part = (overrides: Partial<DownloadPart>): DownloadPart =>
  ({
    bytes: 4,
    source: 'https://example.com/a.safetensors',
    status: 'running',
    total_bytes: 10,
    ...overrides,
  }) as DownloadPart;

const pausedJob = (download_parts: DownloadPart[]): ModelInstallJob =>
  ({ download_parts, id: 1, source: { type: 'url', url: 'https://example.com' }, status: 'paused' }) as ModelInstallJob;

const traceback = 'Traceback (most recent call last):\n  File "x.py", line 1\nConnectionError: reset by peer\n';

describe('model install download parts', () => {
  it('resumes past a transiently failed part, but not past a refused range', () => {
    const failed = part({ error: traceback, error_type: 'ConnectionError(reset by peer)', status: 'error' });
    const refused = part({ resume_required: true, status: 'paused' });

    expect(getProblemDownloadParts(pausedJob([part({}), failed]))).toEqual([failed]);
    expect(canResumeDownload(pausedJob([part({}), failed]))).toBe(true);
    expect(canResumeDownload(pausedJob([failed, refused]))).toBe(false);
  });

  it("explains an errored part by its exception, falling back to the traceback's last line", () => {
    expect(describeDownloadPartError(part({ error: traceback, error_type: 'ConnectionError(reset by peer)' }))).toBe(
      'ConnectionError(reset by peer)'
    );
    expect(describeDownloadPartError(part({ error: traceback }))).toBe('ConnectionError: reset by peer');
    expect(describeDownloadPartError(part({ resume_message: 'Partial file missing.' }))).toBe('Partial file missing.');
    expect(describeDownloadPartError(part({}))).toBeNull();
  });
});

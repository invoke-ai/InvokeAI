import type * as transportModule from '@platform/transport/http';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof transportModule>()),
  apiFetch: transport.apiFetch,
  apiFetchJson: transport.apiFetchJson,
}));

import { isProjectConfirmedAbsent } from './api';

beforeEach(() => {
  transport.apiFetch.mockReset();
  transport.apiFetchJson.mockReset();
  accountLifecycle.activate('project-api-test-user');
});

describe('isProjectConfirmedAbsent', () => {
  it('confirms absence only from an authoritative not-found response in the current account scope', async () => {
    const owner = captureAccountScope();

    transport.apiFetchJson.mockRejectedValueOnce(new ApiError('not found', 404));

    await expect(isProjectConfirmedAbsent('project/1', owner)).resolves.toBe(true);
    expect(transport.apiFetchJson).toHaveBeenCalledWith('/api/v1/projects/project%2F1', { signal: owner.signal });
  });

  it('does not confirm absence when the project exists', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project-1' });

    await expect(isProjectConfirmedAbsent('project-1', captureAccountScope())).resolves.toBe(false);
  });

  it('does not authorize cleanup after the owning account scope expires', async () => {
    const owner = captureAccountScope();

    transport.apiFetchJson.mockImplementationOnce(() => {
      accountLifecycle.invalidate();
      return Promise.reject(new ApiError('not found', 404));
    });

    await expect(isProjectConfirmedAbsent('project-1', owner)).resolves.toBe(false);
  });

  it('does not treat transport or server failures as proof of absence', async () => {
    transport.apiFetchJson.mockRejectedValueOnce(new ApiError('unavailable', 503));

    await expect(isProjectConfirmedAbsent('project-1', captureAccountScope())).resolves.toBe(false);
  });
});

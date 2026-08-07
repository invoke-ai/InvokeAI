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

import { createProjectSettled, ProjectCreateAbsentError } from './api';

beforeEach(() => {
  transport.apiFetch.mockReset();
  transport.apiFetchJson.mockReset();
  accountLifecycle.activate('project-api-test-user');
});

/**
 * The caller has already uploaded a board's worth of media by the time this runs, and it deletes
 * that media if — and only if — this proves the project does not exist. Getting it wrong one way
 * leaves clutter; the other way guts a project that does exist. So absence has to be *proved*, never
 * assumed from silence.
 */
describe('createProjectSettled', () => {
  const request = { data: {}, name: 'Imported', project_id: 'project-1' };
  const post = () => ({ body: JSON.stringify(request), method: 'POST', signal: expect.anything() });

  it('returns the record a create succeeded with', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project-1' });

    await expect(createProjectSettled(request, captureAccountScope())).resolves.toMatchObject({
      project_id: 'project-1',
    });
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  /**
   * The case a `GET` cannot answer. The create may be mid-transaction, so a read of the id returns
   * 404 about a project that is moments from existing — and the caller deletes its media. A second
   * `POST` cannot commit before the first, so its answer is about a settled database.
   */
  it('adopts the project when a retried create finds the first one committed', async () => {
    transport.apiFetchJson
      .mockRejectedValueOnce(new TypeError('network error'))
      .mockRejectedValueOnce(new ApiError('conflict', 409))
      .mockResolvedValueOnce({ project_id: 'project-1', name: 'Imported' });

    await expect(createProjectSettled(request, captureAccountScope())).resolves.toMatchObject({
      project_id: 'project-1',
    });
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(1, '/api/v1/projects/', post());
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/projects/', post());
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(3, '/api/v1/projects/project-1', expect.anything());
  });

  it('creates the project when the first attempt never landed', async () => {
    transport.apiFetchJson
      .mockRejectedValueOnce(new ApiError('bad gateway', 502))
      .mockResolvedValueOnce({ project_id: 'project-1' });

    await expect(createProjectSettled(request, captureAccountScope())).resolves.toMatchObject({
      project_id: 'project-1',
    });
  });

  it('proves absence when the retry conflicts and the id is genuinely free', async () => {
    // Somebody else holds the staging board; our id was never written.
    transport.apiFetchJson
      .mockRejectedValueOnce(new TypeError('network error'))
      .mockRejectedValueOnce(new ApiError('conflict', 409))
      .mockRejectedValueOnce(new ApiError('not found', 404));

    await expect(createProjectSettled(request, captureAccountScope())).rejects.toBeInstanceOf(ProjectCreateAbsentError);
  });

  it('proves absence when the server answers with an outright refusal', async () => {
    transport.apiFetchJson.mockRejectedValueOnce(new ApiError('no such board', 404));

    await expect(createProjectSettled(request, captureAccountScope())).rejects.toBeInstanceOf(ProjectCreateAbsentError);
    // Deterministic: no retry, because the server already answered.
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('leaves an unresolved outcome unresolved rather than authorizing a rollback', async () => {
    const failure = new TypeError('network error');

    transport.apiFetchJson.mockRejectedValueOnce(failure).mockRejectedValueOnce(new TypeError('still offline'));

    // The original failure, not a proof of absence: unknown must never authorize deletion.
    await expect(createProjectSettled(request, captureAccountScope())).rejects.toBe(failure);
  });

  it('does not retry a create whose id the server would choose', async () => {
    const failure = new TypeError('network error');

    transport.apiFetchJson.mockRejectedValueOnce(failure);

    // A second POST would mint a second project rather than collide with the first.
    await expect(createProjectSettled({ data: {}, name: 'Imported' }, captureAccountScope())).rejects.toBe(failure);
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('stops rather than acting once the owning account scope expires', async () => {
    const owner = captureAccountScope();

    transport.apiFetchJson.mockImplementationOnce(() => {
      accountLifecycle.invalidate();

      return Promise.reject(new ApiError('not found', 404));
    });

    await expect(createProjectSettled(request, owner)).rejects.not.toBeInstanceOf(ProjectCreateAbsentError);
  });
});

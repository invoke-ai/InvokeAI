import type * as transportModule from '@platform/transport/http';

import { galleryKeys } from '@features/gallery/queries';
import { queryClient } from '@platform/query/client';
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, onTestFinished, vi } from 'vitest';

const transport = vi.hoisted(() => ({
  apiFetch: vi.fn(),
  apiFetchJson: vi.fn(),
}));

vi.mock('@platform/transport/http', async (importOriginal) => ({
  ...(await importOriginal<typeof transportModule>()),
  apiFetch: transport.apiFetch,
  apiFetchJson: transport.apiFetchJson,
}));

import {
  deleteProject,
  createProject,
  createProjectSettled,
  getProjectWriteSizeRefusal,
  getProject,
  isProjectCanvasSchemaUnsupportedError,
  ProjectCreateAbsentError,
  updateProject,
} from './api';

beforeEach(() => {
  transport.apiFetch.mockReset();
  transport.apiFetchJson.mockReset();
  accountLifecycle.activate('project-api-test-user');
});

/** Cleanup requires proven project absence after an ambiguous create. */
describe('createProjectSettled', () => {
  const request = { data: {}, name: 'Imported', project_id: 'project-1' };
  const post = () => ({
    body: JSON.stringify({ ...request, max_canvas_schema_version: 4, minimum_canvas_schema_version: 3 }),
    method: 'POST',
    signal: expect.anything(),
  });

  it('returns the record a create succeeded with', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project-1' });

    await expect(createProjectSettled(request, captureAccountScope())).resolves.toMatchObject({
      project_id: 'project-1',
    });
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('marks board lists fetched before the create stale, since they predate its board', async () => {
    const owner = captureAccountScope();
    const boardsKey = [...galleryKeys.boardsForAccount(owner), {}];
    queryClient.setQueryData(boardsKey, []);
    transport.apiFetchJson.mockResolvedValueOnce({ board_id: 'board-for-project-1', project_id: 'project-1' });

    onTestFinished(() => queryClient.removeQueries({ queryKey: boardsKey }));

    await createProjectSettled(request, owner);

    await vi.waitFor(() => expect(queryClient.getQueryState(boardsKey)?.isInvalidated).toBe(true));
  });

  /** Settle with a retry write: GET 404 can race the initial create transaction. */
  it('adopts the project when a retried create finds the first one committed', async () => {
    transport.apiFetchJson
      .mockRejectedValueOnce(new TypeError('network error'))
      .mockRejectedValueOnce(new ApiError('conflict', 409))
      .mockResolvedValueOnce({
        board_id: 'board-for-project-1',
        data: {},
        minimum_canvas_schema_version: 3,
        name: 'Imported',
        project_id: 'project-1',
      });

    await expect(createProjectSettled(request, captureAccountScope())).resolves.toMatchObject({
      project_id: 'project-1',
    });
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(1, '/api/v1/projects/', post());
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(2, '/api/v1/projects/', post());
    expect(transport.apiFetchJson).toHaveBeenNthCalledWith(
      3,
      '/api/v1/projects/project-1?max_canvas_schema_version=4',
      expect.anything()
    );
  });

  it('does not adopt a different project that occupies the requested id', async () => {
    const conflict = new ApiError('conflict', 409);

    transport.apiFetchJson
      .mockRejectedValueOnce(new TypeError('network error'))
      .mockRejectedValueOnce(conflict)
      .mockResolvedValueOnce({
        board_id: 'board-for-somebody-else',
        data: { owner: 'somebody-else' },
        minimum_canvas_schema_version: 3,
        name: "Somebody else's project",
        project_id: 'project-1',
      });

    await expect(createProjectSettled(request, captureAccountScope())).rejects.toBe(conflict);
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

  it('treats exhausted ingress capacity as proof that an initial create did not run', async () => {
    vi.useFakeTimers();
    const busy = new ApiError('{"detail":{"code":"project_write_busy"}}', 429, new Headers({ 'Retry-After': '0' }));
    const random = vi.spyOn(Math, 'random').mockReturnValue(0);
    transport.apiFetchJson.mockRejectedValue(busy);

    const settled = expect(createProjectSettled(request, captureAccountScope())).rejects.toBeInstanceOf(
      ProjectCreateAbsentError
    );
    await vi.runAllTimersAsync();
    await settled;

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(9);
    vi.useRealTimers();
    random.mockRestore();
  });

  it('does not treat exhausted settlement capacity as proof that a prior create did not commit', async () => {
    vi.useFakeTimers();
    const unknown = new TypeError('response lost');
    const busy = new ApiError('{"detail":{"code":"project_write_busy"}}', 429, new Headers({ 'Retry-After': '0' }));
    const random = vi.spyOn(Math, 'random').mockReturnValue(0);
    let serializedValueReads = 0;
    const settlementRequest = {
      data: {
        get value() {
          serializedValueReads += 1;
          return 'value';
        },
      },
      name: 'Imported',
      project_id: 'project-1',
    };
    transport.apiFetchJson.mockRejectedValueOnce(unknown).mockRejectedValue(busy);

    const settled = expect(createProjectSettled(settlementRequest, captureAccountScope())).rejects.toBe(unknown);
    await vi.runAllTimersAsync();
    await settled;

    expect(transport.apiFetchJson).toHaveBeenCalledTimes(9);
    expect(serializedValueReads).toBe(1);
    expect(new Set(transport.apiFetchJson.mock.calls.map((call) => (call[1] as RequestInit).body)).size).toBe(1);
    vi.useRealTimers();
    random.mockRestore();
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

describe('canvas schema compatibility declarations', () => {
  it('declares the supported schema on every document read', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project/one' });

    await getProject('project/one');

    expect(transport.apiFetchJson).toHaveBeenCalledWith('/api/v1/projects/project%2Fone?max_canvas_schema_version=4', {
      signal: undefined,
    });
  });

  it('declares both the written floor and supported maximum when creating', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project-1' });

    await createProject({ data: {}, name: 'Project' });

    expect(transport.apiFetchJson).toHaveBeenCalledWith('/api/v1/projects/', {
      body: JSON.stringify({
        data: {},
        name: 'Project',
        max_canvas_schema_version: 4,
        minimum_canvas_schema_version: 3,
      }),
      method: 'POST',
      signal: undefined,
    });
  });

  it('sends a requested floor in the same update as the document', async () => {
    transport.apiFetchJson.mockResolvedValueOnce({ project_id: 'project-1' });

    await updateProject('project-1', {
      data: { canvas: { version: 3 } },
      expected_revision: 4,
      minimum_canvas_schema_version: 3,
      name: 'Project',
    });

    expect(transport.apiFetchJson).toHaveBeenCalledWith('/api/v1/projects/project-1', {
      body: JSON.stringify({
        data: { canvas: { version: 3 } },
        expected_revision: 4,
        minimum_canvas_schema_version: 3,
        name: 'Project',
        max_canvas_schema_version: 4,
      }),
      method: 'PUT',
      signal: undefined,
    });
  });

  it('retries a project write refused by the ingress concurrency guard', async () => {
    let serializedValueReads = 0;
    const data = {
      get value() {
        serializedValueReads += 1;
        return 'value';
      },
    };
    const random = vi.spyOn(Math, 'random').mockReturnValue(0);
    transport.apiFetchJson
      .mockRejectedValueOnce(
        new ApiError('{"detail":{"code":"project_write_busy"}}', 429, new Headers({ 'Retry-After': '0' }))
      )
      .mockResolvedValueOnce({ project_id: 'project-1' });

    await expect(updateProject('project-1', { data, expected_revision: 1, name: 'Project' })).resolves.toMatchObject({
      project_id: 'project-1',
    });
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(2);
    expect(serializedValueReads).toBe(1);
    random.mockRestore();
  });

  it('does not retry unrelated 429 responses', async () => {
    const refusal = new ApiError('{"detail":{"code":"other"}}', 429, new Headers({ 'Retry-After': '0' }));
    transport.apiFetchJson.mockRejectedValueOnce(refusal);

    await expect(updateProject('project-1', { data: {}, expected_revision: 1, name: 'Project' })).rejects.toBe(refusal);
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('reserves the final attempt until the server can release its longest-held slot', async () => {
    vi.useFakeTimers();
    const busy = new ApiError('{"detail":{"code":"project_write_busy"}}', 429, new Headers({ 'Retry-After': '1' }));
    const random = vi.spyOn(Math, 'random').mockReturnValue(0);
    transport.apiFetchJson.mockRejectedValueOnce(busy).mockRejectedValueOnce(busy).mockRejectedValueOnce(busy);
    transport.apiFetchJson.mockRejectedValueOnce(busy).mockRejectedValueOnce(busy).mockRejectedValueOnce(busy);
    transport.apiFetchJson.mockRejectedValueOnce(busy).mockRejectedValueOnce(busy).mockResolvedValueOnce({
      project_id: 'project-1',
    });

    const write = updateProject('project-1', { data: {}, expected_revision: 1, name: 'Project' });
    await vi.advanceTimersByTimeAsync(123_999);
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(8);
    await vi.advanceTimersByTimeAsync(1);
    await expect(write).resolves.toMatchObject({ project_id: 'project-1' });
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(9);
    vi.useRealTimers();
    random.mockRestore();
  });

  it('stops retrying when the owning account scope expires during backoff', async () => {
    const owner = captureAccountScope();
    const busy = new ApiError('{"detail":{"code":"project_write_busy"}}', 429, new Headers({ 'Retry-After': '60' }));
    transport.apiFetchJson.mockRejectedValue(busy);

    const write = updateProject('project-1', { data: {}, expected_revision: 1, name: 'Project' }, owner.signal);
    await vi.waitFor(() => expect(transport.apiFetchJson).toHaveBeenCalledTimes(1));
    accountLifecycle.invalidate();

    await expect(write).rejects.toBe(owner.signal.reason);
    expect(transport.apiFetchJson).toHaveBeenCalledTimes(1);
  });

  it('recognizes server-side schema refusals without conflating them with revision conflicts', () => {
    const refusal = new ApiError(
      JSON.stringify({
        detail: {
          code: 'canvas_schema_unsupported',
          max_canvas_schema_version: 3,
          minimum_canvas_schema_version: 4,
        },
      }),
      412
    );

    expect(isProjectCanvasSchemaUnsupportedError(refusal)).toBe(true);
    expect(isProjectCanvasSchemaUnsupportedError(new ApiError('conflict', 409))).toBe(false);
    expect(
      isProjectCanvasSchemaUnsupportedError(
        new ApiError(
          JSON.stringify({
            detail: { code: 'different_precondition', max_canvas_schema_version: 3, minimum_canvas_schema_version: 4 },
          }),
          412
        )
      )
    ).toBe(false);
    expect(
      isProjectCanvasSchemaUnsupportedError(
        new ApiError(
          JSON.stringify({
            detail: {
              code: 'canvas_schema_unsupported',
              max_canvas_schema_version: 3,
              minimum_canvas_schema_version: 3.5,
            },
          }),
          412
        )
      )
    ).toBe(false);
  });
});

describe('project write size refusals', () => {
  it('parses document and request limits as distinct refusals', () => {
    const refusal = new ApiError(
      JSON.stringify({
        detail: {
          actual_bytes: 33_554_433,
          code: 'project_document_too_large',
          max_bytes: 33_554_432,
        },
      }),
      413
    );

    expect(getProjectWriteSizeRefusal(refusal)).toEqual({
      actualBytes: 33_554_433,
      kind: 'document',
      maxBytes: 33_554_432,
    });
    expect(
      getProjectWriteSizeRefusal(
        new ApiError(
          JSON.stringify({
            detail: { actual_bytes: 35_651_585, code: 'project_request_too_large', max_bytes: 35_651_584 },
          }),
          413
        )
      )
    ).toEqual({ actualBytes: 35_651_585, kind: 'request', maxBytes: 35_651_584 });
  });

  it.each([
    ['plain message', new ApiError('too large', 413)],
    ['wrong status', new ApiError('{"detail":{"code":"project_document_too_large"}}', 409)],
    ['wrong code', new ApiError('{"detail":{"actual_bytes":17,"code":"other","max_bytes":16}}', 413)],
    [
      'equal bounds',
      new ApiError('{"detail":{"actual_bytes":16,"code":"project_document_too_large","max_bytes":16}}', 413),
    ],
    [
      'fractional bound',
      new ApiError('{"detail":{"actual_bytes":17.5,"code":"project_document_too_large","max_bytes":16}}', 413),
    ],
    [
      'negative bound',
      new ApiError('{"detail":{"actual_bytes":17,"code":"project_document_too_large","max_bytes":-1}}', 413),
    ],
    [
      'unsafe bound',
      new ApiError(
        JSON.stringify({
          detail: { actual_bytes: Number.MAX_SAFE_INTEGER + 1, code: 'project_document_too_large', max_bytes: 16 },
        }),
        413
      ),
    ],
  ])('rejects %s', (_name, error) => {
    expect(getProjectWriteSizeRefusal(error)).toBeNull();
  });
});

describe('deleteProject', () => {
  it('asks the server to release the boards unless told to delete them', async () => {
    transport.apiFetch.mockResolvedValue(undefined);

    await deleteProject('project-1');
    await deleteProject('project-1', undefined, 'release');
    await deleteProject('project-1', undefined, 'delete');

    expect(transport.apiFetch.mock.calls.map(([url]) => url)).toEqual([
      '/api/v1/projects/project-1',
      '/api/v1/projects/project-1',
      '/api/v1/projects/project-1?boards=delete',
    ]);
    expect(transport.apiFetch.mock.calls.every(([, init]) => (init as { method: string }).method === 'DELETE')).toBe(
      true
    );
  });
});

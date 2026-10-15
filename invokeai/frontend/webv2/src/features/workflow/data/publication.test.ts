import type { AccountScope } from '@platform/state/accountLifecycle';

import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { WorkflowRecordDTO } from './api';

import { WorkflowLibraryWriteRefusedError } from './api';
import { createWorkflowPublicationController, isWorkflowContentEquivalent } from './publication';

const record = (overrides: Partial<WorkflowRecordDTO> = {}): WorkflowRecordDTO => ({
  category: 'user',
  description: '',
  name: 'Template',
  revision: 1,
  workflow: { name: 'Template', nodes: [] },
  workflow_id: 'lib-1',
  ...overrides,
});

const workflow = { name: 'Template', nodes: [{ id: 'n1' }] };

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, reject, resolve };
};

describe('workflow publication controller', () => {
  let owner: AccountScope;
  const deps = {
    createRecord: vi.fn(),
    getRecord: vi.fn(),
    invalidateCache: vi.fn(),
    reserveId: vi.fn(() => 'reserved-1'),
    updateRecord: vi.fn(),
  };

  beforeEach(() => {
    accountLifecycle.activate('user-a');
    owner = captureAccountScope();
    deps.createRecord.mockReset();
    deps.getRecord.mockReset();
    deps.invalidateCache.mockReset();
    deps.reserveId.mockReset().mockReturnValue('reserved-1');
    deps.updateRecord.mockReset();
  });

  const request = (destination: Parameters<typeof controller.publish>[0]['destination']) => ({
    destination,
    owner,
    projectId: 'project-1',
    workflow,
    workflowId: 'wf-1',
  });
  const controller = createWorkflowPublicationController(deps);

  it('creates under a reserved id, names the template, and invalidates the library', async () => {
    deps.createRecord.mockResolvedValue(record({ name: 'Named', revision: 1, workflow_id: 'reserved-1' }));

    const result = await controller.publish(request({ kind: 'create', name: 'Named' }));

    expect(result).toEqual({
      kind: 'created',
      libraryWorkflowId: 'reserved-1',
      name: 'Named',
      revision: 1,
      status: 'published',
    });
    expect(deps.createRecord).toHaveBeenCalledWith(
      { ...workflow, name: 'Named' },
      { reservedId: 'reserved-1', signal: owner.signal }
    );
    expect(deps.invalidateCache).toHaveBeenCalledWith('reserved-1');
  });

  it('refuses to overlap a publication of the same workflow and reports when it is busy', async () => {
    const first = deferred<WorkflowRecordDTO>();
    deps.createRecord.mockReturnValueOnce(first.promise);

    const inFlight = controller.publish(request({ kind: 'create', name: 'A' }));

    expect(controller.isPublishing('project-1', 'wf-1')).toBe(true);
    await expect(controller.publish(request({ kind: 'create', name: 'B' }))).resolves.toEqual({ status: 'busy' });
    expect(deps.createRecord).toHaveBeenCalledTimes(1);

    first.resolve(record({ workflow_id: 'reserved-1' }));
    await inFlight;
    expect(controller.isPublishing('project-1', 'wf-1')).toBe(false);
  });

  it('keeps the reserved id and captured content across a retry of a lost creation', async () => {
    deps.createRecord.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    deps.createRecord.mockResolvedValueOnce(record({ name: 'Named', workflow_id: 'reserved-1' }));

    const failed = await controller.publish(request({ kind: 'create', name: 'Named' }));

    expect(failed.status).toBe('failed');
    if (failed.status !== 'failed') {
      throw new Error('expected a failure');
    }

    await expect(failed.retry()).resolves.toMatchObject({ libraryWorkflowId: 'reserved-1', status: 'published' });
    expect(deps.createRecord.mock.calls.map((call) => call[1].reservedId)).toEqual(['reserved-1', 'reserved-1']);
    expect(deps.reserveId).toHaveBeenCalledTimes(1);
  });

  it("updates at the expected revision under the template's own name and reports the server revision back", async () => {
    deps.getRecord.mockResolvedValue(record({ name: 'Library name', revision: 2 }));
    deps.updateRecord.mockResolvedValue(record({ name: 'Library name', revision: 3 }));

    const result = await controller.publish(
      request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' })
    );

    expect(result).toEqual({
      kind: 'updated',
      libraryWorkflowId: 'lib-1',
      name: 'Library name',
      revision: 3,
      status: 'published',
    });
    expect(deps.updateRecord).toHaveBeenCalledWith(
      'lib-1',
      { ...workflow, name: 'Library name' },
      { expectedRevision: 2, signal: owner.signal }
    );
    expect(deps.invalidateCache).toHaveBeenCalledWith('lib-1');
  });

  it('reports a conflict from the record read alone, without sending', async () => {
    deps.getRecord.mockResolvedValue(record({ revision: 5 }));

    await expect(
      controller.publish(request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' }))
    ).resolves.toEqual({ currentRevision: 5, libraryWorkflowId: 'lib-1', status: 'conflict' });
    expect(deps.updateRecord).not.toHaveBeenCalled();
  });

  it('maps server refusals to conflict, unavailable, and invalid outcomes', async () => {
    deps.getRecord.mockResolvedValue(record({ revision: 2 }));
    deps.updateRecord.mockRejectedValueOnce(new WorkflowLibraryWriteRefusedError('revision-conflict', 'stale', 7));
    await expect(
      controller.publish(request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' }))
    ).resolves.toEqual({
      currentRevision: 7,
      libraryWorkflowId: 'lib-1',
      status: 'conflict',
    });

    deps.getRecord.mockRejectedValueOnce(new WorkflowLibraryWriteRefusedError('bundled', 'read-only'));
    await expect(
      controller.publish(request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' }))
    ).resolves.toEqual({
      libraryWorkflowId: 'lib-1',
      reason: 'bundled',
      status: 'unavailable',
    });

    deps.getRecord.mockRejectedValueOnce(new ApiError(JSON.stringify({ detail: 'nope' }), 404));
    await expect(
      controller.publish(request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' }))
    ).resolves.toEqual({
      libraryWorkflowId: 'lib-1',
      reason: 'missing',
      status: 'unavailable',
    });

    deps.createRecord.mockRejectedValueOnce(new WorkflowLibraryWriteRefusedError('invalid', 'bad graph'));
    await expect(controller.publish(request({ kind: 'create', name: 'X' }))).resolves.toEqual({
      message: 'bad graph',
      status: 'invalid',
    });
  });

  it('reconciles a lost update against the record before resending', async () => {
    deps.getRecord.mockResolvedValueOnce(record({ revision: 2 }));
    deps.updateRecord.mockRejectedValueOnce(new TypeError('Failed to fetch'));

    const failed = await controller.publish(
      request({ expectedRevision: 2, kind: 'update', libraryWorkflowId: 'lib-1' })
    );
    if (failed.status !== 'failed') {
      throw new Error('expected a failure');
    }

    // The write landed: the record moved exactly one revision on and holds this content.
    deps.getRecord.mockResolvedValueOnce(record({ revision: 3, workflow }));
    await expect(failed.retry()).resolves.toMatchObject({ kind: 'updated', revision: 3, status: 'published' });
    expect(deps.updateRecord).toHaveBeenCalledTimes(1);

    // The write never landed: resend at the same revision.
    deps.getRecord.mockResolvedValueOnce(record({ revision: 2 }));
    deps.updateRecord.mockResolvedValueOnce(record({ revision: 3 }));
    await expect(failed.retry()).resolves.toMatchObject({ kind: 'updated', status: 'published' });
    expect(deps.updateRecord).toHaveBeenCalledTimes(2);

    // Someone else moved it: never overwrite.
    deps.getRecord.mockResolvedValueOnce(record({ revision: 3, workflow: { name: 'Theirs', nodes: [] } }));
    await expect(failed.retry()).resolves.toEqual({
      currentRevision: 3,
      libraryWorkflowId: 'lib-1',
      status: 'conflict',
    });
    expect(deps.updateRecord).toHaveBeenCalledTimes(2);
  });

  it('reports cancellation instead of a result once the account changes', async () => {
    const pending = deferred<WorkflowRecordDTO>();
    deps.createRecord.mockReturnValueOnce(pending.promise);

    const inFlight = controller.publish(request({ kind: 'create', name: 'A' }));

    accountLifecycle.invalidate();
    accountLifecycle.activate('user-b');
    pending.resolve(record());

    await expect(inFlight).resolves.toEqual({ status: 'cancelled' });
    expect(deps.invalidateCache).not.toHaveBeenCalled();
  });
});

describe('isWorkflowContentEquivalent', () => {
  it('ignores key order and fields the server does not keep', () => {
    expect(
      isWorkflowContentEquivalent(
        { id: 'x', name: 'A', nodes: [{ b: 1, a: 2 }], meta: { version: '3.0.0', category: 'user' } },
        { name: 'A', nodes: [{ a: 2, b: 1 }], meta: { category: 'user', version: '3.0.0' }, opened_at: 'now' }
      )
    ).toBe(true);
    expect(isWorkflowContentEquivalent({ name: 'A', nodes: [] }, { name: 'B', nodes: [] })).toBe(false);
  });
});

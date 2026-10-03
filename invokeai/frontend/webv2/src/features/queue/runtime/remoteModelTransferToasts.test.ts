import { describe, expect, it, vi } from 'vitest';

vi.mock('@platform/ui', () => ({
  toaster: { create: vi.fn(), update: vi.fn(), dismiss: vi.fn() },
}));

import { toaster } from '@platform/ui';

import { createRemoteModelTransferToasts, parseRemoteModelTransferProgress } from './remoteModelTransferToasts';

const envelope = (overrides: Record<string, unknown> = {}) =>
  `[[IRW_MODEL_TRANSFER]]${JSON.stringify({
    backend_item_id: 10,
    remote_index: 2,
    model_hash: 'hash',
    name: 'Example',
    directory: true,
    phase: 'downloading',
    bytes: 40,
    total_bytes: 100,
    ...overrides,
  })}`;

describe('remote model transfer notifications', () => {
  it('validates the dedicated progress envelope', () => {
    expect(parseRemoteModelTransferProgress('regular progress')).toBeNull();
    expect(parseRemoteModelTransferProgress('[[IRW_MODEL_TRANSFER]]{')).toBeNull();
    expect(parseRemoteModelTransferProgress(envelope({ remote_index: -1 }))).toBeNull();
    expect(parseRemoteModelTransferProgress(envelope({ bytes: -1 }))).toBeNull();
    expect(parseRemoteModelTransferProgress(envelope({ phase: 'unexpected' }))).toBeNull();
    expect(parseRemoteModelTransferProgress(envelope())?.phase).toBe('downloading');
  });

  it('bounds remembered terminal transfer ids while keeping active state separate', () => {
    vi.clearAllMocks();
    const sink = createRemoteModelTransferToasts();

    for (let index = 0; index < 257; index += 1) {
      sink.receive({
        backendItemId: 1000 + index,
        remoteIndex: 1,
        modelHash: `hash-${index}`,
        name: `Model ${index}`,
        directory: true,
        phase: 'completed',
        bytes: 1,
        totalBytes: 1,
      });
    }

    expect(toaster.create).toHaveBeenCalledTimes(257);

    sink.receive({
      backendItemId: 1000,
      remoteIndex: 1,
      modelHash: 'hash-0',
      name: 'Model 0',
      directory: true,
      phase: 'downloading',
      bytes: 0,
      totalBytes: 1,
    });
    expect(toaster.create).toHaveBeenCalledTimes(258);

    sink.receive({
      backendItemId: 1256,
      remoteIndex: 1,
      modelHash: 'hash-256',
      name: 'Model 256',
      directory: true,
      phase: 'downloading',
      bytes: 0,
      totalBytes: 1,
    });
    expect(toaster.create).toHaveBeenCalledTimes(258);
  });

  it('updates one loading toast, settles it and does not resurrect it', () => {
    vi.clearAllMocks();
    const sink = createRemoteModelTransferToasts();
    const progress = parseRemoteModelTransferProgress(envelope());
    const completed = parseRemoteModelTransferProgress(envelope({ phase: 'completed', bytes: 100 }));
    expect(progress).not.toBeNull();
    expect(completed).not.toBeNull();
    if (!progress || !completed) {
      return;
    }
    sink.receive(progress);
    sink.receive({ ...progress, bytes: 60 });
    sink.receive(completed);
    sink.receive(progress);
    expect(toaster.create).toHaveBeenCalledTimes(1);
    expect(toaster.update).toHaveBeenCalledTimes(2);
    expect(toaster.update).toHaveBeenLastCalledWith(expect.any(String), expect.objectContaining({ type: 'success' }));
    sink.dispose();
    expect(toaster.dismiss).toHaveBeenCalledTimes(1);
  });
});

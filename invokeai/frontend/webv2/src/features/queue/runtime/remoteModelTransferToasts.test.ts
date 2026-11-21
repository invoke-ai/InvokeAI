import { accountLifecycle } from '@platform/state/accountLifecycle';
import { describe, expect, it, vi } from 'vitest';

vi.mock('@platform/ui/toaster', () => ({
  toaster: { create: vi.fn(), update: vi.fn(), dismiss: vi.fn() },
}));

import { toaster } from '@platform/ui/toaster';

import {
  createRemoteModelTransferToasts,
  parseRemoteModelTransferProgress,
  subscribeRemoteModelTransferToasts,
} from './remoteModelTransferToasts';

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
    const sink = createRemoteModelTransferToasts(toaster);

    const transfer = (index: number) => ({
      backendItemId: 1000 + index,
      remoteIndex: 1,
      modelHash: `hash-${index}`,
      name: `Model ${index}`,
      directory: true,
      totalBytes: 1,
    });

    for (let index = 0; index < 257; index += 1) {
      sink.receive({ ...transfer(index), phase: 'completed', bytes: 1 });
    }

    expect(toaster.create).toHaveBeenCalledTimes(257);

    sink.receive({ ...transfer(0), phase: 'downloading', bytes: 0 });
    expect(toaster.create).toHaveBeenCalledTimes(258);

    sink.receive({ ...transfer(256), phase: 'downloading', bytes: 0 });
    expect(toaster.create).toHaveBeenCalledTimes(258);
  });

  it('updates one loading toast, settles it and does not resurrect it', () => {
    vi.clearAllMocks();
    const sink = createRemoteModelTransferToasts(toaster);
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

describe('remote model transfer socket subscription', () => {
  it.each(['user-1', 'single-user'])('scopes events and disposes the listener on account change (%s)', (accountId) => {
    vi.clearAllMocks();
    accountLifecycle.activate(accountId);
    let handler: ((payload: never) => void) | undefined;
    const detach = vi.fn();
    const socket = {
      on: vi.fn((_event: string, listener: (payload: never) => void) => {
        handler = listener;
        return detach;
      }),
    };
    const dispose = subscribeRemoteModelTransferToasts(socket, toaster);
    const emit = (message: string, userId: string) => handler?.({ message, user_id: userId } as never);
    try {
      expect(socket.on).toHaveBeenCalledWith('invocation_progress', expect.any(Function));
      emit('ordinary generation progress', accountId);
      emit('[[IRW_MODEL_TRANSFER]]{', accountId);
      if (accountId !== 'single-user') {
        emit(envelope(), 'user-2');
      }
      expect(toaster.create).not.toHaveBeenCalled();

      emit(envelope(), accountId === 'single-user' ? 'backend-user' : accountId);
      expect(toaster.create).toHaveBeenCalledTimes(1);
      accountLifecycle.activate(accountId);
      expect(detach).toHaveBeenCalledTimes(1);
      expect(toaster.dismiss).toHaveBeenCalledTimes(1);
      emit(envelope({ backend_item_id: 11 }), accountId);
      expect(toaster.create).toHaveBeenCalledTimes(1);
    } finally {
      dispose();
      accountLifecycle.invalidate();
    }
    expect(detach).toHaveBeenCalledTimes(1);
  });
});

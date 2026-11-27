import type { ConnectionListener } from '@platform/transport/socketHub';
import type { BackendConnectionStatus } from '@platform/transport/types';

import { QueryClient } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';

const hub = vi.hoisted(() => ({
  connection: null as ConnectionListener | null,
  operationChanged: null as ((payload: unknown) => void) | null,
  status: 'connected' as BackendConnectionStatus,
}));

vi.mock('@platform/transport/socketHub', () => ({
  socketHub: {
    on: (event: string, listener: (payload: unknown) => void) => {
      if (event === 'intermediates_operation_changed') {
        hub.operationChanged = listener;
      }
      return () => {
        hub.operationChanged = null;
      };
    },
    onConnectionChange: (listener: ConnectionListener) => {
      hub.connection = listener;
      listener(hub.status);
      return () => {
        hub.connection = null;
      };
    },
  },
}));

const { attachIntermediatesRealtime } = await import('./realtime');
const { intermediatesKeys } = await import('./keys');
const { captureAccountScope } = await import('@platform/state/accountLifecycle');

const operationEvent = (status: string, operationId = 'op-1') => ({
  operation: {
    completed_at: status === 'completed' ? 'later' : null,
    created_at: 'now',
    error: null,
    mode: 'safe',
    operation_id: operationId,
    progress: {
      deleted_images: 1,
      deleted_videos: 0,
      failed_images: 0,
      failed_videos: 0,
      pending_disk_cleanup: 0,
      processed_images: 1,
      processed_videos: 0,
      reclaimed_bytes: 10,
      retained_images: 0,
      retained_videos: 0,
      unknown_size_count: 0,
    },
    scope: { kind: 'owner', targets: [], user_id: 'alice' },
    started_at: 'now',
    status,
    target_images: 2,
    target_videos: 0,
    user_id: 'alice',
  },
});

afterEach(() => {
  hub.status = 'connected';
});

it('does not refetch on open when the socket is already connected, but does after a reconnect', () => {
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const detach = attachIntermediatesRealtime(queryClient);

  expect(invalidate).not.toHaveBeenCalled();
  hub.connection!('disconnected');
  hub.connection!('connected');
  expect(invalidate).toHaveBeenCalledOnce();
  detach();
});

it('refetches once the socket connects after opening while disconnected', () => {
  hub.status = 'disconnected';
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const detach = attachIntermediatesRealtime(queryClient);

  hub.connection!('connected');
  expect(invalidate).toHaveBeenCalledOnce();
  detach();
});

it('stores socket operations and invalidates the account summaries once when one settles', () => {
  const queryClient = new QueryClient();
  const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
  const detach = attachIntermediatesRealtime(queryClient);
  const key = intermediatesKeys.operation(captureAccountScope(), 'op-1');

  hub.operationChanged!(operationEvent('running'));
  expect(queryClient.getQueryData(key)).toMatchObject({ operationId: 'op-1', status: 'running' });
  expect(invalidate).not.toHaveBeenCalled();

  hub.operationChanged!(operationEvent('completed'));
  hub.operationChanged!(operationEvent('completed'));
  expect(queryClient.getQueryData(key)).toMatchObject({ status: 'completed' });
  expect(invalidate).toHaveBeenCalledOnce();
  expect(invalidate).toHaveBeenCalledWith({ queryKey: intermediatesKeys.summaries(captureAccountScope()) });
  detach();
});

it('ignores malformed socket events without throwing', () => {
  const queryClient = new QueryClient();
  const setQueryData = vi.spyOn(queryClient, 'setQueryData');
  const detach = attachIntermediatesRealtime(queryClient);
  const { progress: _progress, ...withoutProgress } = operationEvent('completed').operation;

  for (const payload of [undefined, null, 'x', {}, { operation: null }, { operation: withoutProgress }]) {
    expect(() => hub.operationChanged!(payload)).not.toThrow();
  }
  expect(setQueryData).not.toHaveBeenCalled();
  detach();
});

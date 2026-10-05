import type { SocketHub } from '@platform/transport/socketHub';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { afterEach, expect, it } from 'vitest';

import { createLLMTaskProgressRuntime, llmTaskProgressStore, type LLMTaskProgressEvent } from './llmTaskProgress';

type SocketHandler = (payload: never) => void;

const createFakeSocketHub = () => {
  const handlers = new Map<string, Set<SocketHandler>>();
  const hub: Pick<SocketHub, 'on'> = {
    on: (event, handler) => {
      const eventHandlers = handlers.get(event) ?? new Set<SocketHandler>();
      eventHandlers.add(handler);
      handlers.set(event, eventHandlers);
      return () => eventHandlers.delete(handler);
    },
  };

  return {
    emit: (event: string, payload: unknown) => {
      for (const handler of handlers.get(event) ?? []) {
        handler(payload as never);
      }
    },
    handlerCount: () => [...handlers.values()].reduce((count, eventHandlers) => count + eventHandlers.size, 0),
    hub,
  };
};

const makeProgress = (taskId: string, currentTokens: number): LLMTaskProgressEvent => ({
  current_tokens: currentTokens,
  message: 'Generating',
  percentage: currentTokens / 20,
  phase: 'generating',
  task_id: taskId,
  timestamp: 1,
  total_tokens: 20,
  user_id: 'owner-a',
});

afterEach(() => {
  llmTaskProgressStore.clear();
  accountLifecycle.invalidate();
});

it('tracks concurrent LLM tasks independently and clears terminal tasks', () => {
  accountLifecycle.activate('owner-a');
  const socket = createFakeSocketHub();
  const runtime = createLLMTaskProgressRuntime(socket.hub);

  socket.emit('llm_task_progress', makeProgress('task-a', 3));
  socket.emit('llm_task_progress', makeProgress('task-b', 7));

  expect(llmTaskProgressStore.get('task-a')?.current_tokens).toBe(3);
  expect(llmTaskProgressStore.get('task-b')?.current_tokens).toBe(7);

  socket.emit('llm_task_complete', { task_id: 'task-a', timestamp: 2, user_id: 'owner-a' });
  expect(llmTaskProgressStore.get('task-a')).toBeUndefined();
  expect(llmTaskProgressStore.get('task-b')?.current_tokens).toBe(7);

  socket.emit('llm_task_error', { error: 'failed', task_id: 'task-b', timestamp: 3, user_id: 'owner-a' });
  expect(llmTaskProgressStore.get('task-b')).toBeUndefined();

  runtime.dispose();
});

it('detaches every socket listener and ignores events from an expired account scope', () => {
  accountLifecycle.activate('owner-a');
  const socket = createFakeSocketHub();
  const runtime = createLLMTaskProgressRuntime(socket.hub);

  expect(socket.handlerCount()).toBe(3);
  accountLifecycle.activate('owner-b');
  socket.emit('llm_task_progress', makeProgress('stale-task', 5));
  expect(llmTaskProgressStore.get('stale-task')).toBeUndefined();

  runtime.dispose();
  runtime.dispose();
  expect(socket.handlerCount()).toBe(0);
});

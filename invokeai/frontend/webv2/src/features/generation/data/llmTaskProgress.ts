import type { SocketHub } from '@platform/transport/socketHub';

import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createKeyedTransientStore } from '@platform/state/externalStore';

interface LLMTaskEventBase {
  task_id: string;
  timestamp: number;
  user_id: string;
}

export interface LLMTaskProgressEvent extends LLMTaskEventBase {
  current_tokens: number | null;
  message: string;
  percentage: number | null;
  phase: 'loading_model' | 'generating';
  total_tokens: number | null;
}

export interface LLMTaskCompleteEvent extends LLMTaskEventBase {}

export interface LLMTaskErrorEvent extends LLMTaskEventBase {
  error: string;
}

export const llmTaskProgressStore = createKeyedTransientStore<string, LLMTaskProgressEvent>();

registerAccountOwnedResource({
  clear: llmTaskProgressStore.clear,
  name: 'llm-task-progress',
});

const hasTaskId = (payload: unknown): payload is { task_id: string } =>
  Boolean(payload) && typeof payload === 'object' && typeof (payload as { task_id?: unknown }).task_id === 'string';

export interface LLMTaskProgressRuntime {
  dispose(): void;
}

export const createLLMTaskProgressRuntime = (hub: Pick<SocketHub, 'on'>): LLMTaskProgressRuntime => {
  const owner = captureAccountScope();
  const isCurrent = () => isAccountScopeCurrent(owner);
  const detachers = [
    hub.on('llm_task_progress', (payload: unknown) => {
      if (isCurrent() && hasTaskId(payload)) {
        llmTaskProgressStore.set(payload.task_id, payload as LLMTaskProgressEvent);
      }
    }),
    hub.on('llm_task_complete', (payload: unknown) => {
      if (isCurrent() && hasTaskId(payload)) {
        llmTaskProgressStore.delete(payload.task_id);
      }
    }),
    hub.on('llm_task_error', (payload: unknown) => {
      if (isCurrent() && hasTaskId(payload)) {
        llmTaskProgressStore.delete(payload.task_id);
      }
    }),
  ];
  let disposed = false;

  return {
    dispose: () => {
      if (disposed) {
        return;
      }
      disposed = true;
      for (const detach of detachers) {
        detach();
      }
    },
  };
};

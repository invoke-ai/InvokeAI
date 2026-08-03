import { ChakraProvider } from '@chakra-ui/react';
import { llmTaskProgressStore } from '@features/generation/data/llmTaskProgress';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it } from 'vitest';

import { LLMTaskProgressDisplay } from './LLMTaskProgressDisplay';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          generate: {
            llmTaskGenerating: 'Generating',
            llmTaskLoadingModel: 'Loading model',
            llmTaskProgress: 'LLM task progress',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (taskId: string | null) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <LLMTaskProgressDisplay taskId={taskId} />
        </I18nextProvider>
      </ChakraProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  llmTaskProgressStore.clear();
  host?.remove();
  host = null;
  root = null;
});

describe('LLMTaskProgressDisplay', () => {
  it('shows model loading as indeterminate progress', async () => {
    llmTaskProgressStore.set('loading-task', {
      current_tokens: null,
      message: 'Loading model',
      percentage: null,
      phase: 'loading_model',
      task_id: 'loading-task',
      timestamp: 1,
      total_tokens: null,
      user_id: 'owner-a',
    });

    await render('loading-task');

    const progress = host?.querySelector('[role="progressbar"]');
    expect(progress?.getAttribute('aria-label')).toBe('LLM task progress');
    expect(progress?.hasAttribute('aria-valuenow')).toBe(false);
    expect(host?.textContent).toContain('Loading model');
  });

  it('updates token counts for the task without rendering another task', async () => {
    llmTaskProgressStore.set('task-a', {
      current_tokens: 5,
      message: 'Generating',
      percentage: 0.25,
      phase: 'generating',
      task_id: 'task-a',
      timestamp: 1,
      total_tokens: 20,
      user_id: 'owner-a',
    });
    llmTaskProgressStore.set('task-b', {
      current_tokens: 99,
      message: 'Generating',
      percentage: 0.99,
      phase: 'generating',
      task_id: 'task-b',
      timestamp: 1,
      total_tokens: 100,
      user_id: 'owner-a',
    });

    await render('task-a');

    const progress = host?.querySelector('[role="progressbar"]');
    expect(progress?.getAttribute('aria-valuenow')).toBe('0.25');
    expect(host?.textContent).toContain('5 / 20');
    expect(host?.textContent).not.toContain('99 / 100');
  });
});

/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-function-as-prop */
import type { ModelConfig, ModelIdentificationOutcome } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { IdentifyUnknownModelsDialog } from './IdentifyUnknownModelsDialog';

const api = vi.hoisted(() => ({
  bulkGetModelIdentification: vi.fn<(keys: string[]) => Promise<ModelIdentificationOutcome[]>>(),
  bulkReidentifyModels: vi.fn<(keys: string[]) => Promise<{ failed: never[]; succeeded: string[] }>>(),
  listModels: vi.fn(() => Promise.resolve([] as ModelConfig[])),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  bulkGetModelIdentification: api.bulkGetModelIdentification,
  bulkReidentifyModels: api.bulkReidentifyModels,
  listModels: api.listModels,
}));
vi.mock('@features/models/ui/useModelsNotify', () => ({
  useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }),
}));
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: Record<string, string | number>) =>
      options ? `${key}(${Object.values(options).join(',')})` : key,
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const unknown = (key: string, name: string) =>
  ({
    base: 'unknown',
    file_size: 1,
    format: 'unknown',
    hash: key,
    key,
    name,
    path: `${key}.safetensors`,
    source: key,
    source_type: 'path',
    type: 'unknown',
  }) as unknown as ModelConfig;

const A = unknown('a', 'Model A');
const B = unknown('b', 'Model B');
const C = unknown('c', 'Model C');
const D = unknown('d', 'Model D');
const proposal = (model: ModelConfig): ModelIdentificationOutcome => ({
  identification: {
    changed: true,
    proposed: { ...model, base: 'sdxl', format: 'checkpoint', type: 'main' },
    refusal: null,
  },
  key: model.key,
  status: 'identified',
});

describe('IdentifyUnknownModelsDialog', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(async () => {
    api.bulkGetModelIdentification.mockReset();
    api.bulkReidentifyModels.mockReset();
    accountLifecycle.activate('identify-unknown', ':user:identify-unknown');
    setModelsSnapshotForTests({
      models: [A, B, C, D, { ...A, key: 'known', name: 'Known', type: 'main' } as ModelConfig],
      status: 'loaded',
    });
    api.bulkGetModelIdentification.mockResolvedValue([
      proposal(A),
      proposal(B),
      {
        identification: { changed: false, proposed: null, refusal: 'missing projection' },
        key: C.key,
        status: 'identified',
      },
      { error: 'file is gone', key: D.key, status: 'failed' },
    ]);
    api.bulkReidentifyModels.mockResolvedValue({ failed: [], succeeded: ['a'] });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <IdentifyUnknownModelsDialog isOpen onClose={() => undefined} onExitComplete={() => undefined} />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setModelsSnapshotForTests({ models: [], status: 'idle' });
    accountLifecycle.invalidate();
  });

  it('previews only the Unknown models, lists unresolved ones apart, and applies just the ticked ones', async () => {
    await expect.poll(() => document.body.textContent).toContain('Model A');

    expect(api.bulkGetModelIdentification).toHaveBeenCalledExactlyOnceWith(['a', 'b', 'c', 'd'], expect.anything());
    const dialogText = document.body.textContent;
    expect(dialogText).toContain('models.identifyUnknownBecomes(Unknown,SDXL Main Model)');
    expect(dialogText).toContain('missing projection');
    expect(dialogText).toContain('file is gone');

    await act(() => userEvent.click(document.querySelector('input[aria-label="Model B"]')!.closest('label')!));
    const apply = [...document.querySelectorAll('button')].find((el) =>
      el.textContent.startsWith('models.identifyUnknownApply')
    )!;
    await act(() => userEvent.click(apply));

    await expect.poll(() => api.bulkReidentifyModels.mock.calls.length).toBe(1);
    expect(api.bulkReidentifyModels).toHaveBeenCalledExactlyOnceWith(['a'], expect.anything());
  });

  it('shows progress as batches arrive and issues no further batch once stopped', async () => {
    const many = Array.from({ length: 25 }, (_, index) => unknown(`m${index}`, `Many ${index}`));
    let releaseSecond!: () => void;
    const outcomes = (keys: string[]): ModelIdentificationOutcome[] =>
      keys.map((key) => ({ error: 'gone', key, status: 'failed' }));

    await act(() => root.unmount());
    setModelsSnapshotForTests({ models: many, status: 'loaded' });
    api.bulkGetModelIdentification
      .mockReset()
      .mockImplementationOnce((keys) => Promise.resolve(outcomes(keys)))
      .mockImplementationOnce(
        (keys) =>
          new Promise((resolve) => {
            releaseSecond = () => resolve(outcomes(keys));
          })
      )
      .mockImplementation((keys) => Promise.resolve(outcomes(keys)));
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <IdentifyUnknownModelsDialog isOpen onClose={() => undefined} onExitComplete={() => undefined} />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownProgress(10,25)');
    await act(() =>
      userEvent.click(
        [...document.querySelectorAll('button')].find((el) => el.textContent === 'models.identifyUnknownStop')!
      )
    );
    await act(() => Promise.resolve(releaseSecond()));

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownStopped(20,25)');
    expect(api.bulkGetModelIdentification).toHaveBeenCalledTimes(2);
  });

  const mountScan = async (models: ModelConfig[]) => {
    await act(() => root.unmount());
    setModelsSnapshotForTests({ models, status: 'loaded' });
    root = createRoot(host);
    await act(() =>
      root.render(
        <ChakraProvider value={system}>
          <QueryClientProvider client={new QueryClient()}>
            <IdentifyUnknownModelsDialog isOpen onClose={() => undefined} onExitComplete={() => undefined} />
          </QueryClientProvider>
        </ChakraProvider>
      )
    );
  };
  const buttonByText = (text: string) =>
    [...document.querySelectorAll('button')].find((el) => el.textContent === text)!;

  it('retries after a failed batch with the files not yet checked and keeps the earlier results', async () => {
    const many = Array.from({ length: 25 }, (_, index) => unknown(`m${index}`, `Many ${index}`));
    let failSecond!: () => void;

    api.bulkGetModelIdentification
      .mockReset()
      .mockImplementationOnce((keys) => Promise.resolve(keys.map((key) => proposal(unknown(key, key)))))
      .mockImplementationOnce(
        () =>
          new Promise((_resolve, reject) => {
            failSecond = () => reject(new Error('batch failed'));
          })
      )
      .mockImplementation((keys) => Promise.resolve(keys.map((key) => proposal(unknown(key, key)))));
    await mountScan(many);

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownProgress(10,25)');
    // The Stop button is a control, not part of the live region that announces progress.
    expect(document.querySelector('[role="status"] button')).toBeNull();
    await act(() => userEvent.click(buttonByText('models.identifyUnknownStop')));
    await act(() => failSecond());

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownFailed');
    expect(document.body.textContent).toContain('models.identifyUnknownCanBeIdentified(10)');
    expect(document.body.textContent).not.toContain('models.identifyUnknownNoneFound');

    await act(() => userEvent.click(buttonByText('common.retry')));

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownCanBeIdentified(25)');
    expect(api.bulkGetModelIdentification.mock.calls.map(([keys]) => keys[0])).toEqual(['m0', 'm10', 'm10', 'm20']);
  });

  it('does not claim nothing was identifiable when the first batch fails', async () => {
    api.bulkGetModelIdentification.mockReset().mockRejectedValue(new Error('down'));
    await mountScan([A, B]);

    await expect.poll(() => document.body.textContent).toContain('models.identifyUnknownFailed');
    expect(document.body.textContent).not.toContain('models.identifyUnknownNoneFound');
  });
});

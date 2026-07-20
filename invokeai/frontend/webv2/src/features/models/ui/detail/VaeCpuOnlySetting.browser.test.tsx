import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelsSnapshot, setModelsSnapshotForTests, useModelsSelector } from '@features/models/data/modelsStore';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { supportsVaeCpuOnlySetting, VaeCpuOnlySetting } from './VaeCpuOnlySetting';

const api = vi.hoisted(() => ({
  getModelsDir: vi.fn(),
  listMissingModels: vi.fn(),
  listModels: vi.fn(),
  updateModel: vi.fn(),
}));

vi.mock('@features/models/data/api', () => api);
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const vae = {
  base: 'sdxl',
  cpu_only: null,
  description: null,
  file_size: 1024,
  format: 'diffusers',
  hash: 'hash',
  key: 'sdxl-vae',
  name: 'SDXL VAE',
  source: 'sdxl-vae',
  source_type: 'path',
  type: 'vae',
} as ModelConfig;

const deferred = <T,>() => {
  let reject!: (reason?: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, reject, resolve };
};

const VaeCpuOnlySettingHarness = ({
  onError,
  onSaved,
}: {
  onError: (message: string) => void;
  onSaved: () => void;
}) => {
  const model = useModelsSelector((snapshot) => snapshot.models.find((candidate) => candidate.key === vae.key));

  return model ? <VaeCpuOnlySetting model={model} onError={onError} onSaved={onSaved} /> : null;
};

describe('VaeCpuOnlySetting', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onError: Mock<(message: string) => void>;
  let onSaved: Mock<() => void>;

  beforeEach(async () => {
    api.updateModel.mockReset();
    onError = vi.fn();
    onSaved = vi.fn();
    setModelsSnapshotForTests({ models: [vae], status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <VaeCpuOnlySettingHarness onError={onError} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    setModelsSnapshotForTests({ models: [], status: 'idle' });
  });

  it('is available only for VAE model details', () => {
    expect(supportsVaeCpuOnlySetting({ type: 'vae' })).toBe(true);
    expect(supportsVaeCpuOnlySetting({ type: 'main' })).toBe(false);
  });

  it('saves immediately, disables while pending, and accepts the canonical response', async () => {
    const request = deferred<ModelConfig>();
    api.updateModel.mockReturnValue(request.promise);
    const input = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

    await act(async () => {
      input.click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(vae.key, { cpu_only: true });
    expect(input.checked).toBe(true);
    expect(input.disabled).toBe(true);
    expect(getModelsSnapshot().models[0]?.cpu_only).toBe(true);

    await act(async () => {
      request.resolve({ ...vae, cpu_only: true });
      await request.promise;
    });

    expect(input.checked).toBe(true);
    expect(input.disabled).toBe(false);
    expect(onSaved).toHaveBeenCalledOnce();
    expect(onError).not.toHaveBeenCalled();
  });

  it('restores the previous value and reports a failed save', async () => {
    const request = deferred<ModelConfig>();
    api.updateModel.mockReturnValue(request.promise);
    const input = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

    await act(async () => {
      input.click();
      await Promise.resolve();
    });
    expect(getModelsSnapshot().models[0]?.cpu_only).toBe(true);

    await act(async () => {
      request.reject(new Error('save exploded'));
      await request.promise.catch(() => undefined);
    });

    expect(getModelsSnapshot().models[0]?.cpu_only).toBeNull();
    expect(input.checked).toBe(false);
    expect(input.disabled).toBe(false);
    expect(onError).toHaveBeenCalledWith('save exploded');
    expect(onSaved).not.toHaveBeenCalled();
  });
});

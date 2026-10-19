import type { ModelConfig } from '@features/models/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getModelsSnapshot, setModelsSnapshotForTests, useModelsSelector } from '@features/models/data/modelsStore';
import { type AccountScope, accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { CpuOnlySetting, supportsCpuOnlySetting } from './CpuOnlySetting';

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

const CpuOnlySettingHarness = ({ onError, onSaved }: { onError: (message: string) => void; onSaved: () => void }) => {
  const model = useModelsSelector((snapshot) => snapshot.models.find((candidate) => candidate.key === vae.key));

  return model ? <CpuOnlySetting model={model} onError={onError} onSaved={onSaved} /> : null;
};

describe('CpuOnlySetting', () => {
  let host: HTMLDivElement;
  let root: Root;
  let onError: Mock<(message: string) => void>;
  let onSaved: Mock<() => void>;
  let owner: AccountScope;

  beforeEach(async () => {
    api.updateModel.mockReset();
    onError = vi.fn();
    onSaved = vi.fn();
    owner = accountLifecycle.activate('vae-setting-test-a', ':user:vae-setting-test-a');
    setModelsSnapshotForTests({ models: [vae], status: 'loaded' });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <CpuOnlySettingHarness onError={onError} onSaved={onSaved} />
        </ChakraProvider>
      );
    });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('is available exactly for the types whose config carries cpu_only', () => {
    const supported = [
      'vae',
      'clip_embed',
      't5_encoder',
      'qwen3_encoder',
      'clip_vision',
      'siglip',
      'llava_onevision',
      'text_llm',
      'wan_t5_encoder',
      'gemma2_encoder',
      'mistral_encoder',
      'qwen3_vl_encoder',
    ];

    for (const type of supported) {
      expect(supportsCpuOnlySetting({ type }), type).toBe(true);
    }

    // These configs have no cpu_only field; the PATCH would silently drop it.
    for (const type of ['main', 'lora', 'qwen_vl_encoder', 'pid_decoder']) {
      expect(supportsCpuOnlySetting({ type }), type).toBe(false);
    }
  });

  it('saves immediately, disables while pending, and accepts the canonical response', async () => {
    const request = deferred<ModelConfig>();
    api.updateModel.mockReturnValue(request.promise);
    const input = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

    await act(async () => {
      input.click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(vae.key, { cpu_only: true }, owner.signal);
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

  it('does not commit a late response from account A into account B', async () => {
    const request = deferred<ModelConfig>();
    api.updateModel.mockReturnValue(request.promise);
    const input = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

    await act(async () => {
      input.click();
      await Promise.resolve();
    });

    await act(async () => {
      accountLifecycle.activate('vae-setting-test-b', ':user:vae-setting-test-b');
      setModelsSnapshotForTests({ models: [{ ...vae, cpu_only: null }], status: 'loaded' });
      await Promise.resolve();
    });

    await act(async () => {
      request.resolve({ ...vae, cpu_only: true });
      await request.promise;
    });

    expect(getModelsSnapshot().models[0]?.cpu_only).toBeNull();
    expect(onSaved).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });

  it('does not roll back account B or report account A mutation failures', async () => {
    const request = deferred<ModelConfig>();
    api.updateModel.mockReturnValue(request.promise);
    const input = host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;

    await act(async () => {
      input.click();
      await Promise.resolve();
    });

    await act(async () => {
      accountLifecycle.activate('vae-setting-test-b', ':user:vae-setting-test-b');
      setModelsSnapshotForTests({ models: [{ ...vae, cpu_only: true }], status: 'loaded' });
      await Promise.resolve();
    });

    await act(async () => {
      request.reject(new Error('account A failed late'));
      await request.promise.catch(() => undefined);
    });

    expect(getModelsSnapshot().models[0]?.cpu_only).toBe(true);
    expect(onSaved).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});

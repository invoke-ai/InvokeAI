import type * as ModelsApi from '@features/models/data/api';

import { ChakraProvider } from '@chakra-ui/react';
import { getFp8StorageSupportSnapshot } from '@features/models/data/fp8StorageSupportStore';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DefaultSettingsModel } from './defaultSettingsFields';

import { DefaultSettingsSection } from './DefaultSettingsSection';

const api = vi.hoisted(() => ({
  getFp8StorageSupport: vi.fn(),
  getModelsDir: vi.fn(() => Promise.resolve('')),
  listMissingModels: vi.fn(() => Promise.resolve([])),
  listModels: vi.fn(() => Promise.resolve([])),
  updateModel: vi.fn(),
}));

vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  ...api,
}));
// Interpolating rather than echoing the key: each row's switch is named `models.customizeDefaultField` with the field
// in it, so without this every switch on the panel has the same accessible name and none of them can be addressed by
// it.
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, values?: Record<string, string>) => (values ? `${key}:${Object.values(values).join(',')}` : key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const FP8_LABEL = 'models.defaultFields.fp8Storage';
const PREPROCESSOR_LABEL = 'models.defaultFields.preprocessor';

const settingsModel = (overrides: Partial<DefaultSettingsModel>): DefaultSettingsModel =>
  ({
    base: 'flux',
    default_settings: null,
    format: 'checkpoint',
    key: 'model-key',
    type: 'main',
    ...overrides,
  }) as DefaultSettingsModel;

const row = (base: string, type: string, format: string, supported: boolean) => ({ base, format, supported, type });

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });

  return { promise, resolve };
};

const labels = (host: HTMLElement): string[] =>
  [...host.querySelectorAll('p, span, div')].map((node) => node.textContent ?? '');

const showsFp8Row = (host: HTMLElement): boolean => labels(host).includes(FP8_LABEL);

describe('DefaultSettingsSection FP8 Storage row', () => {
  let host: HTMLDivElement;
  let root: Root;

  const render = async (model: DefaultSettingsModel) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <DefaultSettingsSection model={model} onError={vi.fn()} onSaved={vi.fn()} />
        </ChakraProvider>
      );
    });
  };

  beforeEach(() => {
    vi.resetModules();
    api.getFp8StorageSupport.mockReset();
    api.updateModel.mockReset();
    accountLifecycle.activate('fp8-row-test', ':user:fp8-row-test');
    expect(getFp8StorageSupportSnapshot()).toMatchObject({ byKey: null, status: 'idle' });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    accountLifecycle.invalidate();
  });

  it('shows the row for a model whose loader implements the cast', async () => {
    api.getFp8StorageSupport.mockResolvedValue([row('flux', 'main', 'checkpoint', true)]);

    await render(settingsModel({}));
    await act(async () => {
      await Promise.resolve();
    });

    expect(showsFp8Row(host)).toBe(true);
  });

  it('leaves it out for a FLUX ControlNet, whose loader never casts, and keeps the rest of the section', async () => {
    api.getFp8StorageSupport.mockResolvedValue([
      row('flux', 'main', 'checkpoint', true),
      row('flux', 'controlnet', 'checkpoint', false),
    ]);

    await render(settingsModel({ type: 'controlnet' }));
    await act(async () => {
      await Promise.resolve();
    });

    expect(showsFp8Row(host)).toBe(false);
    expect(labels(host)).toContain(PREPROCESSOR_LABEL);
  });

  it('leaves it out for a quantized main model', async () => {
    api.getFp8StorageSupport.mockResolvedValue([
      row('flux', 'main', 'checkpoint', true),
      row('flux', 'main', 'gguf_quantized', false),
    ]);

    await render(settingsModel({ format: 'gguf_quantized' }));
    await act(async () => {
      await Promise.resolve();
    });

    expect(showsFp8Row(host)).toBe(false);
  });

  it('waits for the table rather than showing a row it cannot vouch for', async () => {
    const request = deferred<ReturnType<typeof row>[]>();
    api.getFp8StorageSupport.mockReturnValue(request.promise);

    await render(settingsModel({}));

    expect(showsFp8Row(host)).toBe(false);

    await act(async () => {
      request.resolve([row('flux', 'main', 'checkpoint', true)]);
      await request.promise;
    });

    expect(showsFp8Row(host)).toBe(true);
  });

  it('saves the setting the row turns on', async () => {
    api.getFp8StorageSupport.mockResolvedValue([row('flux', 'main', 'checkpoint', true)]);
    api.updateModel.mockResolvedValue({ ...settingsModel({}), default_settings: { fp8_storage: true } });

    await render(settingsModel({}));
    await act(async () => {
      await Promise.resolve();
    });

    const fp8Switch = [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((input) =>
      input.closest('div')?.textContent?.includes(FP8_LABEL)
    );
    expect(fp8Switch, 'the FP8 Storage row renders no switch').toBeDefined();

    await act(async () => {
      fp8Switch!.click();
      await Promise.resolve();
    });
    const save = [...host.querySelectorAll('button')].find((button) => button.textContent === 'models.saveDefaults');
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(
      'model-key',
      expect.objectContaining({ default_settings: expect.objectContaining({ fp8_storage: true }) }),
      expect.anything()
    );
  });

  it('asks the backend once, not once per model the user opens', async () => {
    api.getFp8StorageSupport.mockResolvedValue([row('flux', 'main', 'checkpoint', true)]);

    await render(settingsModel({}));
    await act(async () => {
      await Promise.resolve();
    });
    await act(() => root.unmount());
    host.remove();
    await render(settingsModel({ key: 'another-model' }));
    await act(async () => {
      await Promise.resolve();
    });

    expect(api.getFp8StorageSupport).toHaveBeenCalledTimes(1);
  });
});

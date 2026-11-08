import type { ModelConfig } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';
import type { ModelSchedulerPolicy, ModelsUiAdapter, SchedulerPolicyState } from '@features/models/ui/ModelsUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { getFp8StorageSupportSnapshot } from '@features/models/data/fp8StorageSupportStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { DefaultSettingsModel } from './defaultSettingsFields';

import { DefaultSettingsSection } from './DefaultSettingsSection';

/**
 * The FP8 Storage row is shown only where the backend says the setting reaches the model.
 *
 * It used to be shown for every `main`, `controlnet` and `t2i_adapter` record, which offered it for 19
 * loader keys that ignore it -- a GGUF main model whose packed weights must never be re-encoded, a FLUX
 * ControlNet whose loader never casts. The answer now comes from the server, per `(base, type, format)`.
 */

const api = vi.hoisted(() => ({
  getFp8StorageSupport: vi.fn(),
  // The models store fetches on activation; stubbed so this test touches no network at all.
  getModelsDir: vi.fn(() => Promise.resolve('')),
  listMissingModels: vi.fn(() => Promise.resolve([])),
  listModels: vi.fn(() => Promise.resolve([])),
  updateModel: vi.fn(),
}));

// Spread over the real module rather than replacing it: the section pulls in the models store, which
// imports the rest of this transport module, and a bare replacement drops those exports.
vi.mock('@features/models/data/api', async (importOriginal) => ({
  ...(await importOriginal<typeof ModelsApi>()),
  ...api,
}));
// Interpolating rather than echoing the key: each row's switch is named
// `models.customizeDefaultField` with the field in it, so without this every switch on the panel has
// the same accessible name and none of them can be addressed by it.
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
    // The store is a static import, so `resetModules` does not touch it -- only the account lifecycle
    // clears it. Asserted rather than assumed: if that stopped working, the later cells would inherit
    // the first one's table and pass for the wrong reason.
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

    // Same base as the supported row above: this is the case a per-architecture answer cannot express.
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

    // In flight: hidden, because a control that turns out to be inert is the defect being removed and
    // a control that appears a moment late is not.
    expect(showsFp8Row(host)).toBe(false);

    await act(async () => {
      request.resolve([row('flux', 'main', 'checkpoint', true)]);
      await request.promise;
    });

    // And it appears when the answer arrives, which is what the subscription is for.
    expect(showsFp8Row(host)).toBe(true);
  });

  it('saves the setting the row turns on', async () => {
    // The other cells only prove the row is rendered. Without this one the `Switch` could be deleted
    // from it and every assertion here would still pass, which is the same shape of defect as a
    // control that renders and does nothing.
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

const TEXT_ENCODERS_LABEL = 'models.defaultFields.textEncoders';
const qwen3_8b = { base: 'any', key: 'qwen3-8b', name: 'Qwen3 8B', type: 'qwen3_encoder', variant: 'qwen3_8b' };
const qwen3_4b = { ...qwen3_8b, key: 'qwen3-4b', name: 'Qwen3 4B', variant: 'qwen3_4b' };
/** A Klein 9B's rules: one Qwen3 slot that takes only the 8B encoder. */
const KLEIN_POLICY: ModelSchedulerPolicy = {
  appliesToGraph: false,
  defaultValue: 'euler',
  options: [],
  textEncoderSlots: [
    {
      accepts: (candidate) => candidate.variant === 'qwen3_8b',
      key: 'qwen3EncoderModel',
      label: 'Qwen3 Encoder',
      modelTypes: ['qwen3_encoder'],
    },
  ],
};
const KLEIN_POLICY_STATE: SchedulerPolicyState = { read: () => KLEIN_POLICY, status: 'ready' };
const KLEIN_RULES_ADAPTER: ModelsUiAdapter = {
  canManageModels: true,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => true,
  managerProjectId: null,
  schedulerPolicy: { getSnapshot: () => KLEIN_POLICY_STATE, load: () => undefined, subscribe: () => () => undefined },
  setInstallInPlace: () => undefined,
};

describe('DefaultSettingsSection text encoders row', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    api.getFp8StorageSupport.mockResolvedValue([]);
    api.updateModel.mockReset();
    accountLifecycle.activate('text-encoders-test', ':user:text-encoders-test');
    setModelsSnapshotForTests({ error: null, models: [qwen3_4b, qwen3_8b] as ModelConfig[], status: 'loaded' });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    accountLifecycle.invalidate();
  });

  it('saves the encoder picked for a slot, offering only those Generate accepts there', async () => {
    const model = settingsModel({ base: 'flux2', format: 'gguf_quantized', variant: 'klein_9b' });
    api.updateModel.mockResolvedValue({ ...model, default_settings: { text_encoders: [qwen3_8b.key] } });
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={KLEIN_RULES_ADAPTER}>
            <DefaultSettingsSection model={model} onError={vi.fn()} onSaved={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    const panel = [...host.querySelectorAll<HTMLElement>('div')].find(
      (node) => node.firstElementChild?.textContent?.startsWith(TEXT_ENCODERS_LABEL) && node.querySelector('input')
    );
    expect(panel, 'the text encoders row is missing').toBeDefined();

    await act(async () => {
      panel!.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click();
      await Promise.resolve();
    });
    await act(() => panel!.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!.click());
    await expect
      .poll(() => [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? ''))
      .toEqual([expect.stringContaining('Qwen3 8B')]);

    await act(() => document.querySelector<HTMLElement>('[role="option"]')!.click());
    const save = [...host.querySelectorAll('button')].find((button) => button.textContent === 'models.saveDefaults');
    await act(async () => {
      save!.click();
      await Promise.resolve();
    });

    expect(api.updateModel).toHaveBeenCalledWith(
      'model-key',
      expect.objectContaining({ default_settings: expect.objectContaining({ text_encoders: [qwen3_8b.key] }) }),
      expect.anything()
    );
  });
});

import type { ModelConfig, ModelTaxonomyType } from '@features/models/core/types';
import type * as ModelsApi from '@features/models/data/api';
import type {
  DefaultComponentSlotPolicy,
  DefaultComponentSlotReader,
  ModelDefaultsRulesSource,
  ModelDefaultsRulesState,
  ModelsUiAdapter,
} from '@features/models/ui/ModelsUiContext';

import { ChakraProvider } from '@chakra-ui/react';
import { getFp8StorageSupportSnapshot } from '@features/models/data/fp8StorageSupportStore';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider, NO_DEFAULTS_RULES } from '@features/models/ui/ModelsUiContext';
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

describe('DefaultSettingsSection default components', () => {
  let host: HTMLDivElement;
  let root: Root;

  const component = (key: string, name: string, type: string) => ({ base: 'any', key, name, type }) as ModelConfig;
  const catalog = [
    component('t5-a', 'T5 A', 't5_encoder'),
    component('t5-b', 'T5 B', 't5_encoder'),
    component('qwen3-a', 'Qwen3 A', 'qwen3_encoder'),
    component('tower-a', 'Tower A', 'qwen3_vl_vision_tower'),
    { ...component('flux2-vae', 'FLUX.2 VAE', 'vae'), base: 'flux2' },
    { ...component('sdxl-vae', 'SDXL VAE', 'vae'), base: 'sdxl' },
    { ...component('moss-vae', 'MOSS VAE', 'vae'), base: 'moss-image' },
  ];
  // The `vae.accepted` bases of the served rows these tests read: MOSS-Image decodes with FLUX.2's VAE as well as
  // its own; a row without the facet accepts its own base only. The real rule is covered beside the reader.
  const ACCEPTED_VAE_BASES: Record<string, string[]> = { 'moss-image': ['flux2', 'moss-image'] };
  const acceptsDefaultVae: Extract<ModelDefaultsRulesState, { status: 'ready' }>['acceptsDefaultVae'] = (
    model,
    candidate
  ) => (ACCEPTED_VAE_BASES[String(model.base)] ?? [String(model.base)]).includes(String(candidate.base));
  const slot = (persistenceKey: string, label: string, type: string): DefaultComponentSlotPolicy => ({
    filter: (candidate) => candidate.type === type,
    label,
    modelTypes: [type as ModelTaxonomyType],
    persistenceKey,
  });
  // Generation's rules are the app's to supply; this stands in for them with a family whose vision tower is offered
  // only beside a Qwen3 encoder, and whose PiD decoder has no installed model.
  const readComponentSlots: DefaultComponentSlotReader = (_model, components) => [
    slot('t5_encoder', 'T5 Encoder', 't5_encoder'),
    slot('qwen3_encoder', 'Qwen3 Encoder', 'qwen3_encoder'),
    ...(components.qwen3_encoder ? [slot('qwen3_vl_vision_tower', 'Vision Tower', 'qwen3_vl_vision_tower')] : []),
    slot('pid_decoder', 'PiD Decoder', 'pid_decoder'),
  ];
  const readyState: ModelDefaultsRulesState = {
    acceptsDefaultVae,
    read: () => ({ appliesToGraph: false, defaultValue: 'euler', options: [] }),
    readComponentSlots,
    status: 'ready',
  };
  /** A rules source whose state a test moves, as the lazy loader does. */
  const createSource = (initial: ModelDefaultsRulesState) => {
    let state = initial;
    const listeners = new Set<() => void>();
    const source: ModelDefaultsRulesSource = {
      getSnapshot: () => state,
      load: () => undefined,
      subscribe: (listener) => {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };

    return {
      set: (next: ModelDefaultsRulesState) => {
        state = next;
        listeners.forEach((listener) => listener());
      },
      source,
    };
  };
  const adapterFor = (source: ModelDefaultsRulesSource): ModelsUiAdapter => ({
    canManageModels: true,
    defaultsRules: source,
    enableModelDescriptions: true,
    installInPlace: true,
    isProjectActive: () => true,
    managerProjectId: null,
    setInstallInPlace: () => undefined,
  });
  const readyAdapter = adapterFor(createSource(readyState).source);
  const unavailableAdapter = adapterFor(NO_DEFAULTS_RULES);

  const render = async (model: DefaultSettingsModel, uiAdapter: ModelsUiAdapter = readyAdapter) => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={uiAdapter}>
            <DefaultSettingsSection model={model} onError={vi.fn()} onSaved={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });
    await act(async () => {
      await Promise.resolve();
    });
  };

  /** The picker's trigger, found from the slot's label. */
  const pickerOf = (label: string): HTMLElement => {
    let node: HTMLElement | null =
      [...host.querySelectorAll<HTMLElement>('label, span, p')].find((candidate) => candidate.textContent === label) ??
      null;

    while (node && !node.querySelector('[aria-haspopup="listbox"]')) {
      node = node.parentElement;
    }

    return node!.querySelector<HTMLElement>('[aria-haspopup="listbox"]')!;
  };
  const choose = async (label: string, name: string) => {
    await act(() => pickerOf(label).click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);
    await act(() =>
      [...document.querySelectorAll<HTMLElement>('[role="option"]')]
        .find((option) => option.textContent?.includes(name))!
        .click()
    );
  };
  const clear = (label: string) =>
    act(() => {
      let node: HTMLElement | null = pickerOf(label);

      while (node && !node.querySelector('[aria-label="models.clearSelectedModel"]')) {
        node = node.parentElement;
      }
      node!.querySelector<HTMLElement>('[aria-label="models.clearSelectedModel"]')!.click();
    });
  const saveButton = () =>
    [...host.querySelectorAll('button')].find((candidate) => candidate.textContent === 'models.saveDefaults')!;
  const save = async () => {
    await act(async () => {
      saveButton().click();
      await Promise.resolve();
    });
  };
  const savedDefaults = () => api.updateModel.mock.calls[0]?.[1]?.default_settings;

  beforeEach(() => {
    api.getFp8StorageSupport.mockReset();
    api.getFp8StorageSupport.mockResolvedValue([]);
    api.updateModel.mockReset();
    api.updateModel.mockImplementation((_key, body) => Promise.resolve({ ...settingsModel({}), ...body }));
    accountLifecycle.activate('default-components-test', ':user:default-components-test');
    setModelsSnapshotForTests({ error: null, models: catalog, status: 'loaded' });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
    accountLifecycle.invalidate();
  });

  it('offers a picker for each component slot with a candidate and saves the choice under its slot name', async () => {
    await render(settingsModel({}));

    expect(labels(host)).toEqual(expect.arrayContaining(['T5 Encoder', 'Qwen3 Encoder']));
    // Nothing installed fills the PiD decoder, so it stays out of the way.
    expect(labels(host)).not.toContain('PiD Decoder');

    await choose('T5 Encoder', 'T5 B');
    await save();

    expect(savedDefaults()).toMatchObject({ components: { t5_encoder: 't5-b' } });
  });

  it('keeps a slot that holds a value, even one nothing installed can fill, so it can be cleared', async () => {
    await render(settingsModel({ default_settings: { components: { pid_decoder: 'gone' } } }));
    await clear('PiD Decoder');
    await save();

    expect(savedDefaults().components).toBeNull();
  });

  describe('clearing a slot', () => {
    it('removes it from the stored map and keeps the others', async () => {
      await render(
        settingsModel({ default_settings: { components: { qwen3_encoder: 'qwen3-a', t5_encoder: 't5-a' } } })
      );
      await clear('T5 Encoder');
      await save();

      expect(savedDefaults().components).toEqual({ qwen3_encoder: 'qwen3-a' });
    });

    it('stores no map once the last slot is cleared', async () => {
      await render(settingsModel({ default_settings: { components: { t5_encoder: 't5-a' } } }));
      await clear('T5 Encoder');
      await save();

      expect(savedDefaults().components).toBeNull();
    });

    it('takes a dependent slot along when the slot it hangs on is cleared', async () => {
      await render(
        settingsModel({
          default_settings: { components: { qwen3_encoder: 'qwen3-a', qwen3_vl_vision_tower: 'tower-a' } },
        })
      );
      expect(labels(host)).toContain('Vision Tower');

      await clear('Qwen3 Encoder');
      await save();

      expect(savedDefaults().components).toBeNull();
    });
  });

  it('keeps a stored key for a slot this version does not offer through an unrelated edit', async () => {
    // Deliberate: another version or client may know slots this one does not, and an edit elsewhere must not drop them.
    await render(settingsModel({ default_settings: { components: { t5_encoder: 't5-a', unknown_slot: 'x' } } }));
    await choose('T5 Encoder', 'T5 B');
    await save();

    expect(savedDefaults().components).toEqual({ t5_encoder: 't5-b', unknown_slot: 'x' });
  });

  it('is clean again when the choice is put back to the saved one', async () => {
    await render(settingsModel({ default_settings: { components: { t5_encoder: 't5-a' } } }));
    expect(saveButton().disabled).toBe(true);

    await choose('T5 Encoder', 'T5 B');
    expect(saveButton().disabled).toBe(false);

    await choose('T5 Encoder', 'T5 A');
    expect(saveButton().disabled).toBe(true);
  });

  it('is clean after a save', async () => {
    await render(settingsModel({}));
    await choose('T5 Encoder', 'T5 B');
    await save();

    expect(api.updateModel).toHaveBeenCalledTimes(1);
    expect(saveButton().disabled).toBe(true);
  });

  it('shows no group while the rules are unavailable, and still saves the stored components with other edits', async () => {
    await render(settingsModel({ default_settings: { components: { t5_encoder: 't5-a' } } }), unavailableAdapter);

    expect(labels(host)).not.toContain('models.defaultComponents');

    const stepsSwitch = [...host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]')].find((input) =>
      input.closest('div')?.textContent?.includes('models.defaultFields.steps')
    );
    await act(async () => {
      stepsSwitch!.click();
      await Promise.resolve();
    });
    await save();

    expect(savedDefaults()).toMatchObject({ components: { t5_encoder: 't5-a' }, steps: 30 });
  });

  it('shows the pickers once the rules arrive', async () => {
    const rules = createSource({ status: 'unavailable' });

    await render(settingsModel({}), adapterFor(rules.source));
    expect(labels(host)).not.toContain('T5 Encoder');

    await act(() => rules.set(readyState));

    expect(labels(host)).toContain('T5 Encoder');
  });

  describe('the default VAE picker', () => {
    const mossModel = settingsModel({ base: 'moss-image', default_settings: { vae: 'default' } });
    const offeredVaes = async () => {
      await act(() => pickerOf('models.defaultFields.vae').click());
      await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);

      return [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');
    };

    it('offers the VAEs Generate accepts for the model, whatever their base, and not the ones it rejects', async () => {
      await render(mossModel);
      const offered = (await offeredVaes()).join('|');

      expect(offered).toContain('FLUX.2 VAE');
      expect(offered).toContain('MOSS VAE');
      expect(offered).not.toContain('SDXL VAE');
    });

    it("offers the model's own base only until the rules arrive", async () => {
      await render(mossModel, unavailableAdapter);
      const offered = (await offeredVaes()).join('|');

      expect(offered).toContain('MOSS VAE');
      expect(offered).not.toContain('FLUX.2 VAE');
    });

    it('keeps a stored default VAE the rule no longer accepts, so it can be cleared', async () => {
      await render(settingsModel({ base: 'moss-image', default_settings: { vae: 'sdxl-vae' } }));

      expect(host.textContent).toContain('SDXL VAE');
      await clear('models.defaultFields.vae');
      await save();

      expect(savedDefaults().vae).toBe('default');
    });
  });
});

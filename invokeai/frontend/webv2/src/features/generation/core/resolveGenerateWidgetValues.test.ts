import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { GenerateWidgetValues, MainModelConfig } from './types';

import capabilitiesFixture from './__fixtures__/architectureCapabilities.json';
import {
  type ArchitectureCapabilitiesRow,
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from './architectureCapabilities';
import { getDefaultGenerateSettings } from './baseGenerationPolicies';
import { resolveGenerateWidgetValues } from './resolveGenerateWidgetValues';

const createModel = (key: string, overrides: Partial<MainModelConfig> = {}): MainModelConfig => ({
  base: 'sdxl',
  key,
  name: key,
  type: 'main',
  ...overrides,
});

const createValues = (model: MainModelConfig, overrides: Partial<GenerateWidgetValues> = {}): GenerateWidgetValues => ({
  ...getDefaultGenerateSettings(model),
  model,
  seed: 1,
  seedMode: 'fixed',
  ...overrides,
});

const applySystemPatch = (
  storedValues: Record<string, unknown>,
  systemPatch: Partial<GenerateWidgetValues>
): Record<string, unknown> => ({ ...storedValues, ...systemPatch });

// Seed backend fixtures and clean up the registry to isolate policy tests.
beforeEach(() => {
  setArchitectureCapabilities(capabilitiesFixture as ArchitectureCapabilitiesRow[]);
});

afterEach(resetArchitectureCapabilities);

describe('resolveGenerateWidgetValues', () => {
  it('returns null when the catalog has no supported generation model', () => {
    expect(
      resolveGenerateWidgetValues({
        models: [{ base: 'sdxl', key: 'control', name: 'ControlNet', type: 'controlnet' }],
        storedValues: {},
      })
    ).toBeNull();
  });

  it.each(['checkpoint', 'gguf_quantized'] as const)(
    'never defaults to a model that cannot run on its own (%s)',
    (format) => {
      // Default resolution excludes unconditional Ideogram branches that cannot generate independently.
      const unconditional = createModel('ideogram4-uncond', {
        base: 'ideogram-4',
        branch: 'unconditional',
        format,
      });
      const conditional = createModel('ideogram4-cond', {
        base: 'ideogram-4',
        branch: 'conditional',
        format,
      });

      const result = resolveGenerateWidgetValues({ models: [unconditional, conditional], storedValues: {} });

      expect(result?.values.model.key).toBe('ideogram4-cond');
      // And with nothing else installed there is no default to fall back to at all.
      expect(resolveGenerateWidgetValues({ models: [unconditional], storedValues: {} })).toBeNull();
    }
  );

  it('creates canonical defaults for the first supported model', () => {
    const unsupported = { base: 'sdxl', key: 'control', name: 'ControlNet', type: 'controlnet' };
    const first = createModel('first');
    const second = createModel('second');
    const result = resolveGenerateWidgetValues({ models: [unsupported, first, second], storedValues: {} });

    expect(result?.values).toMatchObject({
      batchCount: 1,
      model: first,
      modelKey: first.key,
      negativePrompt: '',
      positivePrompt: '',
    });
    expect(result?.systemPatch).not.toBeNull();
    expect(result?.systemPatch).not.toHaveProperty('batchCount');
  });

  it('reads dynamic prompts as on for a project saved before the toggle, without writing the flag in', () => {
    const model = createModel('current');
    const { dynamicPromptsEnabled: _absent, ...storedValues } = createValues(model);
    const result = resolveGenerateWidgetValues({ models: [model], storedValues });

    expect(result?.values.dynamicPromptsEnabled).toBe(true);
    expect(result?.systemPatch).toBeNull();
  });

  it('keeps dynamic prompts off once the user turned them off', () => {
    const model = createModel('current');
    const storedValues = createValues(model, { dynamicPromptsEnabled: false });
    const result = resolveGenerateWidgetValues({ models: [model], storedValues });

    expect(result?.values.dynamicPromptsEnabled).toBe(false);
    expect(result?.systemPatch).toBeNull();
  });

  it('keeps a chosen model that is no longer installed, so the card can name it and Invoke waits', () => {
    const removed = createModel('removed');
    const available = createModel('available');
    const storedValues = createValues(removed, { positivePrompt: 'keep this prompt' });
    const result = resolveGenerateWidgetValues({ models: [available], storedValues });

    expect(result?.values).toBe(storedValues);
    expect(result?.values.model).toBe(removed);
    expect(result?.systemPatch).toBeNull();
  });

  it('finds a reinstalled model again by its file hash, though it came back under a new key', () => {
    const removed = createModel('removed', { hash: 'blake3:abc' });
    const reinstalled = createModel('reinstalled', { hash: 'blake3:abc', name: 'Same file' });
    const storedValues = createValues(removed, { positivePrompt: 'keep this prompt' });
    const result = resolveGenerateWidgetValues({ models: [createModel('other'), reinstalled], storedValues });

    expect(result?.values.model).toBe(reinstalled);
    expect(result?.values.positivePrompt).toBe('keep this prompt');
    expect(result?.systemPatch).toMatchObject({ model: reinstalled, modelKey: reinstalled.key });
  });

  it('does not take a file with the same hash registered under another base or variant for the saved model', () => {
    const removed = createModel('removed', { hash: 'blake3:abc', variant: 'normal' });
    const otherBase = createModel('other-base', { base: 'sd-1', hash: 'blake3:abc', variant: 'normal' });
    const otherVariant = createModel('other-variant', { hash: 'blake3:abc', variant: 'inpaint' });
    const storedValues = createValues(removed);
    const result = resolveGenerateWidgetValues({ models: [otherBase, otherVariant], storedValues });

    expect(result?.values.model).toBe(removed);
  });

  it('finds a reinstalled concept again by its file hash, unless the list already holds it', () => {
    const model = createModel('current');
    const removedLora = { base: 'sdxl', hash: 'blake3:lora', key: 'old-lora', name: 'Style', type: 'lora' as const };
    const reinstalledLora = { ...removedLora, key: 'new-lora' };
    const storedValues = createValues(model, { loras: [{ isEnabled: true, model: removedLora, weight: 0.7 }] });
    const result = resolveGenerateWidgetValues({ models: [model, reinstalledLora], storedValues });

    expect(result?.values.loras).toEqual([{ isEnabled: true, model: reinstalledLora, weight: 0.7 }]);
    expect(result?.systemPatch?.loras).toEqual(result?.values.loras);

    const both = createValues(model, {
      loras: [
        { isEnabled: true, model: removedLora, weight: 0.7 },
        { isEnabled: true, model: reinstalledLora, weight: 1 },
      ],
    });

    expect(resolveGenerateWidgetValues({ models: [model, reinstalledLora], storedValues: both })?.values.loras).toBe(
      both.loras
    );

    // Two removed copies of one file: only the first takes the reinstalled concept; the other stays as saved.
    const copy = { ...removedLora, key: 'old-lora-copy' };
    const twice = createValues(model, {
      loras: [
        { isEnabled: true, model: removedLora, weight: 0.7 },
        { isEnabled: true, model: copy, weight: 0.4 },
      ],
    });

    expect(
      resolveGenerateWidgetValues({ models: [model, reinstalledLora], storedValues: twice })?.values.loras.map(
        (lora) => lora.model.key
      )
    ).toEqual(['new-lora', 'old-lora-copy']);
  });

  it('defaults a choice that never named a model', () => {
    const available = createModel('available');
    const { model: _, ...withoutModel } = createValues(createModel('unknown'), { modelKey: '' });
    const result = resolveGenerateWidgetValues({ models: [available], storedValues: withoutModel });

    expect(result?.values.model).toBe(available);
    expect(result?.systemPatch).toMatchObject({ model: available, modelKey: available.key });
  });

  it('keeps authored settings while replacing a selection of an unsupported family with the default model', () => {
    const unavailable = createModel('unavailable', { base: 'not-supported' });
    const available = createModel('available');
    const storedValues = createValues(unavailable, { positivePrompt: 'keep this prompt' });
    const result = resolveGenerateWidgetValues({ models: [available], storedValues });

    expect(result?.values.model).toBe(available);
    expect(result?.values.modelKey).toBe(available.key);
    expect(result?.values.positivePrompt).toBe('keep this prompt');
    expect(result?.systemPatch).toMatchObject({ model: available, modelKey: available.key });
  });

  it('uses a valid selected model instead of the first supported model', () => {
    const first = createModel('first');
    const selected = createModel('selected');
    const storedValues = createValues(selected);
    const result = resolveGenerateWidgetValues({ models: [first, { ...selected }], storedValues });

    expect(result?.values).toBe(storedValues);
    expect(result?.systemPatch).toBeNull();
  });

  it('refreshes denormalized model snapshots from the catalog', () => {
    const stale = createModel('selected', { name: 'Stale name' });
    const current = createModel('selected', { name: 'Current name' });
    const storedValues = createValues(stale);
    const result = resolveGenerateWidgetValues({ models: [current], storedValues });

    expect(result?.values.model).toBe(current);
    expect(result?.systemPatch).toMatchObject({ model: current });
  });

  it('refreshes an applied prompt-template snapshot from a successful catalog', () => {
    const model = createModel('model');
    const storedTemplate = {
      id: 'template-1',
      name: 'Old name',
      negativePrompt: 'old negative',
      positivePrompt: 'old {prompt}',
    };
    const currentTemplate = {
      id: 'template-1',
      name: 'Current name',
      negativePrompt: 'current negative',
      positivePrompt: 'current {prompt}',
    };
    const storedValues = createValues(model, { promptTemplate: storedTemplate, promptTemplateViewMode: true });
    const result = resolveGenerateWidgetValues({
      models: [model],
      promptTemplates: [currentTemplate],
      storedValues,
    });

    expect(result?.values.promptTemplate).toEqual(currentTemplate);
    expect(result?.values.promptTemplate).not.toBe(currentTemplate);
    expect(result?.systemPatch?.promptTemplate).toEqual(currentTemplate);
  });

  it.each([
    ['unread or failed', undefined],
    ['successfully read but missing', []],
  ])('preserves the applied prompt-template snapshot when the catalog is %s', (_case, promptTemplates) => {
    const model = createModel('model');
    const storedTemplate = {
      id: 'template-1',
      name: 'Saved name',
      negativePrompt: 'saved negative',
      positivePrompt: 'saved {prompt}',
    };
    const storedValues = createValues(model, { promptTemplate: storedTemplate, promptTemplateViewMode: true });
    const result = resolveGenerateWidgetValues({ models: [model], promptTemplates, storedValues });

    expect(result?.values).toBe(storedValues);
    expect(result?.values.promptTemplate).toBe(storedTemplate);
    expect(result?.systemPatch).toBeNull();
  });

  it('automatically selects a compatible FLUX.2 Diffusers component source', () => {
    const model = createModel('flux2-quantized', {
      base: 'flux2',
      format: 'gguf_quantized',
      variant: 'klein_9b',
    });
    const incompatibleSource = createModel('flux2-4b-source', {
      base: 'flux2',
      format: 'diffusers',
      variant: 'klein_4b',
    });
    const source = createModel('flux2-9b-source', {
      base: 'flux2',
      format: 'diffusers',
      variant: 'klein_9b',
    });
    const storedValues = createValues(model);
    const result = resolveGenerateWidgetValues({
      models: [model, incompatibleSource, source],
      storedValues,
    });

    expect(result?.values.componentSourceModel).toBe(source);
    expect(result?.systemPatch?.componentSourceModel).toBe(source);
  });

  it('maps the random toggle saved before seed modes instead of reusing the record as-is', () => {
    // Reused as-is, the record reaches the seed menu with no mode and the icon lookup throws.
    const model = createModel('model');
    const { seedMode: _, ...legacy } = createValues(model);
    const storedValues = { ...legacy, shouldRandomizeSeed: true };
    const result = resolveGenerateWidgetValues({ models: [model], storedValues });

    expect(result?.values.seedMode).toBe('random');
    expect(result?.systemPatch?.seedMode).toBe('random');
  });

  it('repairs stale template view mode through canonical normalization', () => {
    const model = createModel('model');
    const storedValues = { ...createValues(model), promptTemplate: null, promptTemplateViewMode: true };
    const result = resolveGenerateWidgetValues({ models: [model], storedValues });

    expect(result?.values.promptTemplateViewMode).toBe(false);
    expect(result?.systemPatch?.promptTemplateViewMode).toBe(false);
  });

  it('preserves topbar-owned batch count without including it in a system patch', () => {
    const stale = createModel('model', { name: 'Stale name' });
    const current = createModel('model', { name: 'Current name' });
    const storedValues = createValues(stale, { batchCount: 7 });
    const result = resolveGenerateWidgetValues({ models: [current], storedValues });

    expect(result?.values.batchCount).toBe(7);
    expect(result?.systemPatch).not.toHaveProperty('batchCount');
  });

  it('reaches a fixed point after applying its system patch', () => {
    const model = createModel('model');
    const first = resolveGenerateWidgetValues({ models: [model], storedValues: {} });

    expect(first?.systemPatch).not.toBeNull();

    const storedValues = applySystemPatch({}, first!.systemPatch!);
    const second = resolveGenerateWidgetValues({ models: [model], storedValues });

    expect(second?.values).toEqual(first?.values);
    expect(second?.systemPatch).toBeNull();
  });
});

describe('without the backend capability table', () => {
  it('resolves nothing rather than falling back to generic defaults', () => {
    // Return null before capabilities load rather than persist fallback defaults.
    resetArchitectureCapabilities();

    const model: MainModelConfig = { base: 'sdxl', key: 'model', name: 'model', type: 'main' };

    expect(resolveGenerateWidgetValues({ models: [model], storedValues: {} })).toBeNull();
  });
});

describe('resolveGenerateWidgetValues and an architecture the table omits', () => {
  const rows = capabilitiesFixture as ArchitectureCapabilitiesRow[];

  it('will not select a model whose architecture the backend did not describe', () => {
    // Reject missing rows to protect persisted settings from generic defaults.
    setArchitectureCapabilities(rows.filter((row) => row.base !== 'cogview4'));

    expect(
      resolveGenerateWidgetValues({ models: [createModel('cogview', { base: 'cogview4' })], storedValues: undefined })
    ).toBeNull();
  });

  it('falls back to a described model rather than blocking the whole catalog', () => {
    setArchitectureCapabilities(rows.filter((row) => row.base !== 'cogview4'));

    const resolved = resolveGenerateWidgetValues({
      models: [createModel('cogview', { base: 'cogview4' }), createModel('sdxl-model')],
      storedValues: { modelKey: 'cogview' },
    });

    expect(resolved?.values.modelKey).toBe('sdxl-model');
  });
});

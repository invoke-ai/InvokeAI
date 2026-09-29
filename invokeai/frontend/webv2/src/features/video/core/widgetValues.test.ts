import type { GenerationModelCatalogItem, MainModelConfig } from '@features/generation/contracts';

import { describe, expect, it } from 'vitest';

import { normalizeVideoWidgetValues } from './settings';
import { getAcceleratorToggleResult } from './videoPolicies';
import {
  createDefaultVideoWidgetValues,
  getVideoWidgetValidationReasons,
  resolveVideoSeed,
  syncVideoWidgetValuesWithModels,
} from './widgetValues';

const wanModel = (variant: string, format = 'diffusers', key = `wan-${variant}-${format}`): MainModelConfig => ({
  base: 'wan',
  format,
  key,
  name: `Wan 2.2 ${variant}`,
  type: 'main',
  variant,
});

const h3Model = (format = 'diffusers', key = `h3-${format}`): MainModelConfig => ({
  base: 'minimax-h3',
  format,
  key,
  name: 'MiniMax H3',
  type: 'main',
  variant: 'fl2va',
});

const lora = (name: string, base = 'wan', variant: string | null = 'a14b'): GenerationModelCatalogItem => ({
  base,
  key: `lora-${name}`,
  name,
  type: 'lora',
  variant,
});

const WAN_VAE_16 = { base: 'wan', key: 'vae16', latent_channels: 16, name: 'Wan 2.1 VAE', type: 'vae' as const };

describe('createDefaultVideoWidgetValues', () => {
  it('seeds the first supported installed model and its family defaults', () => {
    const wan = wanModel('t2v_a14b');
    const values = createDefaultVideoWidgetValues([lora('Style'), wan, h3Model()]);

    expect(values.model?.key).toBe(wan.key);
    expect(values.numFrames).toBe(81);
    expect(values.steps).toBe(40);
  });

  it('picks a single-file MiniMax H3 checkpoint but never a components-only install', () => {
    expect(createDefaultVideoWidgetValues([h3Model('checkpoint')]).model?.key).toBe(h3Model('checkpoint').key);
    expect(createDefaultVideoWidgetValues([{ ...h3Model(), components_only: true }]).model).toBeNull();
  });

  it('turns the accelerator on when its LoRAs are installed', () => {
    const values = createDefaultVideoWidgetValues([h3Model(), lora('MiniMax H3 Turbo LoRA', 'minimax-h3', null)]);

    expect(values).toMatchObject({ acceleratorEnabled: true, steps: 6 });
  });
});

describe('syncVideoWidgetValuesWithModels', () => {
  const model = wanModel('i2v_a14b', 'gguf_quantized');

  it('promotes a legacy transformer override to the top model slot', () => {
    // Migrate legacy transformer overrides to model identity and retain Diffusers components.
    const install = h3Model();
    const checkpoint = { ...h3Model('checkpoint', 'h3-ckpt'), name: 'MiniMax H3 Ref2VA (int8)', variant: 'ref2va' };
    const stored = {
      ...createDefaultVideoWidgetValues([install, checkpoint]),
      h3TransformerModel: checkpoint,
      model: install,
      modelKey: install.key,
    };
    const synced = syncVideoWidgetValuesWithModels(stored, [install, checkpoint]);

    expect(synced.model?.key).toBe(checkpoint.key);
    expect(synced.modelKey).toBe(checkpoint.key);
    expect(synced.componentSourceModel?.key).toBe(install.key);
    expect(synced.h3TransformerModel).toBeNull();
    // Stable: a second sync of the promoted value returns it untouched.
    expect(syncVideoWidgetValuesWithModels(synced, [install, checkpoint])).toBe(synced);
  });

  it('keeps the installed Diffusers main when the legacy transformer override is uninstalled', () => {
    // A missing override must not displace a still-installed main or trigger a different-family fallback.
    const wan = wanModel('t2v_a14b');
    const install = h3Model();
    const checkpoint = { ...h3Model('checkpoint', 'h3-ckpt'), variant: 'ref2va' };
    const stored = {
      ...createDefaultVideoWidgetValues([wan, install, checkpoint]),
      h3TransformerModel: checkpoint,
      model: install,
      modelKey: install.key,
    };
    const synced = syncVideoWidgetValuesWithModels(stored, [wan, install]);

    expect(synced.model?.key).toBe(install.key);
    expect(synced.h3TransformerModel).toBeNull();
    expect(synced.componentSourceModel).toBeNull();
  });

  it('returns the same object when nothing changed', () => {
    const catalog = [model, WAN_VAE_16];
    const values = { ...createDefaultVideoWidgetValues(catalog), vae: WAN_VAE_16 };

    expect(syncVideoWidgetValuesWithModels(values, catalog)).toBe(values);
  });

  it('clears the accelerator flag when no video main is left installed', () => {
    // Clear accelerator keys with empty LoRAs so persisted state remains valid.
    const h3 = h3Model();
    const turbo = { base: 'minimax-h3', key: 'turbo', name: 'MiniMax H3 Turbo LoRA', type: 'lora' as const };
    const values = createDefaultVideoWidgetValues([h3, turbo as never]);

    expect(values).toMatchObject({ acceleratorEnabled: true });

    const synced = syncVideoWidgetValuesWithModels(values, [turbo as never]);

    expect(synced).toMatchObject({ acceleratorEnabled: false, model: null });
    expect(synced.acceleratorLoraKeys).toEqual([]);
    expect(synced.loras).toEqual([]);
    // Still stable: a second sync of the cleared value returns it untouched.
    expect(syncVideoWidgetValuesWithModels(synced, [turbo as never])).toBe(synced);
  });

  it('falls back to an installed supported main when the stored one is gone', () => {
    const replacement = wanModel('t2v_a14b');
    const values = createDefaultVideoWidgetValues([model]);
    const synced = syncVideoWidgetValuesWithModels(values, [replacement]);

    expect(synced.model?.key).toBe(replacement.key);
    expect(synced.modelKey).toBe(replacement.key);
  });

  it('turns auto duration off with an uninstalled head, so a reinstall does not switch it back on', () => {
    const ltx2 = {
      base: 'ltx-2',
      format: 'checkpoint',
      key: 'ltx2',
      name: 'LTX-2.5',
      type: 'main',
      variant: 'ltx2_dev',
    };
    const head = { base: 'ltx-2', key: 'head', name: 'Duration head', type: 'ltx2_duration_head' };
    const values = {
      ...createDefaultVideoWidgetValues([ltx2 as never]),
      autoDuration: true,
      ltx2DurationHeadModel: head as never,
    };

    const synced = syncVideoWidgetValuesWithModels(values, [ltx2 as never]);

    expect(synced.ltx2DurationHeadModel).toBeNull();
    expect(synced.autoDuration).toBe(false);
  });

  it('drops components that no longer pass the slot filter for the resolved model', () => {
    const values = { ...createDefaultVideoWidgetValues([model]), vae: WAN_VAE_16 };
    // Catalog holds the H3 main only — the Wan VAE has no slot there.
    const synced = syncVideoWidgetValuesWithModels(values, [h3Model(), WAN_VAE_16]);

    expect(synced.model?.base).toBe('minimax-h3');
    expect(synced.vae).toBeNull();
  });

  it('keeps a hybrid quality base while its slot accepts it, drops it once the resolved model has no such slot', () => {
    const ref2va: MainModelConfig = { ...h3Model('checkpoint', 'h3-ref2va'), pruned: true, variant: 'ref2va' };
    const fl2vaBase: MainModelConfig = { ...h3Model('checkpoint', 'h3-fl2va-base'), pruned: true };
    const install = h3Model();
    const values = {
      ...createDefaultVideoWidgetValues([ref2va, fl2vaBase, install]),
      componentSourceModel: install,
      h3HybridBaseModel: fl2vaBase,
      h3HybridStartBlock: 30,
      model: ref2va,
      modelKey: ref2va.key,
    };

    // Same object back: the base is installed and still passes the Ref2VA main's slot filter.
    expect(syncVideoWidgetValuesWithModels(values, [ref2va, fl2vaBase, install])).toBe(values);

    // The base uninstalled: the slot value goes, the block setting is plain state and stays.
    const baseGone = syncVideoWidgetValuesWithModels(values, [ref2va, install]);

    expect(baseGone.h3HybridBaseModel).toBeNull();
    expect(baseGone.h3HybridStartBlock).toBe(30);

    // The Ref2VA main uninstalled: the resolved FL2VA main offers no hybrid slot.
    const mainGone = syncVideoWidgetValuesWithModels(values, [fl2vaBase, install]);

    expect(mainGone.model?.key).toBe(fl2vaBase.key);
    expect(mainGone.h3HybridBaseModel).toBeNull();
  });

  it('snaps family constraints when it auto-picks a different-family model', () => {
    // Auto-picked H3 must reconcile Wan sampling values, including fps with no H3 control to repair it.
    const stored = { ...createDefaultVideoWidgetValues([wanModel('t2v_a14b')]), model: null };
    const synced = syncVideoWidgetValuesWithModels(stored, [h3Model()]);

    expect(synced.model?.base).toBe('minimax-h3');
    // Steps carry over (the selection transition preserves user sampling); the
    // family-bound constraints snap.
    expect(synced).toMatchObject({ fps: 24, numFrames: 90, steps: 40, targetResolution: '768 highres' });
    expect(getVideoWidgetValidationReasons(synced, [h3Model()])).toEqual([]);
  });

  it('bootstraps the picked model’s family defaults (accelerator included) for a never-seeded store', () => {
    // Fresh widgets use selected-model defaults, including acceleration, instead of generic healing fallbacks.
    const catalog = [h3Model(), lora('MiniMax H3 Turbo LoRA', 'minimax-h3', null)];
    const healed = normalizeVideoWidgetValues({})!;
    const synced = syncVideoWidgetValuesWithModels(healed, catalog);

    expect(synced.model?.base).toBe('minimax-h3');
    // numFrames 124 is H3's own default — not the Wan fallback (81) snapped
    // onto the H3 grid (90), which is what the preserve path would produce.
    expect(synced).toMatchObject({ acceleratorEnabled: true, cfgScale: 1, fps: 24, numFrames: 124, steps: 6 });

    const seeded = normalizeVideoWidgetValues({
      firstFrameImage: { height: 480, image_name: 'seed.png', width: 832 },
    })!;
    const syncedSeeded = syncVideoWidgetValuesWithModels(seeded, catalog);

    expect(syncedSeeded.firstFrameImage).toEqual({ height: 480, image_name: 'seed.png', width: 832 });
    expect(syncedSeeded.acceleratorEnabled).toBe(true);
  });

  it('drops uninstalled or incompatible LoRAs and clears an orphaned accelerator flag', () => {
    const catalog = [model, lora('Wan Lightning High Noise'), lora('Wan Lightning Low Noise')];
    const enabled = getAcceleratorToggleResult(createDefaultVideoWidgetValues([model]), model, catalog, true).settings;
    const values = { ...enabled, model };

    // Losing accelerator LoRAs restores normal sampling as well as clearing intent; four-step nondistilled runs
    // are invalid defaults.
    const synced = syncVideoWidgetValuesWithModels(values, [model]);

    expect(synced.loras).toEqual([]);
    expect(synced.acceleratorEnabled).toBe(false);
    expect(synced.acceleratorLoraKeys).toEqual([]);
    expect(synced.steps).toBe(40);
    expect(synced.cfgScale).toBe(5);
  });
});

describe('getVideoWidgetValidationReasons', () => {
  it('requires a model first', () => {
    const values = { ...createDefaultVideoWidgetValues(), model: null };

    expect(getVideoWidgetValidationReasons(values)).toEqual(['Video needs a Wan 2.2, MiniMax H3 or LTX-2 main model.']);
  });

  it('passes through settings validation and availability checks', () => {
    const model = wanModel('t2v_a14b');
    const values = createDefaultVideoWidgetValues([model]);

    expect(getVideoWidgetValidationReasons(values, [model])).toEqual([]);
    expect(getVideoWidgetValidationReasons({ ...values, numFrames: 80 }, [model])).toEqual([
      expect.stringContaining('4·n + 1'),
    ]);
    expect(getVideoWidgetValidationReasons(values, [])).toEqual([expect.stringContaining('no longer installed')]);
  });
});

describe('resolveVideoSeed', () => {
  it('keeps a fixed seed and randomizes otherwise', () => {
    expect(resolveVideoSeed({ seed: 42, seedMode: 'fixed' })).toBe(42);

    const randomized = resolveVideoSeed({ seed: 42, seedMode: 'random' });

    expect(Number.isInteger(randomized)).toBe(true);
    expect(randomized).toBeGreaterThanOrEqual(0);
  });
});

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

  it('skips a checkpoint-format MiniMax H3 main (transformer override, not a main)', () => {
    const values = createDefaultVideoWidgetValues([h3Model('checkpoint')]);

    expect(values.model).toBeNull();
  });

  it('turns the accelerator on when its LoRAs are installed', () => {
    const values = createDefaultVideoWidgetValues([h3Model(), lora('MiniMax H3 Turbo LoRA', 'minimax-h3', null)]);

    expect(values).toMatchObject({ acceleratorEnabled: true, steps: 6 });
  });
});

describe('syncVideoWidgetValuesWithModels', () => {
  const model = wanModel('i2v_a14b', 'gguf_quantized');

  it('returns the same object when nothing changed', () => {
    const catalog = [model, WAN_VAE_16];
    const values = { ...createDefaultVideoWidgetValues(catalog), vae: WAN_VAE_16 };

    expect(syncVideoWidgetValuesWithModels(values, catalog)).toBe(values);
  });

  it('falls back to an installed supported main when the stored one is gone', () => {
    const replacement = wanModel('t2v_a14b');
    const values = createDefaultVideoWidgetValues([model]);
    const synced = syncVideoWidgetValuesWithModels(values, [replacement]);

    expect(synced.model?.key).toBe(replacement.key);
    expect(synced.modelKey).toBe(replacement.key);
  });

  it('drops components that no longer pass the slot filter for the resolved model', () => {
    const values = { ...createDefaultVideoWidgetValues([model]), vae: WAN_VAE_16 };
    // Catalog holds the H3 main only — the Wan VAE has no slot there.
    const synced = syncVideoWidgetValuesWithModels(values, [h3Model(), WAN_VAE_16]);

    expect(synced.model?.base).toBe('minimax-h3');
    expect(synced.vae).toBeNull();
  });

  it('snaps family constraints when it auto-picks a different-family model', () => {
    // Wan-shaped stored values (frames 81, fps 16, 720p) with no surviving
    // model, in a catalog whose only supported main is MiniMax H3: without the
    // selection transition the panel would be stuck at fps 16 with no FPS
    // control to fix it.
    const stored = { ...createDefaultVideoWidgetValues([wanModel('t2v_a14b')]), model: null };
    const synced = syncVideoWidgetValuesWithModels(stored, [h3Model()]);

    expect(synced.model?.base).toBe('minimax-h3');
    // Steps carry over (the selection transition preserves user sampling); the
    // family-bound constraints snap.
    expect(synced).toMatchObject({ fps: 24, numFrames: 90, steps: 40, targetResolution: '768 highres' });
    expect(getVideoWidgetValidationReasons(synced, [h3Model()])).toEqual([]);
  });

  it('bootstraps the picked model’s family defaults (accelerator included) for a never-seeded store', () => {
    // A fresh project's widget store is `{}`; healing yields model-agnostic
    // fallbacks with NO modelKey. The selection transition must bootstrap the
    // picked model's own defaults instead of preserving the fallbacks —
    // otherwise a fresh H3 panel opens at 40 steps / CFG 5 with Turbo off.
    const catalog = [h3Model(), lora('MiniMax H3 Turbo LoRA', 'minimax-h3', null)];
    const healed = normalizeVideoWidgetValues({})!;
    const synced = syncVideoWidgetValuesWithModels(healed, catalog);

    expect(synced.model?.base).toBe('minimax-h3');
    // numFrames 124 is H3's own default — not the Wan fallback (81) snapped
    // onto the H3 grid (90), which is what the preserve path would produce.
    expect(synced).toMatchObject({ acceleratorEnabled: true, cfgScale: 1, fps: 24, numFrames: 124, steps: 6 });

    // A pre-open seeded payload ("Send to Video") keeps its media through the
    // same bootstrap.
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

    // The pair vanishes from the catalog: entries drop, flag clears with them,
    // and the sampling params Lightning wrote (steps 4 / CFG 1) are restored
    // to the model's own defaults — clearing only the flag would leave a
    // silent 4-step non-distilled setup.
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

    expect(getVideoWidgetValidationReasons(values)).toEqual(['Video needs a Wan 2.2 or MiniMax H3 main model.']);
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
    expect(resolveVideoSeed({ seed: 42, shouldRandomizeSeed: false })).toBe(42);

    const randomized = resolveVideoSeed({ seed: 42, shouldRandomizeSeed: true });

    expect(Number.isInteger(randomized)).toBe(true);
    expect(randomized).toBeGreaterThanOrEqual(0);
  });
});

import { describe, expect, it } from 'vitest';

import type { VideoSettings } from './types';

import {
  cloneVideoWidgetValues,
  isVideoSettings,
  isVideoSourceClip,
  normalizeVideoSettings,
  normalizeVideoWidgetValues,
  resolveVideoMode,
} from './settings';
import { getDefaultVideoSettings } from './videoPolicies';

const FIRST_FRAME = { height: 1080, image_name: 'first.png', width: 1920 };
const LAST_FRAME = { height: 1080, image_name: 'last.png', width: 1920 };
const SOURCE_VIDEO = {
  endFrame: -2,
  fps: 16,
  height: 480,
  numFrames: 81,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 832,
};

const createSettings = (overrides: Partial<VideoSettings> = {}): VideoSettings => ({
  ...getDefaultVideoSettings(),
  ...overrides,
});

describe('resolveVideoMode', () => {
  it('infers the mode from which inputs are filled', () => {
    expect(resolveVideoMode(createSettings())).toBe('txt2vid');
    expect(resolveVideoMode(createSettings({ firstFrameImage: FIRST_FRAME }))).toBe('first-frame');
    expect(resolveVideoMode(createSettings({ firstFrameImage: FIRST_FRAME, lastFrameImage: LAST_FRAME }))).toBe(
      'first-last'
    );
    expect(resolveVideoMode(createSettings({ lastFrameImage: LAST_FRAME }))).toBe('last-frame');
    expect(resolveVideoMode(createSettings({ sourceVideo: SOURCE_VIDEO }))).toBe('extend');
    // A last frame with a source video is still extend — it is the destination anchor.
    expect(resolveVideoMode(createSettings({ lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO }))).toBe('extend');
  });
});

describe('normalizeVideoSettings', () => {
  it('round-trips canonical settings', () => {
    const settings = createSettings({ firstFrameImage: FIRST_FRAME, positivePrompt: 'a cat' });
    const normalized = normalizeVideoSettings(settings);

    expect(normalized).toEqual(settings);
    expect(isVideoSettings(settings)).toBe(true);
  });

  it('rejects values missing the core shape', () => {
    expect(normalizeVideoSettings(null)).toBeNull();
    expect(normalizeVideoSettings({})).toBeNull();
    expect(normalizeVideoSettings({ ...createSettings(), numFrames: 'many' })).toBeNull();
    expect(normalizeVideoSettings({ ...createSettings(), positivePrompt: 7 })).toBeNull();
  });

  it('fills fields older persisted projects predate with defaults', () => {
    const legacy: Record<string, unknown> = {
      cfgScale: 5,
      fps: 16,
      modelKey: 'wan-key',
      negativePrompt: '',
      numFrames: 81,
      positivePrompt: 'a dog',
      seed: 123,
      shouldRandomizeSeed: false,
      steps: 40,
    };
    const normalized = normalizeVideoSettings(legacy);

    expect(normalized).not.toBeNull();
    expect(normalized?.aspectRatioId).toBe('16:9');
    expect(normalized?.targetResolution).toBe('720p');
    expect(normalized?.firstFrameImage).toBeNull();
    expect(normalized?.sourceVideo).toBeNull();
    expect(normalized?.loras).toEqual([]);
    expect(normalized?.acceleratorEnabled).toBe(false);
    expect(normalized?.positivePrompt).toBe('a dog');
  });

  it('drops malformed media values instead of failing wholesale', () => {
    const normalized = normalizeVideoSettings({
      ...createSettings(),
      firstFrameImage: { image_name: 'x.png' },
      sourceVideo: { video_name: 'clip.mp4' },
    });

    expect(normalized?.firstFrameImage).toBeNull();
    expect(normalized?.sourceVideo).toBeNull();
  });

  it('clears an accelerator flag whose recorded LoRAs are gone — the flag means they are active', () => {
    const lightningLora = {
      isEnabled: true,
      model: { base: 'wan', key: 'lit', name: 'Wan Lightning High Noise', type: 'lora' as const },
      weight: 1,
    };

    // Flag without recorded keys, or with a recorded key missing from the list, clears.
    expect(
      normalizeVideoSettings({ ...createSettings(), acceleratorEnabled: true, loras: [lightningLora] })
        ?.acceleratorEnabled
    ).toBe(false);
    expect(
      normalizeVideoSettings({
        ...createSettings(),
        acceleratorEnabled: true,
        acceleratorLoraKeys: ['lit', 'gone'],
        loras: [lightningLora],
      })
    ).toMatchObject({ acceleratorEnabled: false, acceleratorLoraKeys: [] });
    // Flag with all recorded keys present survives.
    expect(
      normalizeVideoSettings({
        ...createSettings(),
        acceleratorEnabled: true,
        acceleratorLoraKeys: ['lit'],
        loras: [lightningLora],
      })
    ).toMatchObject({ acceleratorEnabled: true, acceleratorLoraKeys: ['lit'] });
    expect(isVideoSettings({ ...createSettings(), acceleratorEnabled: true, acceleratorLoraKeys: [], loras: [] })).toBe(
      false
    );
    expect(
      isVideoSettings({
        ...createSettings(),
        acceleratorEnabled: true,
        acceleratorLoraKeys: ['lit'],
        loras: [lightningLora],
      })
    ).toBe(true);
    // A disabled flag must not carry stale keys.
    expect(isVideoSettings({ ...createSettings(), acceleratorEnabled: false, acceleratorLoraKeys: ['lit'] })).toBe(
      false
    );
  });

  it('resolves an illegal first-frame + source-video combination in favor of the first frame', () => {
    const normalized = normalizeVideoSettings({
      ...createSettings(),
      firstFrameImage: FIRST_FRAME,
      sourceVideo: SOURCE_VIDEO,
    });

    expect(normalized?.firstFrameImage).toEqual(FIRST_FRAME);
    expect(normalized?.sourceVideo).toBeNull();
  });
});

describe('isVideoSettings', () => {
  it('is strict over the keys normalize would invent', () => {
    expect(isVideoSettings({ ...createSettings(), aspectRatioId: 'Free' })).toBe(false);
    expect(isVideoSettings({ ...createSettings(), targetResolution: '4k' })).toBe(false);
    expect(isVideoSettings({ ...createSettings(), acceleratorEnabled: 'yes' })).toBe(false);
    expect(isVideoSettings({ ...createSettings(), firstFrameImage: FIRST_FRAME, sourceVideo: SOURCE_VIDEO })).toBe(
      false
    );
  });
});

describe('isVideoSourceClip', () => {
  it('requires the trim and probe fields', () => {
    expect(isVideoSourceClip(SOURCE_VIDEO)).toBe(true);
    expect(isVideoSourceClip({ ...SOURCE_VIDEO, fps: undefined })).toBe(false);
    expect(isVideoSourceClip({ ...SOURCE_VIDEO, video_name: 7 })).toBe(false);
  });
});

describe('normalizeVideoWidgetValues / cloneVideoWidgetValues', () => {
  const model = { base: 'wan', key: 'wan-key', name: 'Wan', type: 'main' as const, variant: 't2v_a14b' };

  it('carries a valid main model and nulls an invalid one', () => {
    expect(normalizeVideoWidgetValues({ ...createSettings(), model })?.model).toEqual(model);
    expect(normalizeVideoWidgetValues({ ...createSettings(), model: { key: 'x' } })?.model).toBeNull();
  });

  it('clones deeply enough that mutating the clone leaves the original untouched', () => {
    const values = { ...createSettings({ firstFrameImage: FIRST_FRAME, sourceVideo: null }), model };
    const clone = cloneVideoWidgetValues(values);

    expect(clone).toEqual(values);
    (clone.firstFrameImage as { image_name: string }).image_name = 'mutated.png';
    if (clone.model) {
      clone.model.key = 'mutated';
    }
    expect(values.firstFrameImage?.image_name).toBe('first.png');
    expect(values.model?.key).toBe('wan-key');
  });
});

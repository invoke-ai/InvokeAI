import { describe, expect, it } from 'vitest';

import type { VideoReferenceItem, VideoSettings, VideoWidgetValues } from './types';

import { MINIMAX_H3_NUM_FRAMES_CHOICES } from './dimensions';
import {
  clampReferenceSampleFrames,
  DEFAULT_REFERENCE_SAMPLE_FRAMES,
  referenceSampleFrames,
  resizeReferenceSampleWindow,
  slideReferenceSampleWindow,
  anchorReferenceConditioning,
  applyReferenceExtendSourceVideo,
  applyReferenceExtendNumFrames,
  canPlaceReferenceExtendAnchor,
  pinReferenceExtendAnchor,
  clearDeletedVideoMedia,
  cloneVideoWidgetValues,
  createVideoSourceClip,
  deriveReferenceExtendClip,
  formatReferencePromptLabels,
  getDefaultReferenceClip,
  getDefaultReferenceConditioning,
  getDefaultReferenceImageDetail,
  getInitialVideoPatch,
  getReferencesPatch,
  isVideoSettings,
  createVideoConditioningClip,
  isVideoSourceClip,
  normalizeVideoSettings,
  normalizeVideoWidgetValues,
  referencePromptLabels,
  resolveVideoMode,
  videoClipSpanSeconds,
  VIDEO_SOURCE_FALLBACK_FPS,
} from './settings';
import { getDefaultVideoSettings } from './videoPolicies';

const FIRST_FRAME = { height: 1080, image_name: 'first.png', width: 1920 };
const LAST_FRAME = { height: 1080, image_name: 'last.png', width: 1920 };
const SOURCE_VIDEO = {
  endFrame: 79,
  fps: 16,
  height: 480,
  numFrames: 81,
  startFrame: 0,
  video_name: 'clip.mp4',
  width: 832,
};

const CONDITIONING_CLIP = { fps: 24, height: 704, numFrames: 96, video_name: 'conditioning.mp4', width: 1248 };

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

  it("reads the conditioning clip's role as the mode, ahead of every frame and clip slot", () => {
    expect(
      resolveVideoMode(createSettings({ conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' } }))
    ).toBe('audio-to-video');
    expect(
      resolveVideoMode(createSettings({ conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'video' } }))
    ).toBe('video-to-audio');
    // Not reachable through the setters, but a recalled or stored record can hold both; the mode
    // has to name the one that would actually run, which validation then refuses.
    expect(
      resolveVideoMode(
        createSettings({
          conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' },
          sourceVideo: SOURCE_VIDEO,
        })
      )
    ).toBe('audio-to-video');
  });
});

describe('createVideoConditioningClip', () => {
  it('takes an uploaded soundtrack for its audio and anything else for its picture', () => {
    const item = { durationSeconds: 4, fps: 24, height: 704, name: 'clip.mp4', width: 1248 };

    // An `audio_upload` record is a bare soundtrack wrapped as a video: it has no picture to
    // condition on, so the only role it can fill is the audio one.
    expect(createVideoConditioningClip({ ...item, mediaOrigin: 'audio_upload' }).role).toBe('audio');
    expect(createVideoConditioningClip(item).role).toBe('video');
  });

  it('carries no trim bounds, which the conditioning nodes would not honour', () => {
    const clip = createVideoConditioningClip({ durationSeconds: 4, fps: 24, height: 704, name: 'c.mp4', width: 1248 });

    expect(clip.clip).not.toHaveProperty('startFrame');
    expect(clip.clip).not.toHaveProperty('endFrame');
    expect(clip.clip).toEqual({ fps: 24, height: 704, numFrames: 96, video_name: 'c.mp4', width: 1248 });
  });
});

describe('normalizeVideoSettings', () => {
  it('reads the seed mode saved before modes existed from the random toggle', () => {
    const legacy = { ...createSettings({}), seedMode: undefined };

    expect(normalizeVideoSettings({ ...legacy, shouldRandomizeSeed: false })?.seedMode).toBe('fixed');
    expect(normalizeVideoSettings({ ...legacy, shouldRandomizeSeed: true })?.seedMode).toBe('random');
    expect(normalizeVideoSettings({ ...legacy, seedMode: 'increment', shouldRandomizeSeed: false })?.seedMode).toBe(
      'increment'
    );
    expect(normalizeVideoSettings(legacy)?.seedMode).toBe('random');
    // A record whose mode normalize has to invent is not canonical, so nothing may reuse it raw.
    expect(isVideoSettings({ ...legacy, shouldRandomizeSeed: true })).toBe(false);
    expect(isVideoSettings(legacy)).toBe(false);
  });

  it('round-trips canonical settings', () => {
    const settings = createSettings({ firstFrameImage: FIRST_FRAME, positivePrompt: 'a cat' });
    const normalized = normalizeVideoSettings(settings);

    expect(normalized).toEqual(settings);
    expect(isVideoSettings(settings)).toBe(true);
  });

  it('folds a low-noise CFG below 1 back into reuse-primary', () => {
    expect(normalizeVideoSettings({ ...createSettings(), cfgScaleLowNoise: 0.5 })?.cfgScaleLowNoise).toBeNull();
    expect(normalizeVideoSettings({ ...createSettings(), cfgScaleLowNoise: 1 })?.cfgScaleLowNoise).toBe(1);
  });

  it('does not call a record with a garbage conditioning clip valid settings', () => {
    // The guard's whole job is deciding whether a stored record can be used as-is; a slot it does
    // not look at is a slot that reaches the graph builder unchecked.
    const clip = { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' as const };

    expect(isVideoSettings(createSettings({ conditioningClip: clip }))).toBe(true);
    expect(isVideoSettings({ ...createSettings(), conditioningClip: { clip: CONDITIONING_CLIP } })).toBe(false);
    expect(isVideoSettings({ ...createSettings(), conditioningClip: 'nonsense' })).toBe(false);
  });

  it('rejects non-records but heals partial records field-by-field, upscale-style', () => {
    expect(normalizeVideoSettings(null)).toBeNull();
    expect(normalizeVideoSettings(7)).toBeNull();
    // Preserve seeded partial Send to Video payloads through normalization and reconciliation.
    const seeded = normalizeVideoSettings({ firstFrameImage: FIRST_FRAME, sourceVideo: null });

    expect(seeded).not.toBeNull();
    expect(seeded?.firstFrameImage).toEqual(FIRST_FRAME);
    expect(seeded).toMatchObject({ fps: 16, modelKey: '', numFrames: 81, steps: 40, targetResolution: '720p' });
    // Invalid field types heal to defaults rather than failing wholesale.
    expect(normalizeVideoSettings({ ...createSettings(), numFrames: 'many' })?.numFrames).toBe(81);
    expect(normalizeVideoSettings({ ...createSettings(), positivePrompt: 7 })?.positivePrompt).toBe('');
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
      seedMode: 'fixed',
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

  it('heals the hybrid start block to the recommended default and clamps it to the block range', () => {
    const { h3HybridStartBlock: _predatesHybrid, ...legacy } = createSettings();

    expect(normalizeVideoSettings(legacy)?.h3HybridStartBlock).toBe(25);
    expect(normalizeVideoSettings({ ...legacy, h3HybridStartBlock: 120 })?.h3HybridStartBlock).toBe(49);
    expect(normalizeVideoSettings({ ...legacy, h3HybridStartBlock: -3 })?.h3HybridStartBlock).toBe(0);
    expect(normalizeVideoSettings({ ...legacy, h3HybridStartBlock: 12.6 })?.h3HybridStartBlock).toBe(13);
    expect(
      normalizeVideoSettings({ ...legacy, h3HybridBaseModel: { key: 'not-a-main' } })?.h3HybridBaseModel
    ).toBeNull();
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
    // Disabled accelerator LoRAs also clear intent because graph compilation omits them.
    const disabledLightning = { ...lightningLora, isEnabled: false };

    expect(
      normalizeVideoSettings({
        ...createSettings(),
        acceleratorEnabled: true,
        acceleratorLoraKeys: ['lit'],
        loras: [disabledLightning],
      })
    ).toMatchObject({ acceleratorEnabled: false, acceleratorLoraKeys: [] });

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

describe('videoClipSpanSeconds', () => {
  it('runs to the far edge of the last selected frame', () => {
    // 16 fps, frames 0..79 inclusive: stopping at 79/16 would cut the final frame short.
    expect(videoClipSpanSeconds(SOURCE_VIDEO)).toEqual({ endSeconds: 5, startSeconds: 0 });
    expect(videoClipSpanSeconds({ ...SOURCE_VIDEO, endFrame: 47, startFrame: 32 })).toEqual({
      endSeconds: 3,
      startSeconds: 2,
    });
  });

  it('has no span to offer for a clip with no usable frame rate', () => {
    // Finite persisted fps can still be zero; playback must reject unusable rates.
    expect(videoClipSpanSeconds({ ...SOURCE_VIDEO, fps: 0 })).toBeNull();
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

describe('createVideoSourceClip', () => {
  it('estimates frames from duration and defaults the trim to drop the final frame', () => {
    const clip = createVideoSourceClip({ durationSeconds: 5, fps: 16, height: 480, name: 'clip.mp4', width: 832 });

    expect(clip).toEqual({
      endFrame: 78,
      fps: 16,
      height: 480,
      numFrames: 80,
      startFrame: 0,
      video_name: 'clip.mp4',
      width: 832,
    });
  });

  it('falls back to 16 fps when the probe recorded none, mirroring extract_video_range', () => {
    const clip = createVideoSourceClip({ durationSeconds: 2, height: 480, name: 'clip.mp4', width: 832 });

    expect(clip.fps).toBe(VIDEO_SOURCE_FALLBACK_FPS);
    expect(clip.numFrames).toBe(32);
  });

  it('keeps the default end frame at 1 or above so the crossfade tail survives', () => {
    const clip = createVideoSourceClip({ durationSeconds: 0.1, fps: 16, height: 480, name: 'c.mp4', width: 832 });

    expect(clip.endFrame).toBeGreaterThanOrEqual(1);
  });

  it('never produces negative trim bounds for very short clips', () => {
    const clip = createVideoSourceClip({ durationSeconds: 0.05, fps: 16, height: 480, name: 'c.mp4', width: 832 });

    expect(clip.numFrames).toBeGreaterThanOrEqual(1);
    expect(clip.endFrame).toBeGreaterThanOrEqual(0);
    expect(clip.startFrame).toBe(0);
  });
});

describe('getDefaultReferenceConditioning', () => {
  it('starts a wrapped audio upload on its soundtrack alone', () => {
    expect(getDefaultReferenceConditioning('audio_upload')).toBe('audio');
  });

  it('keeps video + audio for ordinary videos', () => {
    expect(getDefaultReferenceConditioning('some_other_origin')).toBe('video_audio');
  });

  it('keeps video + audio when the video carries no marker', () => {
    expect(getDefaultReferenceConditioning(null)).toBe('video_audio');
    expect(getDefaultReferenceConditioning(undefined)).toBe('video_audio');
  });
});

describe('getDefaultReferenceClip', () => {
  // Use a narrow offset fixture so audio defaulting must actively expand to the whole wrapped soundtrack.
  const wrappedAudio = {
    endFrame: 199,
    fps: 24,
    height: 512,
    numFrames: 4320,
    startFrame: 120,
    video_name: 'song.mp4',
    width: 512,
  };

  // Long enough that the sample window actually bites: SOURCE_VIDEO's 81 frames are
  // shorter than DEFAULT_REFERENCE_SAMPLE_FRAMES, so it can only exercise the clamp.
  const footage = { ...SOURCE_VIDEO, endFrame: 599, numFrames: 600 };

  it('samples footage from the head of the clip', () => {
    expect(getDefaultReferenceClip(footage, 'video_audio')).toEqual({
      ...footage,
      endFrame: DEFAULT_REFERENCE_SAMPLE_FRAMES - 1,
      startFrame: 0,
    });
    expect(getDefaultReferenceClip(footage, 'video')).toEqual({
      ...footage,
      endFrame: DEFAULT_REFERENCE_SAMPLE_FRAMES - 1,
      startFrame: 0,
    });
  });

  it('gives the same long clip its whole length once it is audio-only', () => {
    expect(getDefaultReferenceClip(footage, 'audio').endFrame).toBe(599);
  });

  it('gives an audio-only reference the whole clip', () => {
    // Audio references need the full soundtrack window; they add no visual rows and must not default to eight
    // seconds.
    expect(getDefaultReferenceClip(wrappedAudio, 'audio')).toEqual({
      ...wrappedAudio,
      endFrame: 4319,
      startFrame: 0,
    });
  });

  it('still samples the head of that same clip when it carries video', () => {
    expect(getDefaultReferenceClip(wrappedAudio, 'video_audio')).toEqual({
      ...wrappedAudio,
      endFrame: DEFAULT_REFERENCE_SAMPLE_FRAMES - 1,
      startFrame: 0,
    });
  });

  it('never samples past the end of a clip shorter than the window', () => {
    // SOURCE_VIDEO is 81 frames; both conditionings collapse to the whole clip.
    expect(getDefaultReferenceClip(SOURCE_VIDEO, 'video_audio').endFrame).toBe(80);
    expect(getDefaultReferenceClip(SOURCE_VIDEO, 'audio').endFrame).toBe(80);

    const short = { ...SOURCE_VIDEO, endFrame: 11, numFrames: 12 };

    expect(getDefaultReferenceClip(short, 'video_audio').endFrame).toBe(11);
    expect(getDefaultReferenceClip(short, 'audio').endFrame).toBe(11);
  });

  it('keeps the shortest real clip on its single frame', () => {
    // createVideoSourceClip floors numFrames at 1, so this is the smallest clip either
    // branch can be handed; both must land on frame 0, not on -1.
    const single = { ...SOURCE_VIDEO, endFrame: 0, numFrames: 1 };

    expect(getDefaultReferenceClip(single, 'video_audio').endFrame).toBe(0);
    expect(getDefaultReferenceClip(single, 'audio').endFrame).toBe(0);
  });

  it('resets the start frame, so a trimmed source clip does not carry its trim in', () => {
    const trimmed = { ...SOURCE_VIDEO, endFrame: 70, startFrame: 40 };

    expect(getDefaultReferenceClip(trimmed, 'video_audio').startFrame).toBe(0);
    expect(getDefaultReferenceClip(trimmed, 'audio').startFrame).toBe(0);
  });
});

describe('getDefaultReferenceImageDetail', () => {
  const imageReference = {
    detail: 'max',
    image: { height: 1080, image_name: 'ref.png', width: 1920 },
    kind: 'image',
  } as const;
  const videoReference = {
    clip: SOURCE_VIDEO,
    conditioning: 'video_audio',
    kind: 'video',
  } as const;

  it('starts the first image reference at maximum detail', () => {
    expect(getDefaultReferenceImageDetail([])).toBe('max');
    expect(getDefaultReferenceImageDetail([videoReference])).toBe('max');
  });

  it('matches the generation size once an image reference is placed', () => {
    expect(getDefaultReferenceImageDetail([imageReference])).toBe('match');
    expect(getDefaultReferenceImageDetail([videoReference, imageReference])).toBe('match');
  });
});

describe('normalizeVideoSettings — the conditioning clip', () => {
  const clip = { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' as const };

  it('drops a clip stored beside a slot that claims the same conditioning mask', () => {
    // A rolled-back or hand-edited project can hold both. Keeping them would resolve to a mode
    // whose graph silently ignores one of the two.
    expect(
      normalizeVideoSettings({ ...createSettings({ firstFrameImage: FIRST_FRAME }), conditioningClip: clip })
        ?.conditioningClip
    ).toBeNull();
    expect(
      normalizeVideoSettings({ ...createSettings({ sourceVideo: SOURCE_VIDEO }), conditioningClip: clip })
        ?.conditioningClip
    ).toBeNull();
    expect(normalizeVideoSettings(createSettings({ conditioningClip: clip }))?.conditioningClip).toEqual(clip);
  });

  it('drops a malformed clip rather than passing it to the graph builder', () => {
    for (const malformed of [
      { clip: CONDITIONING_CLIP },
      { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'soundtrack' },
      { clip: { video_name: 'c.mp4' }, fpsKnown: true, role: 'audio' },
      { clip: CONDITIONING_CLIP, fpsKnown: 'yes', role: 'audio' },
      'nonsense',
    ]) {
      expect(
        normalizeVideoSettings({ ...createSettings(), conditioningClip: malformed } as unknown as VideoSettings)
          ?.conditioningClip,
        JSON.stringify(malformed)
      ).toBeNull();
    }
  });

  it('deep-copies the clip so a clone cannot alias the original', () => {
    const cloned = cloneVideoWidgetValues(createSettings({ conditioningClip: clip }) as VideoWidgetValues);

    expect(cloned.conditioningClip).toEqual(clip);
    expect(cloned.conditioningClip?.clip).not.toBe(clip.clip);
  });
});

describe('clearDeletedVideoMedia', () => {
  const withMedia = createSettings({
    firstFrameImage: FIRST_FRAME,
    lastFrameImage: LAST_FRAME,
    sourceVideo: null,
  });

  it('returns the same object when nothing referenced was deleted', () => {
    expect(clearDeletedVideoMedia(withMedia, new Set(['other.png']), new Set())).toBe(withMedia);
  });

  it('clears exactly the deleted references', () => {
    const cleared = clearDeletedVideoMedia(withMedia, new Set(['first.png']), new Set());

    expect(cleared.firstFrameImage).toBeNull();
    expect(cleared.lastFrameImage).toEqual(LAST_FRAME);

    const withClip = createSettings({ sourceVideo: SOURCE_VIDEO });
    const clipCleared = clearDeletedVideoMedia(withClip, new Set(), new Set(['clip.mp4']));

    expect(clipCleared.sourceVideo).toBeNull();

    // A conditioning clip is a gallery video too: left behind, it would compile a graph naming
    // a deleted record and fail at the conditioning node rather than in the panel.
    const withConditioning = createSettings({
      conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' },
    });

    expect(
      clearDeletedVideoMedia(withConditioning, new Set(), new Set(['conditioning.mp4'])).conditioningClip
    ).toBeNull();
    expect(clearDeletedVideoMedia(withConditioning, new Set(), new Set(['other.mp4']))).toBe(withConditioning);
  });

  it('clears a reference the exclusion masking would hide from a normalized snapshot', () => {
    // Sweep raw slots before normalization can hide conflicting media references from deletion.
    const rawBoth = { ...createSettings({ firstFrameImage: FIRST_FRAME }), sourceVideo: SOURCE_VIDEO } as Record<
      string,
      unknown
    >;
    const cleared = clearDeletedVideoMedia(rawBoth, new Set(), new Set(['clip.mp4']));

    expect(cleared.sourceVideo).toBeNull();
    expect(cleared.firstFrameImage).toEqual(FIRST_FRAME);

    // Junk in a slot never throws — the guards ignore non-media shapes.
    const junk = { firstFrameImage: 'nonsense', lastFrameImage: 7, sourceVideo: {} } as Record<string, unknown>;

    expect(clearDeletedVideoMedia(junk, new Set(['nonsense']), new Set())).toBe(junk);
  });
});

const IMAGE_REFERENCE = {
  detail: 'max',
  image: { height: 512, image_name: 'ref.png', width: 512 },
  kind: 'image',
} as const;
const VIDEO_REFERENCE = {
  clip: { endFrame: 47, fps: 24, height: 480, numFrames: 48, startFrame: 0, video_name: 'ref.mp4', width: 832 },
  conditioning: 'video_audio',
  kind: 'video',
} as const;

describe('references', () => {
  it('reference mode wins the mode inference', () => {
    expect(resolveVideoMode(createSettings({ references: [IMAGE_REFERENCE] }))).toBe('reference');
    expect(resolveVideoMode(createSettings({ references: [VIDEO_REFERENCE], sourceVideo: SOURCE_VIDEO }))).toBe(
      'reference'
    );
  });

  it('normalization drops frame media when references are present, keeping the source video', () => {
    const normalized = normalizeVideoSettings(
      createSettings({ firstFrameImage: FIRST_FRAME, references: [IMAGE_REFERENCE], sourceVideo: SOURCE_VIDEO })
    );

    expect(normalized?.references).toEqual([IMAGE_REFERENCE]);
    expect(normalized?.firstFrameImage).toBeNull();
    // Source plus references is valid only for models supporting reference extension.
    expect(normalized?.sourceVideo).toEqual(SOURCE_VIDEO);
  });

  it('normalization drops malformed entries and enforces the caps, preserving order', () => {
    const tooMany = [
      ...Array.from({ length: 4 }, (_, index) => ({
        ...VIDEO_REFERENCE,
        clip: { ...VIDEO_REFERENCE.clip, video_name: `v${index}.mp4` },
      })),
      { kind: 'image' },
      IMAGE_REFERENCE,
    ];
    const normalized = normalizeVideoSettings(createSettings({ references: tooMany as never }));

    expect(normalized?.references.map((entry) => (entry.kind === 'video' ? entry.clip.video_name : 'img'))).toEqual([
      'v0.mp4',
      'v1.mp4',
      'v2.mp4',
      'img',
    ]);
  });

  it('over-cap overflow drops the newest videos but never the anchor', () => {
    const named = (entry: { kind: string; clip?: { video_name: string } }) =>
      entry.kind === 'video' ? entry.clip!.video_name : 'img';
    const tooMany = [
      IMAGE_REFERENCE,
      ...Array.from({ length: 5 }, (_unused, index) => ({
        ...VIDEO_REFERENCE,
        clip: { ...VIDEO_REFERENCE.clip, video_name: `v${index}.mp4` },
      })),
    ];
    const normalized = normalizeVideoSettings(createSettings({ references: tooMany as never }));

    expect(normalized?.references.map(named as never)).toEqual(['img', 'v0.mp4', 'v1.mp4', 'v2.mp4']);

    // Discard the racing newest addition rather than the user's oldest reference when anchor insertion exceeds the
    // cap.
    const anchor = {
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name: 'anchor.mp4' },
      fromSourceVideo: true,
    };
    const raced = ['b.mp4', 'c.mp4'].map((video_name) => ({
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name },
    }));
    const healed = normalizeVideoSettings(
      createSettings({
        references: [
          ...raced,
          anchor,
          { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'd.mp4' } },
        ] as never,
      })
    );

    expect(healed?.references.map(named as never)).toEqual(['b.mp4', 'c.mp4', 'anchor.mp4']);
    expect(healed?.references.at(-1)).toMatchObject({ fromSourceVideo: true });
  });

  it('normalization heals a panel saved with the anchor prepended', () => {
    // Heal legacy prepended anchors by pinning last before overflow trimming.
    const anchor = {
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name: 'anchor.mp4' },
      fromSourceVideo: true,
    };
    const named = (entry: { kind: string; clip?: { video_name: string } }) =>
      entry.kind === 'video' ? entry.clip!.video_name : 'img';

    const healed = normalizeVideoSettings(createSettings({ references: [anchor, IMAGE_REFERENCE] as never }));

    expect(healed?.references.map(named as never)).toEqual(['img', 'anchor.mp4']);

    // Over the cap, with the anchor in the position the old build left it.
    const stale = [
      anchor,
      ...Array.from({ length: 3 }, (_unused, index) => ({
        ...VIDEO_REFERENCE,
        clip: { ...VIDEO_REFERENCE.clip, video_name: `v${index}.mp4` },
      })),
    ];
    const trimmed = normalizeVideoSettings(createSettings({ references: stale as never }));

    expect(trimmed?.references.map(named as never)).toEqual(['v0.mp4', 'v1.mp4', 'anchor.mp4']);
    expect(trimmed?.references.at(-1)).toMatchObject({ fromSourceVideo: true });
  });

  it('isVideoSettings rejects references combined with frame media', () => {
    expect(isVideoSettings(createSettings({ firstFrameImage: FIRST_FRAME, references: [IMAGE_REFERENCE] }))).toBe(
      false
    );
    expect(isVideoSettings(createSettings({ references: [IMAGE_REFERENCE] }))).toBe(true);
  });

  it('clone deep-copies references', () => {
    const values = { ...createSettings({ references: [VIDEO_REFERENCE] }), model: null };
    const clone = cloneVideoWidgetValues(values);

    expect(clone.references).toEqual(values.references);
    expect(clone.references[0]).not.toBe(values.references[0]);
  });

  it('clearDeletedVideoMedia filters deleted reference media, preserving order and identity', () => {
    const values = createSettings({ references: [VIDEO_REFERENCE, IMAGE_REFERENCE] });
    const untouched = clearDeletedVideoMedia(values, new Set(), new Set());

    expect(untouched).toBe(values);

    const swept = clearDeletedVideoMedia(values, new Set(['ref.png']), new Set());

    expect(swept.references).toEqual([VIDEO_REFERENCE]);

    const sweptVideo = clearDeletedVideoMedia(values, new Set(), new Set(['ref.mp4']));

    expect(sweptVideo.references).toEqual([IMAGE_REFERENCE]);
  });
});

describe('anchorReferenceConditioning', () => {
  it('converts an audio-only choice, which cannot carry a seam', () => {
    expect(anchorReferenceConditioning('audio')).toBe('video_audio');
  });

  it("leaves the user's own visual answers alone", () => {
    expect(anchorReferenceConditioning('video')).toBe('video');
    expect(anchorReferenceConditioning('video_audio')).toBe('video_audio');
  });
});

describe('referencePromptLabels', () => {
  const withConditioning = (conditioning: 'video_audio' | 'video' | 'audio'): VideoReferenceItem => ({
    ...VIDEO_REFERENCE,
    conditioning,
  });

  // Mirror independent modality counters in backend attachment order; labels are not card positions.
  it('numbers each modality on its own counter', () => {
    expect(
      referencePromptLabels([
        withConditioning('audio'),
        IMAGE_REFERENCE,
        withConditioning('video_audio'),
        IMAGE_REFERENCE,
        withConditioning('video'),
      ]).map(formatReferencePromptLabels)
    ).toEqual([['<Audio 1>'], ['<Picture 1>'], ['<Video 1>', '<Audio 2>'], ['<Picture 2>'], ['<Video 2>']]);
  });

  it('gives an audio-only reference no video number, which the next video then takes', () => {
    expect(referencePromptLabels([withConditioning('audio'), withConditioning('video')])).toEqual([
      { audio: 1, picture: null, video: null },
      { audio: null, picture: null, video: 1 },
    ]);
  });

  it('renumbers on reorder, since order is what the numbering follows', () => {
    const first = { ...IMAGE_REFERENCE, image: { ...IMAGE_REFERENCE.image, image_name: 'first.png' } };
    const second = { ...IMAGE_REFERENCE, image: { ...IMAGE_REFERENCE.image, image_name: 'second.png' } };
    const numbered = (references: VideoReferenceItem[]) =>
      Object.fromEntries(
        references.map((reference, index) => [
          reference.kind === 'image' ? reference.image.image_name : reference.clip.video_name,
          referencePromptLabels(references)[index],
        ])
      );

    // Reordering images updates picture numbering without changing the video's modality counter.
    expect(numbered([first, withConditioning('video_audio'), second])).toEqual({
      'first.png': { audio: null, picture: 1, video: null },
      'ref.mp4': { audio: 1, picture: null, video: 1 },
      'second.png': { audio: null, picture: 2, video: null },
    });
    expect(numbered([second, withConditioning('video_audio'), first])).toEqual({
      'first.png': { audio: null, picture: 2, video: null },
      'ref.mp4': { audio: 1, picture: null, video: 1 },
      'second.png': { audio: null, picture: 1, video: null },
    });
  });

  it('numbers a full list, where the widest label lives', () => {
    const images = Array.from({ length: 9 }, (_unused, index) => ({
      ...IMAGE_REFERENCE,
      image: { ...IMAGE_REFERENCE.image, image_name: `image-${index}.png` },
    }));

    expect(
      referencePromptLabels([
        ...images,
        withConditioning('video'),
        withConditioning('video_audio'),
        withConditioning('audio'),
      ]).map(formatReferencePromptLabels)
    ).toEqual([
      ...images.map((_unused, index) => [`<Picture ${index + 1}>`]),
      ['<Video 1>'],
      ['<Video 2>', '<Audio 1>'],
      ['<Audio 2>'],
    ]);
  });
});

describe('reference-extend anchor: audio-only references', () => {
  const longSource = { ...SOURCE_VIDEO, endFrame: 400, numFrames: 402, video_name: 'long.mp4' };
  const source24 = { ...longSource, fps: 24 };
  const FRAMES = 141;
  // The user's own soundtrack reference to the clip they are extending: whole clip, audio.
  const userAudio = {
    clip: { ...SOURCE_VIDEO, endFrame: 401, numFrames: 402, startFrame: 0, video_name: 'long.mp4' },
    conditioning: 'audio',
    kind: 'video',
  } as const;

  it('appends a real anchor beside an audio-only reference rather than consuming it', () => {
    // Keep the user's audio reference and append a visual anchor; adopting audio would overwrite its trim and
    // provide no seam visuals.
    const linked = applyReferenceExtendSourceVideo([userAudio], source24, 3, FRAMES);

    expect(linked).toHaveLength(2);
    expect(linked[0]).toBe(userAudio);
    expect(linked[1]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260 },
      conditioning: 'video_audio',
      fromSourceVideo: true,
    });
  });

  it('adopts a same-clip reference that CAN carry the seam, in preference to the audio one', () => {
    const userVideo = {
      ...userAudio,
      clip: { ...userAudio.clip, endFrame: 172, startFrame: 100 },
      conditioning: 'video',
    } as const;
    const linked = applyReferenceExtendSourceVideo([userAudio, userVideo], source24, 3, FRAMES);

    // Preserve audio identity while the visual reference becomes the anchor.
    expect(linked).toHaveLength(2);
    expect(linked.find((entry) => entry === userAudio)).toBe(userAudio);
    expect(linked[linked.length - 1]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260 },
      conditioning: 'video',
      fromSourceVideo: true,
    });
  });

  it('converts an already-flagged audio anchor, re-deriving its window in the same pass', () => {
    // Heal legacy anchors' conditioning and tail window together, outside normalization.
    const stale = {
      ...userAudio,
      clip: { ...userAudio.clip, endFrame: 72, startFrame: 0 },
      fromSourceVideo: true,
    } as const;
    const linked = applyReferenceExtendSourceVideo([stale], source24, 3, FRAMES);

    expect(linked[0]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260 },
      conditioning: 'video_audio',
      fromSourceVideo: true,
    });
  });

  it('converts an overridden audio anchor without disturbing the window the user picked', () => {
    // Promote anchor conditioning to include visuals while preserving a deliberate trim override.
    const overridden = {
      ...userAudio,
      clip: { ...userAudio.clip, endFrame: 219, startFrame: 100 },
      fromSourceVideo: true,
      trimOverridden: true,
    } as const;
    const linked = applyReferenceExtendSourceVideo([overridden], { ...source24, endFrame: 300 }, 3, FRAMES);

    expect(linked[0]).toMatchObject({
      clip: { endFrame: 219, startFrame: 100 },
      conditioning: 'video_audio',
      fromSourceVideo: true,
      trimOverridden: true,
    });
  });

  it('normalization never flags an audio-only reference as the anchor', () => {
    // Recall must not adopt audio-only same-name references as visual tail anchors.
    const normalized = normalizeVideoSettings(
      createSettings({ references: [IMAGE_REFERENCE, userAudio], sourceVideo: source24 })
    );

    expect(normalized?.references).toHaveLength(2);
    expect(normalized?.references.some((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)).toBe(
      false
    );
    expect(normalized?.references.find((entry) => entry.kind === 'video')).toMatchObject({
      clip: { endFrame: 401, startFrame: 0 },
      conditioning: 'audio',
    });
  });

  it('still re-derives the flag onto a same-clip reference that can carry the seam', () => {
    // The recall re-derive itself is intact -- this is the case it exists for.
    const recalled = { ...userAudio, conditioning: 'video_audio' as const };
    const normalized = normalizeVideoSettings(
      createSettings({ references: [IMAGE_REFERENCE, recalled], sourceVideo: source24 })
    );

    expect(normalized?.references[1]).toMatchObject({ conditioning: 'video_audio', fromSourceVideo: true });
  });

  it('leaves an audio-only reference beside a real anchor completely alone', () => {
    // An existing anchor leaves unrelated audio conditioning, trim, and identity unchanged.
    const realAnchor = { ...userAudio, conditioning: 'video_audio' as const, fromSourceVideo: true } as const;
    const normalized = normalizeVideoSettings(
      createSettings({ references: [userAudio, realAnchor], sourceVideo: source24 })
    );

    expect(normalized?.references[0]).toMatchObject({ conditioning: 'audio', clip: { endFrame: 401, startFrame: 0 } });
    expect(normalized?.references[0]).not.toHaveProperty('fromSourceVideo', true);
  });
});

describe('reference-extend linkage', () => {
  // H3 generates at a fixed 24 fps; SOURCE_VIDEO runs at 16, so the two rates
  // are exercised separately.
  const longSource = { ...SOURCE_VIDEO, endFrame: 400, numFrames: 402, video_name: 'long.mp4' };
  const source24 = { ...longSource, fps: 24 };
  // The panel's default; every choice is on the 17n+5 grid.
  const FRAMES = 141;

  it('anchors on video + audio -- the role needs visual rows', () => {
    const [ordinary] = applyReferenceExtendSourceVideo([], source24, 3, FRAMES);

    // Anchors require visual conditioning regardless of upload origin; mixed references would otherwise bypass
    // all-audio validation with a broken seam.
    expect(ordinary).toMatchObject({ conditioning: 'video_audio', fromSourceVideo: true });
  });

  it('derives the tail trim: the window ending at the cutpoint, clamped at 0', () => {
    expect(deriveReferenceExtendClip(source24, FRAMES)).toMatchObject({ endFrame: 400, startFrame: 260 });
    // Use the largest supported grid window ending at the cutpoint when the source is shorter than the target
    // tail.
    expect(deriveReferenceExtendClip({ ...SOURCE_VIDEO, fps: 24 }, FRAMES)).toMatchObject({
      endFrame: 79,
      startFrame: 7,
    });
  });

  it('budgets the window against the generated frame count', () => {
    // Limit reference length to generated frames because backend truncation preserves the front and would discard
    // seam frames.
    expect(deriveReferenceExtendClip(source24, 124)).toMatchObject({ endFrame: 400, startFrame: 277 });
    expect(deriveReferenceExtendClip(source24, 90)).toMatchObject({ endFrame: 400, startFrame: 311 });
    // Above the tail window the budget stops binding: ~5s of lead-in is the cap.
    expect(deriveReferenceExtendClip(source24, 345)).toMatchObject({ endFrame: 400, startFrame: 260 });
  });

  it('lands the window ON the 17n+5 grid the backend keeps, at every source rate', () => {
    // Validate resampling, generated-length truncation, and 17n+5 snap-down together; floor rounding can lose a
    // full grid interval at fractional fps.
    const resample = (n: number, fps: number) => Math.floor((n * 24) / fps + 0.5);
    const snapDown = (n: number) => Math.max(1, Math.floor((n - 5) / 17)) * 17 + 5;

    for (const fps of [10, 12, 15, 16, 18, 20, 23.976, 24, 25, 29.97, 30, 60]) {
      for (const numFrames of MINIMAX_H3_NUM_FRAMES_CHOICES) {
        const clip = deriveReferenceExtendClip({ ...longSource, fps }, numFrames);
        const budget = Math.min(141, numFrames);
        const kept = snapDown(Math.min(resample(clip.endFrame - clip.startFrame + 1, fps), numFrames));

        expect({ fps, kept, numFrames }).toEqual({ fps, kept: budget, numFrames });
      }
    }
  });

  it('a clip SHORTER than the window still ends its tail on the cutpoint', () => {
    // Short clips also need grid-aligned budgets; arbitrary whole-clip windows lose their tail during snap-down.
    const resample = (n: number, fps: number) => Math.floor((n * 24) / fps + 0.5);
    const snapDown = (n: number) => Math.max(1, Math.floor((n - 5) / 17)) * 17 + 5;

    for (const fps of [12, 16, 23.976, 24, 25, 30]) {
      for (const total of [40, 60, 80, 100, 120, 141, 200]) {
        for (const numFrames of [90, 124, 141, 345]) {
          const source = { ...longSource, endFrame: total - 1, fps, numFrames: total };
          const clip = deriveReferenceExtendClip(source, numFrames);
          const window = clip.endFrame - clip.startFrame + 1;
          const kept = snapDown(Math.min(resample(window, fps), numFrames));

          const discarded = Math.min(resample(window, fps), numFrames) - kept;

          // Allow at most one seam frame lost to source-rate granularity, not a whole grid interval.
          expect({ discarded: discarded <= 1, fps, numFrames, total }).toEqual({
            discarded: true,
            fps,
            numFrames,
            total,
          });
          // And the window always still ends where it was asked to.
          expect(clip.endFrame).toBe(source.endFrame);
        }
      }
    }
  });

  it('converts the window into the source clip fps', () => {
    // 141 frames of 24 fps material is 94 frames of a 16 fps clip — the same
    // 5.875s of wall time, which is what the tail window actually means.
    expect(deriveReferenceExtendClip(longSource, FRAMES)).toMatchObject({ endFrame: 400, startFrame: 307 });
    // Missing source rate must fall back without collapsing the sample to the two-frame floor.
    expect(deriveReferenceExtendClip({ ...longSource, fps: 0 }, FRAMES)).toMatchObject({ startFrame: 260 });
    expect(deriveReferenceExtendClip({ ...longSource, fps: -30 }, FRAMES)).toMatchObject({ startFrame: 260 });
  });

  it('appends a linked video+audio reference and re-derives it on cutpoint changes', () => {
    // Append the continuity anchor last because generated rotary positions continue from the final reference
    // block.
    const added = applyReferenceExtendSourceVideo([IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(added).toHaveLength(2);
    expect(added[0]).toBe(IMAGE_REFERENCE);
    expect(added[1]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260, video_name: 'long.mp4' },
      conditioning: 'video_audio',
      fromSourceVideo: true,
      kind: 'video',
    });

    const tuned = added.map((entry, index) =>
      index === 1 && entry.kind === 'video' ? { ...entry, conditioning: 'video' as const } : entry
    );
    const retrimmed = applyReferenceExtendSourceVideo(tuned, { ...source24, endFrame: 300 }, 3, FRAMES);

    expect(retrimmed[1]).toMatchObject({
      clip: { endFrame: 300, startFrame: 160 },
      conditioning: 'video',
      fromSourceVideo: true,
    });
    expect(retrimmed[0]).toBe(added[0]);
  });

  it('leaves an overridden anchor window alone across cutpoint and frame-count changes', () => {
    // Allow sampling before a fade while concatenating through it; Ref2VA provides no frame-exact seam.
    const linked = applyReferenceExtendSourceVideo([IMAGE_REFERENCE], source24, 3, FRAMES);
    const overridden = linked.map((entry, index) =>
      index === 1 && entry.kind === 'video'
        ? {
            ...entry,
            clip: resizeReferenceSampleWindow(slideReferenceSampleWindow(entry.clip, 100), 120),
            trimOverridden: true,
          }
        : entry
    );

    expect(overridden[1]).toMatchObject({ clip: { endFrame: 219, startFrame: 100 } });

    // Moving the cutpoint on the SAME clip no longer drags the window with it.
    const retrimmed = applyReferenceExtendSourceVideo(overridden, { ...source24, endFrame: 300 }, 3, FRAMES);

    expect(retrimmed[1]).toMatchObject({
      clip: { endFrame: 219, startFrame: 100, video_name: 'long.mp4' },
      fromSourceVideo: true,
      trimOverridden: true,
    });
    // Nor does a frame-count change re-budget it (identity-preserving).
    expect(applyReferenceExtendNumFrames(retrimmed, 124)).toBe(retrimmed);
  });

  it('drops an overridden anchor window when the initial video changes clip', () => {
    // The bounds index frames of a clip that is no longer there.
    const linked = applyReferenceExtendSourceVideo([], source24, 3, FRAMES);
    const overridden = linked.map((entry) =>
      entry.kind === 'video'
        ? { ...entry, clip: slideReferenceSampleWindow(entry.clip, 100), trimOverridden: true }
        : entry
    );
    const other = { ...source24, endFrame: 200, numFrames: 202, video_name: 'other.mp4' };
    const reset = applyReferenceExtendSourceVideo(overridden, other, 3, FRAMES);

    expect(reset[0]).toMatchObject({ clip: { endFrame: 200, startFrame: 60, video_name: 'other.mp4' } });
    expect(reset[0]).not.toHaveProperty('trimOverridden');
  });

  it('clearing the initial video removes only the linked reference (identity-preserving when none)', () => {
    const list = applyReferenceExtendSourceVideo([VIDEO_REFERENCE, IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(applyReferenceExtendSourceVideo(list, null, 3, FRAMES)).toEqual([VIDEO_REFERENCE, IMAGE_REFERENCE]);

    const unlinked = [VIDEO_REFERENCE, IMAGE_REFERENCE];

    expect(applyReferenceExtendSourceVideo(unlinked, null, 3, FRAMES)).toBe(unlinked);
  });

  it('normalization re-establishes the linkage recall drops, and the invariants reach it', () => {
    // Recall must restore the omitted anchor flag so pinning and generated-frame rebudgeting still protect the
    // seam.
    const recalled = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const normalized = normalizeVideoSettings(
      createSettings({ references: [recalled, IMAGE_REFERENCE], sourceVideo: source24 })
    );

    // Flagged by clip identity -- the same rule the setter adopts by -- and pinned.
    expect(normalized?.references).toHaveLength(2);
    expect(normalized?.references[0]).toBe(IMAGE_REFERENCE);
    expect(normalized?.references[1]).toMatchObject({
      clip: { video_name: 'long.mp4' },
      fromSourceVideo: true,
    });

    // The frame-count re-budget now reaches the recalled window.
    const rebudgeted = applyReferenceExtendNumFrames(normalized!.references, 90);

    expect(rebudgeted[1]).toMatchObject({ clip: { endFrame: 47, startFrame: 9 } });

    // A flagged entry stays authoritative: an unflagged same-name entry beside
    // it is NOT a second anchor.
    const flaggedElsewhere = normalizeVideoSettings(
      createSettings({
        references: [recalled, { ...VIDEO_REFERENCE, fromSourceVideo: true }],
        sourceVideo: source24,
      })
    );

    expect(
      flaggedElsewhere?.references.filter((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)
    ).toHaveLength(1);
    expect(flaggedElsewhere?.references[1]).toMatchObject({ clip: { video_name: 'ref.mp4' }, fromSourceVideo: true });

    // No source video: nothing to link, nothing flagged.
    const unlinked = normalizeVideoSettings(createSettings({ references: [recalled] }));

    expect(unlinked?.references[0]).not.toHaveProperty('fromSourceVideo');
  });

  it('adopts the LAST same-name entry: the pin invariant records the anchor last', () => {
    // Adopt the last same-name reference; the earlier one can be the user's independently trimmed reference.
    const userRef = {
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, endFrame: 20, startFrame: 0, video_name: 'long.mp4' },
    };
    const tailRef = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const normalized = normalizeVideoSettings(
      createSettings({ references: [userRef, tailRef], sourceVideo: source24 })
    );

    expect(normalized?.references[0]).toBe(userRef);
    expect(normalized?.references[1]).toMatchObject({ clip: { endFrame: 47 }, fromSourceVideo: true });
  });

  it('canonicalizes the flag to at most one entry and keeps normalization stable', () => {
    // Canonicalize duplicate anchor flags to avoid normalization oscillation and unbounded exempt entries.
    const flaggedNamed = (video_name: string) => ({
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name },
      fromSourceVideo: true,
    });
    const once = normalizeVideoSettings(createSettings({ references: [flaggedNamed('a.mp4'), flaggedNamed('b.mp4')] }));
    const twice = normalizeVideoSettings(createSettings({ references: once!.references }));

    expect(once?.references.filter((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)).toHaveLength(
      1
    );
    expect(once?.references.at(-1)).toMatchObject({ clip: { video_name: 'b.mp4' }, fromSourceVideo: true });
    expect(twice?.references).toEqual(once?.references);

    const overCap = normalizeVideoSettings(
      createSettings({ references: ['a', 'b', 'c', 'd'].map((name) => flaggedNamed(`${name}.mp4`)) })
    );

    expect(overCap?.references).toHaveLength(3);
    expect(
      overCap?.references.filter((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)
    ).toHaveLength(1);
    expect(overCap?.references.at(-1)).toMatchObject({ clip: { video_name: 'd.mp4' }, fromSourceVideo: true });
  });

  it('an absurd probed frame rate falls back to 24 instead of hanging', () => {
    // Reject unrealistic rates before floating-point increments stop advancing adjustment loops.
    expect(deriveReferenceExtendClip({ ...source24, fps: 1e17 }, 141)).toMatchObject({
      endFrame: 400,
      startFrame: 260,
    });
    // Retain legitimate high-rate timing; an overly low fallback bound can reduce resampled context below backend
    // minimums.
    const highRate = { ...source24, endFrame: 50000, fps: 1200, numFrames: 50001 };

    expect(deriveReferenceExtendClip(highRate, 141)).toMatchObject({ endFrame: 50000, startFrame: 42976 });
    expect(deriveReferenceExtendClip({ ...highRate, fps: 1e17 }, 141)).toMatchObject({ startFrame: 49860 });
  });

  it('a demoted anchor loses its override, so re-setting the clip re-derives', () => {
    // Clear override/sample intent when demoting duplicate anchors so later adoption can restore defaults.
    const overriddenNamed = (video_name: string) => ({
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, endFrame: 60, startFrame: 40, video_name },
      fromSourceVideo: true,
      trimOverridden: true,
    });
    const healed = normalizeVideoSettings(
      createSettings({ references: [overriddenNamed('long.mp4'), overriddenNamed('b.mp4')] })
    );

    expect(healed?.references[0]).toMatchObject({ fromSourceVideo: false, trimOverridden: false });

    const adopted = applyReferenceExtendSourceVideo(healed!.references.slice(0, 1), source24, 3, FRAMES);

    expect(adopted[0]).toMatchObject({ clip: { endFrame: 400, startFrame: 260 }, fromSourceVideo: true });
  });

  it('adopts an unflagged reference for the same clip instead of duplicating it (recall shape)', () => {
    const recalled = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const result = applyReferenceExtendSourceVideo([IMAGE_REFERENCE, recalled], source24, 3, FRAMES);

    expect(result).toHaveLength(2);
    expect(result[1]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260, video_name: 'long.mp4' },
      fromSourceVideo: true,
    });
  });

  it('prefers the flagged entry over an earlier same-name reference on a source swap', () => {
    // Source replacement updates the flagged anchor without overwriting the user's separate reference to the new
    // clip.
    const linkedA = applyReferenceExtendSourceVideo([], source24, 3, FRAMES)[0];
    const handTrimmedB = {
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, endFrame: 200, numFrames: 300, startFrame: 100, video_name: 'b.mp4' },
    };
    const sourceB = { ...source24, endFrame: 290, numFrames: 300, video_name: 'b.mp4' };
    const result = applyReferenceExtendSourceVideo([handTrimmedB, linkedA!], sourceB, 3, FRAMES);

    expect(result[0]).toBe(handTrimmedB);
    expect(result[1]).toMatchObject({
      clip: { endFrame: 290, startFrame: 150, video_name: 'b.mp4' },
      fromSourceVideo: true,
    });
    expect(result.filter((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)).toHaveLength(1);
  });

  it('leaves a full video-reference list unchanged instead of overflowing the cap', () => {
    const full = [
      VIDEO_REFERENCE,
      { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'b.mp4' } },
      { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'c.mp4' } },
    ];

    expect(applyReferenceExtendSourceVideo(full, source24, 3, FRAMES)).toBe(full);
  });

  it('applyReferenceExtendNumFrames re-derives the window and is idempotent', () => {
    const linked = applyReferenceExtendSourceVideo([IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(linked[1]).toMatchObject({ clip: { startFrame: 260 } });

    const at124 = applyReferenceExtendNumFrames(linked, 124);

    expect(at124[1]).toMatchObject({ clip: { endFrame: 400, startFrame: 277 }, fromSourceVideo: true });
    expect(at124[0]).toBe(linked[0]);
    // Idempotent, and identity-preserving when nothing moves.
    expect(applyReferenceExtendNumFrames(at124, 124)).toBe(at124);
    expect(applyReferenceExtendNumFrames(linked, FRAMES)).toBe(linked);
  });

  it('applyReferenceExtendNumFrames does not ratchet: the window re-widens', () => {
    // Rebudget in both directions because number-input keystrokes temporarily shrink frame counts.
    const linked = applyReferenceExtendSourceVideo([], source24, 3, 345);
    const typed = [3, 34, 345].reduce(applyReferenceExtendNumFrames, linked);

    expect(typed[0]).toMatchObject({ clip: { endFrame: 400, startFrame: 260 } });
    // And a slider dragged down and back up ends where it started.
    expect([90, 124, 345].reduce(applyReferenceExtendNumFrames, linked)[0]).toMatchObject({
      clip: { startFrame: 260 },
    });
  });

  it('an updater applied after a concurrent write keeps both changes', () => {
    // Apply async reference additions against live state so intervening source-anchor insertion survives.
    const captured: VideoReferenceItem[] = [];
    const add = (current: VideoReferenceItem[]): VideoReferenceItem[] => [...current, IMAGE_REFERENCE];

    // ...the Initial Video lands mid-flight.
    const live = applyReferenceExtendSourceVideo(captured, source24, 3, FRAMES);

    expect(live.some((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)).toBe(true);

    // ...then the resolve writes, against LIVE state rather than `captured`.
    const merged = pinReferenceExtendAnchor(add(live));

    expect(merged).toHaveLength(2);
    expect(merged[0]).toBe(IMAGE_REFERENCE);
    expect(merged[1]).toMatchObject({ fromSourceVideo: true, kind: 'video' });

    // The snapshot write this replaced would have produced just the image.
    expect(add(captured)).toEqual([IMAGE_REFERENCE]);
  });

  it('pins the continuity anchor last, whatever the add or drag order', () => {
    // Keep the source anchor last when adding images because generated rotary positions follow the last reference
    // block.
    const linked = applyReferenceExtendSourceVideo([], source24, 3, FRAMES);
    const anchor = linked[0]!;

    expect(pinReferenceExtendAnchor([anchor, IMAGE_REFERENCE])).toEqual([IMAGE_REFERENCE, anchor]);
    expect(pinReferenceExtendAnchor([IMAGE_REFERENCE, VIDEO_REFERENCE, anchor])).toEqual([
      IMAGE_REFERENCE,
      VIDEO_REFERENCE,
      anchor,
    ]);
    // Identity-preserving when already last, and when there is no anchor.
    const settled = [IMAGE_REFERENCE, anchor];

    expect(pinReferenceExtendAnchor(settled)).toBe(settled);
    const none = [IMAGE_REFERENCE, VIDEO_REFERENCE];

    expect(pinReferenceExtendAnchor(none)).toBe(none);

    // Adopting a recalled entry re-pins it rather than leaving it in place.
    const recalled = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const adopted = applyReferenceExtendSourceVideo([recalled, IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(adopted[1]).toMatchObject({ fromSourceVideo: true, kind: 'video' });
  });

  it('canPlaceReferenceExtendAnchor agrees with the setter in every state', () => {
    const video = (name: string, flagged = false) => ({
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name: name },
      ...(flagged ? { fromSourceVideo: true } : {}),
    });
    const states: VideoReferenceItem[][] = [
      [],
      [IMAGE_REFERENCE],
      [video('a.mp4')],
      [video('a.mp4'), video('b.mp4')],
      [video('a.mp4'), video('b.mp4'), video('c.mp4')],
      // Recall's shape: three unflagged videos, one naming the source clip.
      [video('long.mp4'), video('b.mp4'), video('c.mp4')],
      [video('a.mp4'), video('b.mp4'), video('long.mp4', true)],
    ];

    for (const references of states) {
      const placed = applyReferenceExtendSourceVideo(references, source24, 3, FRAMES);
      // The setter signals refusal by returning the input array unchanged.
      const setterAccepted = placed !== references;

      expect({
        references: references.map((e) => (e.kind === 'video' ? e.clip.video_name : 'img')),
        placed: canPlaceReferenceExtendAnchor(references, source24.video_name, 3),
      }).toEqual({
        references: references.map((e) => (e.kind === 'video' ? e.clip.video_name : 'img')),
        placed: setterAccepted,
      });
    }
  });

  it('applyReferenceExtendNumFrames leaves unlinked references alone', () => {
    const unlinked = [VIDEO_REFERENCE, IMAGE_REFERENCE];

    expect(applyReferenceExtendNumFrames(unlinked, 90)).toBe(unlinked);
  });
});

describe('reference sample window', () => {
  const clip = (startFrame: number, endFrame: number, numFrames = 300) => ({
    endFrame,
    fps: 24,
    height: 480,
    numFrames,
    startFrame,
    video_name: 'clip.mp4',
    width: 640,
  });

  describe('slideReferenceSampleWindow', () => {
    it('slides a window at constant length', () => {
      const next = slideReferenceSampleWindow(clip(0, 199), 50);
      expect([next.startFrame, next.endFrame]).toEqual([50, 249]);
    });

    it('keeps the length while the clip can still supply it', () => {
      const next = slideReferenceSampleWindow(clip(0, 199), 100);
      expect([next.startFrame, next.endFrame]).toEqual([100, 299]);
    });

    it('pins the length to the frames left instead of blocking the start', () => {
      // The start reaches the frame the user picked; the sample is what gives way.
      const shortened = slideReferenceSampleWindow(clip(0, 199), 250);
      expect([shortened.startFrame, shortened.endFrame]).toEqual([250, 299]);
      const lastFrame = slideReferenceSampleWindow(clip(0, 199), 299);
      expect([lastFrame.startFrame, lastFrame.endFrame]).toEqual([299, 299]);
    });

    it('moves an extend-anchor window off the cutpoint like any other', () => {
      // Allow anchor windows away from the cutpoint; Ref2VA has no frame-exact seam to preserve.
      const next = slideReferenceSampleWindow(clip(180, 298), 100);
      expect([next.startFrame, next.endFrame]).toEqual([100, 218]);
    });

    it('self-heals a corrupt persisted trim into bounds', () => {
      // end < start and end beyond the clip must both come back as a valid window.
      const inverted = slideReferenceSampleWindow(clip(10, 5, 20), 0);
      expect(inverted.startFrame).toBeGreaterThanOrEqual(0);
      expect(inverted.endFrame).toBeGreaterThanOrEqual(inverted.startFrame);
      expect(inverted.endFrame).toBeLessThanOrEqual(19);
      const oversized = slideReferenceSampleWindow(clip(0, 999, 20), 5);
      expect([oversized.startFrame, oversized.endFrame]).toEqual([5, 19]);
    });

    it('handles a single-frame clip', () => {
      const next = slideReferenceSampleWindow(clip(0, 0, 1), 5);
      expect([next.startFrame, next.endFrame]).toEqual([0, 0]);
    });

    it('a drag past the clip end comes back with its length intact', () => {
      // Preserve requested sample length through overshoot-and-return slider updates rather than repeatedly
      // shrinking the effective window.
      let reference: Extract<VideoReferenceItem, { kind: 'video' }> = {
        clip: clip(0, 199),
        conditioning: 'video_audio',
        kind: 'video',
      };
      const drag = (rawStart: number) => {
        const sampleFrames = referenceSampleFrames(reference);

        reference = {
          ...reference,
          clip: slideReferenceSampleWindow(reference.clip, rawStart, sampleFrames),
          sampleFrames,
        };

        return [reference.clip.startFrame, reference.clip.endFrame];
      };

      expect([40, 140, 240, 299, 240, 0].map(drag)).toEqual([
        [40, 239],
        // Pinned at the clip's end on the way out ...
        [140, 299],
        [240, 299],
        [299, 299],
        // ... and restored on the way back.
        [240, 299],
        [0, 199],
      ]);
    });

    it('the length control replaces the recorded request', () => {
      // Shortening the sample WHILE clamped is a deliberate choice, so sliding
      // back must restore that length and not the one it replaced.
      const shortened = resizeReferenceSampleWindow(clip(250, 299), 30);
      expect([shortened.startFrame, shortened.endFrame]).toEqual([250, 279]);
      expect(clampReferenceSampleFrames(shortened, 30)).toBe(30);
      expect(slideReferenceSampleWindow(shortened, 0, 30)).toMatchObject({ endFrame: 29, startFrame: 0 });
    });

    it('referenceSampleFrames falls back to the window until a control is touched', () => {
      const untouched: Extract<VideoReferenceItem, { kind: 'video' }> = {
        clip: clip(10, 29),
        conditioning: 'video_audio',
        kind: 'video',
      };
      expect(referenceSampleFrames(untouched)).toBe(20);
      expect(referenceSampleFrames({ ...untouched, sampleFrames: 200 })).toBe(200);
      // A recorded request larger than the clip could ever hold still clamps.
      expect(referenceSampleFrames({ ...untouched, sampleFrames: 9999 })).toBe(300);
    });
  });

  describe('resizeReferenceSampleWindow', () => {
    it('grows a window forward from its start', () => {
      const next = resizeReferenceSampleWindow(clip(50, 60), 100);
      expect([next.startFrame, next.endFrame]).toEqual([50, 149]);
    });

    it('clamps the length to the clip end', () => {
      const next = resizeReferenceSampleWindow(clip(250, 260), 100);
      expect([next.startFrame, next.endFrame]).toEqual([250, 299]);
    });

    it('shortens an extend-anchor window away from the cutpoint', () => {
      const next = resizeReferenceSampleWindow(clip(180, 298), 20);
      expect([next.startFrame, next.endFrame]).toEqual([180, 199]);
    });

    it('never produces a window shorter than one frame', () => {
      const next = resizeReferenceSampleWindow(clip(50, 199), -5);
      expect([next.startFrame, next.endFrame]).toEqual([50, 50]);
    });
  });
});

describe('panel patches shared by the Video panel and gallery/API placement', () => {
  const clip = createVideoSourceClip({ durationSeconds: 5, fps: 16, height: 480, name: 'clip.mp4', width: 832 });
  const imageReference: VideoReferenceItem = {
    detail: 'max',
    image: { height: 512, image_name: 'ref.png', width: 512 },
    kind: 'image',
  };

  it('sets an initial video on a plain panel without touching the references', () => {
    expect(
      getInitialVideoPatch({ maxVideos: 0, numFrames: 81, referenceExtend: false, references: [], sourceVideo: clip })
    ).toEqual({ conditioningClip: null, firstFrameImage: null, sourceVideo: clip });
  });

  it('clears an initial video without displacing anything', () => {
    expect(
      getInitialVideoPatch({ maxVideos: 3, numFrames: 81, referenceExtend: true, references: [], sourceVideo: null })
    ).toEqual({ references: [], sourceVideo: null });
  });

  it('lets references displace the initial video only on a panel that does not extend from it', () => {
    expect(getReferencesPatch({ referenceExtend: false, references: [imageReference] })).toEqual({
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      references: [imageReference],
      sourceVideo: null,
    });
    expect(getReferencesPatch({ referenceExtend: true, references: [imageReference] })).not.toHaveProperty(
      'sourceVideo'
    );
    expect(getReferencesPatch({ referenceExtend: false, references: [] })).toEqual({ references: [] });
  });
});

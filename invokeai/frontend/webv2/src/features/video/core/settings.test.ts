import { describe, expect, it } from 'vitest';

import type { VideoReferenceItem, VideoSettings, VideoWidgetValues } from './types';

import { MINIMAX_H3_NUM_FRAMES_CHOICES } from './dimensions';
import {
  DEFAULT_REFERENCE_SAMPLE_FRAMES,
  resizeReferenceSampleWindow,
  slideReferenceSampleWindow,
  anchorReferenceConditioning,
  applyReferenceExtendSourceVideo,
  applyReferenceExtendNumFrames,
  pinReferenceExtendAnchor,
  clearDeletedVideoMedia,
  createVideoSourceClip,
  deriveReferenceExtendClip,
  formatReferencePromptLabels,
  getDefaultReferenceClip,
  getDefaultReferenceConditioning,
  getInitialVideoPatch,
  getReferencesPatch,
  isVideoSettings,
  createFrameImageSetter,
  createVideoConditioningClip,
  getFrameImagePatch,
  normalizeVideoSettings,
  normalizeVideoWidgetValues,
  referencePromptLabels,
  resolveVideoMode,
  videoClipSpanSeconds,
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

describe('H3 task override normalization', () => {
  it.each([undefined, 'fl2va', 'ref2va'] as const)(
    'preserves explicit intent %s without rewriting the catalog model',
    (h3TaskOverride) => {
      const model = {
        base: 'minimax-h3',
        format: 'checkpoint',
        type: 'main' as const,
        key: 'h3',
        name: 'H3',
        variant: 'fl2va',
      };
      const values = { ...createSettings({ h3TaskOverride }), model };
      expect(isVideoSettings(values)).toBe(true);
      expect(normalizeVideoWidgetValues(values)).toMatchObject({ model, h3TaskOverride });
      expect(normalizeVideoWidgetValues(values)?.model).toBe(model);
    }
  );

  it('rejects invalid overrides as canonical and normalizes them to Detected', () => {
    const values = { ...createSettings(), h3TaskOverride: 'unknown' };
    expect(isVideoSettings(values)).toBe(false);
    expect(normalizeVideoSettings(values)?.h3TaskOverride).toBeUndefined();
  });
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
    expect(resolveVideoMode(createSettings({ lastFrameImage: LAST_FRAME, sourceVideo: SOURCE_VIDEO }))).toBe('extend');
  });

  it("reads the conditioning clip's role as the mode, ahead of every frame and clip slot", () => {
    expect(
      resolveVideoMode(createSettings({ conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' } }))
    ).toBe('audio-to-video');
    expect(
      resolveVideoMode(createSettings({ conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'video' } }))
    ).toBe('video-to-audio');
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

    expect(createVideoConditioningClip({ ...item, mediaOrigin: 'audio_upload' }).role).toBe('audio');
    expect(createVideoConditioningClip(item).role).toBe('video');
  });
});

describe('normalizeVideoSettings', () => {
  const imageReference: VideoReferenceItem = {
    detail: 'max',
    image: { height: 512, image_name: 'ref.png', width: 512 },
    kind: 'image',
  };

  it('round-trips references beside first/last frames; the model decides validity, not the stored record', () => {
    const settings = createSettings({
      firstFrameImage: FIRST_FRAME,
      lastFrameImage: LAST_FRAME,
      references: [imageReference],
    });

    expect(isVideoSettings(settings)).toBe(true);
    expect(normalizeVideoSettings(settings)).toMatchObject({
      firstFrameImage: FIRST_FRAME,
      lastFrameImage: LAST_FRAME,
      references: [imageReference],
    });
    expect(resolveVideoMode(settings)).toBe('reference');
  });

  it('keeps the source video and drops frames from a stale reference-extension record', () => {
    const stale = createSettings({
      firstFrameImage: FIRST_FRAME,
      lastFrameImage: LAST_FRAME,
      references: [imageReference],
      sourceVideo: SOURCE_VIDEO,
    });

    expect(isVideoSettings(stale)).toBe(false);
    expect(normalizeVideoSettings(stale)).toMatchObject({
      firstFrameImage: null,
      lastFrameImage: null,
      sourceVideo: SOURCE_VIDEO,
    });
  });

  it('reads the seed mode saved before modes existed from the random toggle', () => {
    const legacy = { ...createSettings({}), seedMode: undefined };

    expect(normalizeVideoSettings({ ...legacy, shouldRandomizeSeed: false })?.seedMode).toBe('fixed');
    expect(normalizeVideoSettings({ ...legacy, shouldRandomizeSeed: true })?.seedMode).toBe('random');
    expect(normalizeVideoSettings({ ...legacy, seedMode: 'increment', shouldRandomizeSeed: false })?.seedMode).toBe(
      'increment'
    );
    expect(normalizeVideoSettings(legacy)?.seedMode).toBe('random');
    expect(isVideoSettings({ ...legacy, shouldRandomizeSeed: true })).toBe(false);
    expect(isVideoSettings(legacy)).toBe(false);
  });

  it("reads dynamic prompt settings saved before Video had them as off, with Video's prompt cap", () => {
    const legacy: Record<string, unknown> = { ...createSettings() };

    for (const key of Object.keys(legacy).filter((name) => name.startsWith('dynamicPrompts'))) {
      delete legacy[key];
    }

    expect(normalizeVideoSettings(legacy)).toMatchObject({
      dynamicPromptsCombinatorial: true,
      dynamicPromptsEnabled: false,
      dynamicPromptsMaxPrompts: 10,
      dynamicPromptsResample: true,
      dynamicPromptsSampleSeed: 0,
      dynamicPromptsSeedBehaviour: 'per-iteration',
    });
    expect(isVideoSettings(legacy)).toBe(false);
  });

  it('round-trips canonical settings', () => {
    const settings = createSettings({ firstFrameImage: FIRST_FRAME, positivePrompt: 'a cat' });
    const normalized = normalizeVideoSettings(settings);

    expect(normalized).toEqual(settings);
    expect(isVideoSettings(settings)).toBe(true);
  });

  it('rejects non-records but heals partial records field-by-field, upscale-style', () => {
    expect(normalizeVideoSettings(null)).toBeNull();
    expect(normalizeVideoSettings(7)).toBeNull();
    const seeded = normalizeVideoSettings({ firstFrameImage: FIRST_FRAME, sourceVideo: null });

    expect(seeded).not.toBeNull();
    expect(seeded?.firstFrameImage).toEqual(FIRST_FRAME);
    expect(seeded).toMatchObject({ fps: 16, modelKey: '', numFrames: 81, steps: 40, targetResolution: '720p' });
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

  it('drops malformed media values instead of failing wholesale', () => {
    const normalized = normalizeVideoSettings({
      ...createSettings(),
      firstFrameImage: { image_name: 'x.png' },
      sourceVideo: { video_name: 'clip.mp4' },
    });

    expect(normalized?.firstFrameImage).toBeNull();
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

describe('videoClipSpanSeconds', () => {
  it('runs to the far edge of the last selected frame', () => {
    // 16 fps, frames 0..79 inclusive: stopping at 79/16 would cut the final frame short.
    expect(videoClipSpanSeconds(SOURCE_VIDEO)).toEqual({ endSeconds: 5, startSeconds: 0 });
    expect(videoClipSpanSeconds({ ...SOURCE_VIDEO, endFrame: 47, startFrame: 32 })).toEqual({
      endSeconds: 3,
      startSeconds: 2,
    });
  });
});

describe('normalizeVideoWidgetValues / cloneVideoWidgetValues', () => {
  const model = { base: 'wan', key: 'wan-key', name: 'Wan', type: 'main' as const, variant: 't2v_a14b' };

  it('carries a valid main model and nulls an invalid one', () => {
    expect(normalizeVideoWidgetValues({ ...createSettings(), model })?.model).toEqual(model);
    expect(normalizeVideoWidgetValues({ ...createSettings(), model: { key: 'x' } })?.model).toBeNull();
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
});

describe('getDefaultReferenceConditioning', () => {
  it('starts a wrapped audio upload on its soundtrack alone', () => {
    expect(getDefaultReferenceConditioning('audio_upload')).toBe('audio');
  });
});

describe('getDefaultReferenceClip', () => {
  const wrappedAudio = {
    endFrame: 199,
    fps: 24,
    height: 512,
    numFrames: 4320,
    startFrame: 120,
    video_name: 'song.mp4',
    width: 512,
  };

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

  it('gives an audio-only reference the whole clip', () => {
    expect(getDefaultReferenceClip(wrappedAudio, 'audio')).toEqual({
      ...wrappedAudio,
      endFrame: 4319,
      startFrame: 0,
    });
  });
});

describe('getFrameImagePatch', () => {
  const soundtrack = { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' as const };
  const picture = { ...soundtrack, role: 'video' as const };

  it('displaces a clip held for its picture, which already supplies every frame', () => {
    expect(getFrameImagePatch('firstFrameImage', FIRST_FRAME, picture)).toEqual({
      conditioningClip: null,
      firstFrameImage: FIRST_FRAME,
      sourceVideo: null,
    });
    expect(getFrameImagePatch('lastFrameImage', FIRST_FRAME, picture)).toEqual({
      conditioningClip: null,
      lastFrameImage: FIRST_FRAME,
    });
  });
});

describe('createFrameImageSetter', () => {
  const soundtrack = { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' as const };
  const picture = { ...soundtrack, role: 'video' as const };

  const racePanel = (startRole: typeof soundtrack | typeof picture, endRole: typeof soundtrack | typeof picture) => {
    let raw: Record<string, unknown> = { ...createSettings(), conditioningClip: startRole };
    const patches: Partial<VideoWidgetValues>[] = [];
    const setLastFrame = createFrameImageSetter(
      'lastFrameImage',
      () => raw,
      (values) => patches.push(values)
    );

    raw = { ...raw, conditioningClip: endRole };
    setLastFrame(FIRST_FRAME);

    return patches;
  };

  it('displaces a clip switched to its picture while the frame was resolving', () => {
    expect(racePanel(soundtrack, picture)).toEqual([{ conditioningClip: null, lastFrameImage: FIRST_FRAME }]);
  });
});

describe('normalizeVideoSettings — the conditioning clip', () => {
  const clip = { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' as const };

  it('drops a clip stored beside a slot that claims the same conditioning mask', () => {
    const picture = { ...clip, role: 'video' as const };

    expect(
      normalizeVideoSettings({ ...createSettings({ firstFrameImage: FIRST_FRAME }), conditioningClip: picture })
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
});

describe('clearDeletedVideoMedia', () => {
  const withMedia = createSettings({
    firstFrameImage: FIRST_FRAME,
    lastFrameImage: LAST_FRAME,
    sourceVideo: null,
  });

  it('clears exactly the deleted references', () => {
    const cleared = clearDeletedVideoMedia(withMedia, new Set(['first.png']), new Set());

    expect(cleared.firstFrameImage).toBeNull();
    expect(cleared.lastFrameImage).toEqual(LAST_FRAME);

    const withClip = createSettings({ sourceVideo: SOURCE_VIDEO });
    const clipCleared = clearDeletedVideoMedia(withClip, new Set(), new Set(['clip.mp4']));

    expect(clipCleared.sourceVideo).toBeNull();

    const withConditioning = createSettings({
      conditioningClip: { clip: CONDITIONING_CLIP, fpsKnown: true, role: 'audio' },
    });

    expect(
      clearDeletedVideoMedia(withConditioning, new Set(), new Set(['conditioning.mp4'])).conditioningClip
    ).toBeNull();
    expect(clearDeletedVideoMedia(withConditioning, new Set(), new Set(['other.mp4']))).toBe(withConditioning);
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
    const anchor = {
      ...VIDEO_REFERENCE,
      clip: { ...VIDEO_REFERENCE.clip, video_name: 'anchor.mp4' },
      fromSourceVideo: true,
    };
    const named = (entry: { kind: string; clip?: { video_name: string } }) =>
      entry.kind === 'video' ? entry.clip!.video_name : 'img';

    const healed = normalizeVideoSettings(createSettings({ references: [anchor, IMAGE_REFERENCE] as never }));

    expect(healed?.references.map(named as never)).toEqual(['img', 'anchor.mp4']);

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

  it('preserves valid Wan LoRA targets and lowNoiseWeight and strips invalid ones on normalizeVideoSettings', () => {
    const loraModel = { base: 'wan', key: 'wan-lora', name: 'Wan LoRA', type: 'lora' };
    const normalized = normalizeVideoSettings(
      createSettings({
        loras: [
          { isEnabled: true, lowNoiseWeight: 0.4, model: loraModel as never, target: 'both', weight: 0.8 },
          { isEnabled: true, lowNoiseWeight: 0.4, model: loraModel as never, target: 'low', weight: 0.6 },
          { isEnabled: true, model: loraModel as never, target: 'bogus' as never, weight: 0.5 },
        ],
      })
    );

    expect(normalized?.loras).toEqual([
      { isEnabled: true, lowNoiseWeight: 0.4, model: loraModel, target: 'both', weight: 0.8 },
      { isEnabled: true, model: loraModel, target: 'low', weight: 0.6 },
      { isEnabled: true, model: loraModel, weight: 0.5 },
    ]);
    expect(isVideoSettings(normalized)).toBe(true);
    expect(
      isVideoSettings({
        ...normalized,
        loras: [{ isEnabled: true, lowNoiseWeight: 0.4, model: loraModel, target: 'low', weight: 0.6 }],
      })
    ).toBe(false);
  });
});

describe('anchorReferenceConditioning', () => {
  it('converts an audio-only choice, which cannot carry a seam', () => {
    expect(anchorReferenceConditioning('audio')).toBe('video_audio');
  });
});

describe('referencePromptLabels', () => {
  const withConditioning = (conditioning: 'video_audio' | 'video' | 'audio'): VideoReferenceItem => ({
    ...VIDEO_REFERENCE,
    conditioning,
  });

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
});

describe('reference-extend anchor: audio-only references', () => {
  const longSource = { ...SOURCE_VIDEO, endFrame: 400, numFrames: 402, video_name: 'long.mp4' };
  const source24 = { ...longSource, fps: 24 };
  const FRAMES = 141;
  const userAudio = {
    clip: { ...SOURCE_VIDEO, endFrame: 401, numFrames: 402, startFrame: 0, video_name: 'long.mp4' },
    conditioning: 'audio',
    kind: 'video',
  } as const;

  it('appends a real anchor beside an audio-only reference rather than consuming it', () => {
    const linked = applyReferenceExtendSourceVideo([userAudio], source24, 3, FRAMES);

    expect(linked).toHaveLength(2);
    expect(linked[0]).toBe(userAudio);
    expect(linked[1]).toMatchObject({
      clip: { endFrame: 400, startFrame: 260 },
      conditioning: 'video_audio',
      fromSourceVideo: true,
    });
  });
});

describe('reference-extend linkage', () => {
  const longSource = { ...SOURCE_VIDEO, endFrame: 400, numFrames: 402, video_name: 'long.mp4' };
  const source24 = { ...longSource, fps: 24 };
  const FRAMES = 141;

  it('derives the tail trim: the window ending at the cutpoint, clamped at 0', () => {
    expect(deriveReferenceExtendClip(source24, FRAMES)).toMatchObject({ endFrame: 400, startFrame: 260 });
    expect(deriveReferenceExtendClip({ ...SOURCE_VIDEO, fps: 24 }, FRAMES)).toMatchObject({
      endFrame: 79,
      startFrame: 7,
    });
  });

  it('lands the window ON the 17n+5 grid the backend keeps, at every source rate', () => {
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

  it('appends a linked video+audio reference and re-derives it on cutpoint changes', () => {
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

  it('clearing the initial video removes only the linked reference (identity-preserving when none)', () => {
    const list = applyReferenceExtendSourceVideo([VIDEO_REFERENCE, IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(applyReferenceExtendSourceVideo(list, null, 3, FRAMES)).toEqual([VIDEO_REFERENCE, IMAGE_REFERENCE]);

    const unlinked = [VIDEO_REFERENCE, IMAGE_REFERENCE];

    expect(applyReferenceExtendSourceVideo(unlinked, null, 3, FRAMES)).toBe(unlinked);
  });

  it('normalization re-establishes the linkage recall drops, and the invariants reach it', () => {
    const recalled = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const normalized = normalizeVideoSettings(
      createSettings({ references: [recalled, IMAGE_REFERENCE], sourceVideo: source24 })
    );

    expect(normalized?.references).toHaveLength(2);
    expect(normalized?.references[0]).toBe(IMAGE_REFERENCE);
    expect(normalized?.references[1]).toMatchObject({
      clip: { video_name: 'long.mp4' },
      fromSourceVideo: true,
    });

    const rebudgeted = applyReferenceExtendNumFrames(normalized!.references, 90);

    expect(rebudgeted[1]).toMatchObject({ clip: { endFrame: 47, startFrame: 9 } });

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

    const unlinked = normalizeVideoSettings(createSettings({ references: [recalled] }));

    expect(unlinked?.references[0]).not.toHaveProperty('fromSourceVideo');
  });

  it('an updater applied after a concurrent write keeps both changes', () => {
    const captured: VideoReferenceItem[] = [];
    const add = (current: VideoReferenceItem[]): VideoReferenceItem[] => [...current, IMAGE_REFERENCE];

    const live = applyReferenceExtendSourceVideo(captured, source24, 3, FRAMES);

    expect(live.some((entry) => entry.kind === 'video' && entry.fromSourceVideo === true)).toBe(true);

    const merged = pinReferenceExtendAnchor(add(live));

    expect(merged).toHaveLength(2);
    expect(merged[0]).toBe(IMAGE_REFERENCE);
    expect(merged[1]).toMatchObject({ fromSourceVideo: true, kind: 'video' });

    expect(add(captured)).toEqual([IMAGE_REFERENCE]);
  });

  it('pins the continuity anchor last, whatever the add or drag order', () => {
    const linked = applyReferenceExtendSourceVideo([], source24, 3, FRAMES);
    const anchor = linked[0]!;

    expect(pinReferenceExtendAnchor([anchor, IMAGE_REFERENCE])).toEqual([IMAGE_REFERENCE, anchor]);
    expect(pinReferenceExtendAnchor([IMAGE_REFERENCE, VIDEO_REFERENCE, anchor])).toEqual([
      IMAGE_REFERENCE,
      VIDEO_REFERENCE,
      anchor,
    ]);
    const settled = [IMAGE_REFERENCE, anchor];

    expect(pinReferenceExtendAnchor(settled)).toBe(settled);
    const none = [IMAGE_REFERENCE, VIDEO_REFERENCE];

    expect(pinReferenceExtendAnchor(none)).toBe(none);

    const recalled = { ...VIDEO_REFERENCE, clip: { ...VIDEO_REFERENCE.clip, video_name: 'long.mp4' } };
    const adopted = applyReferenceExtendSourceVideo([recalled, IMAGE_REFERENCE], source24, 3, FRAMES);

    expect(adopted[1]).toMatchObject({ fromSourceVideo: true, kind: 'video' });
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

    it('pins the length to the frames left instead of blocking the start', () => {
      const shortened = slideReferenceSampleWindow(clip(0, 199), 250);
      expect([shortened.startFrame, shortened.endFrame]).toEqual([250, 299]);
      const lastFrame = slideReferenceSampleWindow(clip(0, 199), 299);
      expect([lastFrame.startFrame, lastFrame.endFrame]).toEqual([299, 299]);
    });

    it('self-heals a corrupt persisted trim into bounds', () => {
      const inverted = slideReferenceSampleWindow(clip(10, 5, 20), 0);
      expect(inverted.startFrame).toBeGreaterThanOrEqual(0);
      expect(inverted.endFrame).toBeGreaterThanOrEqual(inverted.startFrame);
      expect(inverted.endFrame).toBeLessThanOrEqual(19);
      const oversized = slideReferenceSampleWindow(clip(0, 999, 20), 5);
      expect([oversized.startFrame, oversized.endFrame]).toEqual([5, 19]);
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
  });
});

describe('panel patches shared by the Video panel and gallery/API placement', () => {
  const imageReference: VideoReferenceItem = {
    detail: 'max',
    image: { height: 512, image_name: 'ref.png', width: 512 },
    kind: 'image',
  };

  it('clears the last frame under a placed Initial Video only where frames can sit beside references', () => {
    const sourceVideo = createVideoSourceClip({ durationSeconds: 4, fps: 24, height: 768, name: 's.mp4', width: 1344 });
    const patch = (referenceFrames: boolean) =>
      getInitialVideoPatch({
        maxVideos: 3,
        numFrames: 97,
        referenceExtend: false,
        referenceFrames,
        references: [],
        sourceVideo,
      });

    expect(patch(true)).toMatchObject({ firstFrameImage: null, lastFrameImage: null });
    expect(patch(false)).not.toHaveProperty('lastFrameImage');
  });

  it('displaces the frames on a reference-extend panel too, beside the linked tail reference', () => {
    const sourceVideo = createVideoSourceClip({ durationSeconds: 4, fps: 24, height: 768, name: 's.mp4', width: 1344 });
    const placed = getInitialVideoPatch({
      maxVideos: 3,
      numFrames: 97,
      referenceExtend: true,
      referenceFrames: true,
      references: [],
      sourceVideo,
    });

    expect(placed).toMatchObject({ firstFrameImage: null, lastFrameImage: null, sourceVideo });
    expect(placed?.references).toHaveLength(1);
  });

  it('lets references displace the initial video only on a panel that does not extend from it', () => {
    expect(
      getReferencesPatch({ referenceExtend: false, referenceFrames: false, references: [imageReference] })
    ).toEqual({
      conditioningClip: null,
      firstFrameImage: null,
      lastFrameImage: null,
      references: [imageReference],
      sourceVideo: null,
    });
    expect(getReferencesPatch({ referenceExtend: false, referenceFrames: true, references: [imageReference] })).toEqual(
      {
        conditioningClip: null,
        references: [imageReference],
        sourceVideo: null,
      }
    );
    expect(
      getReferencesPatch({ referenceExtend: true, referenceFrames: false, references: [imageReference] })
    ).not.toHaveProperty('sourceVideo');
    expect(getReferencesPatch({ referenceExtend: false, referenceFrames: false, references: [] })).toEqual({
      references: [],
    });
  });
});

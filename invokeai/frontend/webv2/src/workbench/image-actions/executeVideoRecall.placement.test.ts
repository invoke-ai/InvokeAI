import type { ModelConfig } from '@features/models';
import type { VideoReferenceItem } from '@features/video';

import {
  createDefaultVideoWidgetValues,
  createVideoConditioningClip,
  createVideoReferenceEntry,
} from '@features/video';
import { describe, expect, it } from 'vitest';

import {
  appendReferenceVideo,
  getVideoPlacementRoom,
  placeConditioningClip,
  placeInitialVideo,
  placeVideoImage,
} from './executeVideoRecall';

const model = (fields: Record<string, unknown>) =>
  ({
    file_size: 1,
    hash: 'h',
    path: '/m',
    source: 'local',
    source_type: 'path',
    type: 'main',
    ...fields,
  }) as ModelConfig;

const WAN_I2V = model({ base: 'wan', format: 'diffusers', key: 'wan-i2v', name: 'Wan I2V', variant: 'i2v_a14b' });
const WAN_T2V = model({ base: 'wan', format: 'diffusers', key: 'wan-t2v', name: 'Wan T2V', variant: 't2v_a14b' });
// A Ref2VA Diffusers folder is only a component source; the selectable main is a checkpoint.
const REF2VA = model({
  base: 'minimax-h3',
  format: 'checkpoint',
  key: 'h3-ref2va',
  name: 'H3 Ref2VA',
  variant: 'ref2va',
});
const WAN_TI2V = model({ base: 'wan', format: 'diffusers', key: 'wan-ti2v', name: 'Wan TI2V', variant: 'ti2v_5b' });
const LTX2 = model({ base: 'ltx-2', format: 'checkpoint', key: 'ltx2', name: 'LTX-2.5 dev', variant: 'ltx2_dev' });

const clip = { durationSeconds: 5, fps: 16, height: 480, name: 'clip.mp4', width: 832 };
const videoReference = (name: string): Extract<VideoReferenceItem, { kind: 'video' }> =>
  createVideoReferenceEntry({ ...clip, name });

const panel = (selected: ModelConfig, overrides: Record<string, unknown> = {}) => ({
  ...(createDefaultVideoWidgetValues([selected]) as unknown as Record<string, unknown>),
  ...overrides,
});

describe('placeInitialVideo', () => {
  it('sets the initial video and displaces the first frame on a model that extends clips', () => {
    const placement = placeInitialVideo({
      models: [WAN_I2V],
      video: clip,
      videoValues: panel(WAN_I2V, { firstFrameImage: { height: 480, image_name: 'first.png', width: 832 } }),
    });

    expect(placement).toMatchObject({
      patch: { conditioningClip: null, firstFrameImage: null, sourceVideo: { video_name: 'clip.mp4' } },
      status: 'placed',
      usable: true,
    });
    expect(placement.status === 'placed' && 'references' in placement.patch).toBe(false);
  });

  it('still fills the slot on a model that cannot extend, but says it will go unused', () => {
    expect(placeInitialVideo({ models: [WAN_T2V], video: clip, videoValues: panel(WAN_T2V) })).toMatchObject({
      status: 'placed',
      usable: false,
    });
  });

  it('links the continuity anchor on a reference-extend panel, pinned after the existing references', () => {
    const placement = placeInitialVideo({
      models: [REF2VA],
      video: clip,
      videoValues: panel(REF2VA, { references: [videoReference('dance.mp4')] }),
    });

    // Ref2VA has no extend mode of its own; it extends through the linked reference.
    expect(placement).toMatchObject({ status: 'placed', usable: true });
    const references = placement.status === 'placed' ? (placement.patch.references ?? []) : [];
    expect(
      references.map((entry) => (entry.kind === 'video' ? [entry.clip.video_name, entry.fromSourceVideo] : []))
    ).toEqual([
      ['dance.mp4', undefined],
      ['clip.mp4', true],
    ]);
  });

  it('refuses a reference-extend panel with no video reference slot left for the anchor', () => {
    const full = ['a.mp4', 'b.mp4', 'c.mp4'].map(videoReference);

    expect(
      placeInitialVideo({ models: [REF2VA], video: clip, videoValues: panel(REF2VA, { references: full }) })
    ).toEqual({
      status: 'full',
    });
  });
});

describe('appendReferenceVideo', () => {
  it('appends footage as a short video-and-audio sample, and wrapped audio whole as a soundtrack', () => {
    // 20 s at 16 fps = 320 frames: longer than the 200-frame default sample.
    const long = { ...clip, durationSeconds: 20, name: 'long.mp4' };
    const append = (video: typeof long & { mediaOrigin?: string }) => {
      const placement = appendReferenceVideo({ models: [REF2VA], video, videoValues: panel(REF2VA) });
      return placement.status === 'appended' ? placement.patch.references?.[0] : undefined;
    };

    expect(append(long)).toMatchObject({
      clip: { endFrame: 199, numFrames: 320, startFrame: 0, video_name: 'long.mp4' },
      conditioning: 'video_audio',
      kind: 'video',
    });
    expect(append({ ...long, mediaOrigin: 'audio_upload' })).toMatchObject({
      clip: { endFrame: 319, startFrame: 0 },
      conditioning: 'audio',
    });
  });

  it('counts only video references against the video cap', () => {
    const image: VideoReferenceItem = {
      detail: 'max',
      image: { height: 512, image_name: 'ref.png', width: 512 },
      kind: 'image',
    };
    const videoValues = panel(REF2VA, { references: [videoReference('a.mp4'), videoReference('b.mp4'), image, image] });

    expect(getVideoPlacementRoom({ models: [REF2VA], videoValues }).referenceVideo).toBe(true);
    expect(appendReferenceVideo({ models: [REF2VA], video: clip, videoValues }).status).toBe('appended');
  });

  it('keeps a reference-extend panel continuity anchor last', () => {
    const anchor: VideoReferenceItem = { ...videoReference('source.mp4'), fromSourceVideo: true };
    const placement = appendReferenceVideo({
      models: [REF2VA],
      video: clip,
      videoValues: panel(REF2VA, { references: [anchor] }),
    });

    const references = placement.status === 'appended' ? (placement.patch.references ?? []) : [];
    expect(references.map((entry) => (entry.kind === 'video' ? entry.clip.video_name : null))).toEqual([
      'clip.mp4',
      'source.mp4',
    ]);
  });

  it('declines when every video reference slot is taken', () => {
    const videoValues = panel(REF2VA, { references: ['a.mp4', 'b.mp4', 'c.mp4'].map(videoReference) });

    expect(appendReferenceVideo({ models: [REF2VA], video: clip, videoValues })).toEqual({ status: 'full' });
    expect(getVideoPlacementRoom({ models: [REF2VA], videoValues }).referenceVideo).toBe(false);
  });

  it('declines, rather than switching models, when the panel model takes no reference videos', () => {
    const videoValues = panel(WAN_I2V);

    expect(appendReferenceVideo({ models: [WAN_I2V, REF2VA], video: clip, videoValues })).toEqual({
      status: 'unsupported',
    });
    expect(getVideoPlacementRoom({ models: [WAN_I2V, REF2VA], videoValues }).referenceVideo).toBe(false);
    expect(getVideoPlacementRoom({ models: [REF2VA], videoValues: panel(REF2VA) }).referenceVideo).toBe(true);
  });
});

describe('placeConditioningClip', () => {
  it('sets the clip in the requested role and clears the conditioning slots that role conflicts with', () => {
    const framed = panel(LTX2, {
      firstFrameImage: { height: 480, image_name: 'first.png', width: 832 },
      lastFrameImage: { height: 480, image_name: 'last.png', width: 832 },
    });

    // The picture role holds every frame, so the start and end images go.
    expect(placeConditioningClip({ models: [LTX2], role: 'video', video: clip, videoValues: framed })).toEqual({
      displaced: true,
      patch: {
        conditioningClip: expect.objectContaining({
          clip: expect.objectContaining({ video_name: 'clip.mp4' }),
          role: 'video',
        }),
        firstFrameImage: null,
        lastFrameImage: null,
        references: [],
        sourceVideo: null,
      },
      status: 'placed',
    });
    // A soundtrack leaves them to anchor the picture generated for it.
    expect(placeConditioningClip({ models: [LTX2], role: 'audio', video: clip, videoValues: framed })).toEqual({
      displaced: false,
      patch: {
        conditioningClip: expect.objectContaining({ role: 'audio' }),
        references: [],
        sourceVideo: null,
      },
      status: 'placed',
    });
    // An initial video conflicts with either role.
    expect(
      placeConditioningClip({
        models: [LTX2],
        role: 'audio',
        video: clip,
        videoValues: panel(LTX2, { sourceVideo: { ...videoReference('source.mp4').clip } }),
      })
    ).toMatchObject({ displaced: true, patch: { sourceVideo: null }, status: 'placed' });
  });

  it('replacing a clip on an otherwise empty panel displaces nothing', () => {
    const first = placeConditioningClip({ models: [LTX2], role: 'video', video: clip, videoValues: panel(LTX2) });
    const held = first.status === 'placed' ? first.patch : {};

    expect(
      placeConditioningClip({
        models: [LTX2],
        role: 'audio',
        video: { ...clip, name: 'other.mp4' },
        videoValues: panel(LTX2, held),
      })
    ).toMatchObject({ displaced: false, status: 'placed' });
  });

  it('the requested role overrides the one the gallery record would default to', () => {
    // A plain clip defaults to lending its picture; the caller asked for its soundtrack.
    const placement = placeConditioningClip({ models: [LTX2], role: 'audio', video: clip, videoValues: panel(LTX2) });

    expect(placement.status === 'placed' && placement.patch.conditioningClip?.role).toBe('audio');
  });

  it('refuses the picture of a wrapped audio upload, but takes its soundtrack', () => {
    const song = { ...clip, mediaOrigin: 'audio_upload', name: 'song.mp4' };

    expect(placeConditioningClip({ models: [LTX2], role: 'video', video: song, videoValues: panel(LTX2) })).toEqual({
      status: 'no-picture',
    });
    expect(placeConditioningClip({ models: [LTX2], role: 'audio', video: song, videoValues: panel(LTX2) }).status).toBe(
      'placed'
    );
  });

  it('declines, rather than switching models, when the panel model takes no conditioning clip', () => {
    const videoValues = panel(REF2VA, { references: [videoReference('dance.mp4')] });

    expect(placeConditioningClip({ models: [REF2VA, LTX2], role: 'audio', video: clip, videoValues })).toEqual({
      status: 'unsupported',
    });
    expect(getVideoPlacementRoom({ models: [REF2VA, LTX2], videoValues })).toEqual({
      conditioningAudio: false,
      conditioningVideo: false,
      referenceVideo: true,
    });
    expect(getVideoPlacementRoom({ models: [LTX2], videoValues: panel(LTX2) })).toEqual({
      conditioningAudio: true,
      conditioningVideo: true,
      referenceVideo: false,
    });
  });
});

describe('placeVideoImage', () => {
  const still = (name: string) => ({ height: 512, image_name: name, width: 512 });
  const imageReference = (name: string): VideoReferenceItem => ({ detail: 'match', image: still(name), kind: 'image' });
  const referenceNames = (placement: ReturnType<typeof placeVideoImage>) =>
    (placement.status === 'placed' ? (placement.patch.references ?? []) : []).map((entry) =>
      entry.kind === 'image' ? entry.image.image_name : entry.clip.video_name
    );

  describe('on a model that takes reference images', () => {
    const held = [imageReference('a.png'), videoReference('dance.mp4'), imageReference('b.png')];

    it('replaces the reference images, keeping the reference videos', () => {
      const placement = placeVideoImage({
        append: false,
        image: still('new.png'),
        models: [REF2VA],
        videoValues: panel(REF2VA, { references: held }),
      });

      expect(placement).toMatchObject({ slot: 'reference', status: 'placed' });
      expect(referenceNames(placement)).toEqual(['dance.mp4', 'new.png']);
    });

    it('appends after the held references, with the detail a later image defaults to', () => {
      const placement = placeVideoImage({
        append: true,
        image: still('new.png'),
        models: [REF2VA],
        videoValues: panel(REF2VA, { references: held }),
      });

      expect(referenceNames(placement)).toEqual(['a.png', 'dance.mp4', 'b.png', 'new.png']);
      expect(placement.status === 'placed' && placement.patch.references?.at(-1)).toMatchObject({ detail: 'match' });
    });

    it('gives the first reference image max detail', () => {
      const placement = placeVideoImage({
        append: false,
        image: still('new.png'),
        models: [REF2VA],
        videoValues: panel(REF2VA, { references: held }),
      });

      expect(placement.status === 'placed' && placement.patch.references?.at(-1)).toMatchObject({ detail: 'max' });
    });

    it('keeps a reference-extend continuity anchor last', () => {
      const anchor: VideoReferenceItem = { ...videoReference('source.mp4'), fromSourceVideo: true };
      const placement = placeVideoImage({
        append: true,
        image: still('new.png'),
        models: [REF2VA],
        videoValues: panel(REF2VA, { references: [anchor] }),
      });

      expect(referenceNames(placement)).toEqual(['new.png', 'source.mp4']);
    });

    it('declines an append when every image slot is taken, but a send still replaces them', () => {
      const videoValues = panel(REF2VA, {
        references: Array.from({ length: 9 }, (_, index) => imageReference(`${index}.png`)),
      });

      expect(placeVideoImage({ append: true, image: still('new.png'), models: [REF2VA], videoValues })).toEqual({
        status: 'full',
      });
      expect(
        referenceNames(placeVideoImage({ append: false, image: still('new.png'), models: [REF2VA], videoValues }))
      ).toEqual(['new.png']);
    });
  });

  describe('on a model that takes frames', () => {
    const first = { height: 480, image_name: 'first.png', width: 832 };
    const last = { height: 480, image_name: 'last.png', width: 832 };

    it('sends the image as the first frame, clearing the last frame and the initial video', () => {
      const placement = placeVideoImage({
        append: false,
        image: still('new.png'),
        models: [WAN_I2V],
        videoValues: panel(WAN_I2V, { lastFrameImage: last, sourceVideo: { ...videoReference('source.mp4').clip } }),
      });

      expect(placement).toEqual({
        displaced: true,
        patch: {
          firstFrameImage: still('new.png'),
          lastFrameImage: null,
          references: [],
          sourceVideo: null,
        },
        slot: 'firstFrame',
        status: 'placed',
      });
    });

    it('appends into the first frame, then the last, then declines', () => {
      let videoValues = panel(WAN_I2V);
      const slots: string[] = [];

      for (const name of ['one.png', 'two.png', 'three.png']) {
        const placement = placeVideoImage({ append: true, image: still(name), models: [WAN_I2V], videoValues });

        slots.push(placement.status === 'placed' ? placement.slot : placement.status);
        videoValues = placement.status === 'placed' ? { ...videoValues, ...placement.patch } : videoValues;
      }

      expect(slots).toEqual(['firstFrame', 'lastFrame', 'full']);
      expect(videoValues).toMatchObject({ firstFrameImage: still('one.png'), lastFrameImage: still('two.png') });
    });

    it('appends into the last frame beside an initial video, which holds the first', () => {
      const placement = placeVideoImage({
        append: true,
        image: still('new.png'),
        models: [WAN_I2V],
        videoValues: panel(WAN_I2V, { sourceVideo: { ...videoReference('source.mp4').clip } }),
      });

      expect(placement).toEqual({
        displaced: false,
        patch: { lastFrameImage: still('new.png'), references: [] },
        slot: 'lastFrame',
        status: 'placed',
      });
    });

    it('clears stale references that would hide an appended frame, and the last frame they hid', () => {
      const staleReferences = [{ detail: 'max' as const, image: still('ref.png'), kind: 'image' as const }];
      const afterFirst = placeVideoImage({
        append: true,
        image: still('new.png'),
        models: [LTX2],
        videoValues: panel(LTX2, { lastFrameImage: last, references: staleReferences }),
      });

      expect(afterFirst).toMatchObject({
        patch: { firstFrameImage: still('new.png'), lastFrameImage: null, references: [] },
        slot: 'firstFrame',
      });

      const afterLast = placeVideoImage({
        append: true,
        image: still('new.png'),
        models: [LTX2],
        videoValues: panel(LTX2, {
          references: staleReferences,
          sourceVideo: { ...videoReference('source.mp4').clip },
        }),
      });

      expect(afterLast).toMatchObject({
        patch: { lastFrameImage: still('new.png'), references: [] },
        slot: 'lastFrame',
      });
    });

    it('declines an appended second frame on a model without a last frame', () => {
      expect(
        placeVideoImage({
          append: true,
          image: still('new.png'),
          models: [WAN_TI2V],
          videoValues: panel(WAN_TI2V, { firstFrameImage: first }),
        })
      ).toEqual({ status: 'full' });
    });

    it('clears a clip that holds the picture, as the frame fields do, but keeps a soundtrack', () => {
      const pictureClip = { ...createVideoConditioningClip(clip), role: 'video' as const };
      const soundtrack = { ...createVideoConditioningClip(clip), role: 'audio' as const };

      for (const append of [false, true]) {
        expect(
          placeVideoImage({
            append,
            image: still('new.png'),
            models: [LTX2],
            videoValues: panel(LTX2, { conditioningClip: pictureClip }),
          })
        ).toMatchObject({ displaced: true, patch: { conditioningClip: null, firstFrameImage: still('new.png') } });

        const kept = placeVideoImage({
          append,
          image: still('new.png'),
          models: [LTX2],
          videoValues: panel(LTX2, { conditioningClip: soundtrack }),
        });

        expect(kept).toMatchObject({ displaced: false, patch: { firstFrameImage: still('new.png') } });
        expect(kept.status === 'placed' && 'conditioningClip' in kept.patch).toBe(false);
      }
    });

    it('clears a picture clip hidden behind the first frame when appending the last', () => {
      const pictureClip = { ...createVideoConditioningClip(clip), role: 'video' as const };

      // Normalization hides the stored clip behind the first frame; left stored, it would resurface beside the last
      // frame once the first is cleared. It was never on screen, so it isn't reported as displaced.
      expect(
        placeVideoImage({
          append: true,
          image: still('new.png'),
          models: [LTX2],
          videoValues: panel(LTX2, { conditioningClip: pictureClip, firstFrameImage: first }),
        })
      ).toEqual({
        displaced: false,
        patch: { conditioningClip: null, lastFrameImage: still('new.png'), references: [] },
        slot: 'lastFrame',
        status: 'placed',
      });
    });
  });

  it('declines, rather than switching models, when the panel model takes no images', () => {
    expect(
      placeVideoImage({
        append: false,
        image: still('new.png'),
        models: [WAN_T2V, WAN_I2V],
        videoValues: panel(WAN_T2V),
      })
    ).toEqual({ status: 'unsupported' });
  });
});

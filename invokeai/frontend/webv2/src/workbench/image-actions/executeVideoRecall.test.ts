import type { GalleryVideoItem } from '@features/gallery';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: { metadata: vi.fn(), resolveMany: vi.fn() },
  galleryItems: { resolve: vi.fn() },
  galleryVideos: { metadata: vi.fn() },
}));

vi.mock('@features/gallery', () => galleryApi);

import { createDefaultVideoWidgetValues } from '@features/video';

import { executeVideoRecall } from './executeVideoRecall';

const wanModel = {
  base: 'wan',
  file_size: 1,
  format: 'diffusers',
  hash: 'wan-hash',
  key: 'wan-t2v',
  name: 'Wan 2.2 t2v_a14b',
  path: '/models/wan',
  source: 'local',
  source_type: 'path',
  type: 'main',
  variant: 't2v_a14b',
} as ModelConfig;

const item: GalleryVideoItem = {
  boardId: 'none',
  category: 'general',
  createdAt: '2026-08-26T00:00:00.000Z',
  durationSeconds: 5,
  fullUrl: '/clip.mp4',
  height: 720,
  isIntermediate: false,
  kind: 'video',
  name: 'clip.mp4',
  starred: false,
  thumbnailUrl: '/clip-thumb.jpg',
  width: 1280,
};

const metadata = {
  cfg_scale: 5,
  generation_mode: 'wan_t2v',
  height: 720,
  model: { base: 'wan', key: wanModel.key, name: wanModel.name, type: 'main' },
  negative_prompt: 'blurry',
  num_frames: 81,
  positive_prompt: 'a fox running',
  seed: 4321,
  steps: 40,
  width: 1280,
};

const h3Model = {
  base: 'minimax-h3',
  format: 'diffusers',
  key: 'h3-main',
  name: 'MiniMax H3',
  path: '/models/h3',
  type: 'main',
  variant: 'ref2va',
} as unknown as ModelConfig;

/** A wrapped audio upload, as the gallery hands one back. */
const wrappedAudioItem: GalleryVideoItem = {
  ...item,
  durationSeconds: 6,
  fps: 24,
  mediaOrigin: 'audio_upload',
  name: 'song.mp4',
};

const h3ReferenceMetadata = (conditioning: unknown) => ({
  generation_mode: 'minimax_h3_ref2v',
  height: 768,
  minimax_h3_references: [
    {
      end_frame: 47,
      kind: 'video',
      start_frame: 2,
      video_name: 'song.mp4',
      ...(conditioning === undefined ? {} : { conditioning }),
    },
  ],
  model: { base: 'minimax-h3', key: 'h3-main', name: 'MiniMax H3', type: 'main' },
  num_frames: 124,
  positive_prompt: 'a red fox',
  seed: 7,
  steps: 4,
  width: 1344,
});

// `executeVideoRecall` memoizes metadata per video name, so each case recalls a distinct
// output video rather than re-reading a cached answer.
let recallSeq = 0;

const recallReferences = async (conditioning: unknown) => {
  recallSeq += 1;
  const recalledItem: GalleryVideoItem = { ...item, name: `output-${recallSeq}.mp4` };

  galleryApi.galleryVideos.metadata.mockResolvedValue(h3ReferenceMetadata(conditioning));
  galleryApi.galleryItems.resolve.mockResolvedValue(wrappedAudioItem);
  const patchValues = vi.fn();
  const commands = {
    notifications: { add: vi.fn(), reportError: vi.fn() } as unknown as WorkbenchCommands['notifications'],
    widgets: { patchValues } as unknown as WorkbenchCommands['widgets'],
  };

  await executeVideoRecall({
    commands,
    getVideoValues: () => createDefaultVideoWidgetValues([h3Model]) as unknown as Record<string, unknown>,
    item: recalledItem,
    kind: 'all',
    models: [h3Model],
  });

  const written = patchValues.mock.calls.at(-1)?.[1] as { references?: { conditioning?: string }[] } | undefined;

  return written?.references;
};

const ltx2Model = {
  base: 'ltx-2',
  format: 'checkpoint',
  key: 'ltx2-main',
  name: 'LTX-2.5 dev',
  path: '/models/ltx2',
  type: 'main',
  variant: 'ltx2_dev',
} as unknown as ModelConfig;

const ltx2ConditioningMetadata = (role: unknown, videoName: unknown | null = 'song.mp4') => ({
  generation_mode: role === 'audio' ? 'ltx2_a2v' : 'ltx2_v2a',
  height: 704,
  ltx2_conditioning_role: role,
  ...(videoName === null ? {} : { ltx2_conditioning_video: { video_name: videoName } }),
  model: { base: 'ltx-2', key: 'ltx2-main', name: 'LTX-2.5 dev', type: 'main' },
  num_frames: 89,
  positive_prompt: 'a red fox',
  seed: 7,
  steps: 30,
  width: 1248,
});

/** Recalls an LTX-2 conditioned run onto a panel that starts with `startingValues`. */
const recallConditioningClip = async (
  role: unknown,
  { startingValues, videoName }: { startingValues?: Record<string, unknown>; videoName?: unknown | null } = {}
) => {
  recallSeq += 1;

  galleryApi.galleryVideos.metadata.mockResolvedValue(ltx2ConditioningMetadata(role, videoName));
  // An ordinary gallery clip: `getDefaultConditioningRole` would take it for the VIDEO role, so a
  // recalled `audio` role can only come from the metadata.
  galleryApi.galleryItems.resolve.mockResolvedValue({ ...item, durationSeconds: 4, fps: 24, name: 'song.mp4' });
  const { commands, patchValues } = createCommands();

  await executeVideoRecall({
    commands,
    getVideoValues: () => ({
      ...(createDefaultVideoWidgetValues([ltx2Model]) as unknown as Record<string, unknown>),
      ...startingValues,
    }),
    item: { ...item, name: `ltx2-output-${recallSeq}.mp4` },
    kind: 'all',
    models: [ltx2Model],
  });

  return patchValues.mock.calls.at(-1)?.[1] as Record<string, unknown> | undefined;
};

const createCommands = () => {
  const patchValues = vi.fn();
  const commands = {
    notifications: { add: vi.fn(), reportError: vi.fn() } as unknown as WorkbenchCommands['notifications'],
    widgets: { patchValues } as unknown as WorkbenchCommands['widgets'],
  };

  return { commands, patchValues };
};

describe('executeVideoRecall', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryVideos.metadata.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
  });

  it('writes only the prompt keys for a prompts-only recall', async () => {
    galleryApi.galleryVideos.metadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();
    // A prompts-only recall must not persist the model-family transition produced by resnapshotting an uninstalled
    // main.
    const videoValues = { ...createDefaultVideoWidgetValues([wanModel]), numFrames: 41, steps: 4 };

    const didRecall = await executeVideoRecall({
      commands,
      getVideoValues: () => videoValues as unknown as Record<string, unknown>,
      item,
      kind: 'prompts',
      models: [],
    });

    expect(didRecall).toBe(true);
    expect(patchValues).toHaveBeenCalledTimes(1);
    expect(patchValues).toHaveBeenCalledWith(
      'video',
      { negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a fox running' },
      undefined
    );
  });

  it('writes the whole values object when the recall carries more than prompts', async () => {
    galleryApi.galleryVideos.metadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();
    const videoValues = createDefaultVideoWidgetValues([wanModel]);

    const didRecall = await executeVideoRecall({
      commands,
      getVideoValues: () => videoValues as unknown as Record<string, unknown>,
      item,
      kind: 'all',
      models: [wanModel],
    });

    expect(didRecall).toBe(true);
    expect(patchValues).toHaveBeenCalledTimes(1);
    // Prompts ride along with everything else — they are ordinary video values.
    expect(patchValues.mock.calls[0]?.[1]).toMatchObject({
      negativePrompt: 'blurry',
      positivePrompt: 'a fox running',
      seed: 4321,
    });
  });
});

describe('the LTX-2 keyframe and extend modes on recall', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryVideos.metadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  const recallLtx2 = async (metadata: Record<string, unknown>) => {
    recallSeq += 1;
    galleryApi.galleryVideos.metadata.mockResolvedValue({
      height: 704,
      model: { base: 'ltx-2', key: 'ltx2-main', name: 'LTX-2.5 dev', type: 'main' },
      num_frames: 121,
      positive_prompt: 'a red fox',
      seed: 7,
      steps: 30,
      width: 1248,
      ...metadata,
    });
    galleryApi.galleryItems.resolve.mockResolvedValue({ ...item, durationSeconds: 4, fps: 24, name: 'source.mp4' });
    galleryApi.galleryImages.resolveMany.mockResolvedValue([{ height: 704, imageName: 'last.png', width: 1248 }]);
    const { commands, patchValues } = createCommands();

    await executeVideoRecall({
      commands,
      getVideoValues: () => createDefaultVideoWidgetValues([ltx2Model]) as unknown as Record<string, unknown>,
      item: { ...item, name: `ltx2-recall-${recallSeq}.mp4` },
      kind: 'all',
      models: [ltx2Model],
    });

    return patchValues.mock.calls.at(-1)?.[1] as Record<string, unknown> | undefined;
  };

  // Recognising the mode string is only the first half: an unlisted one is refused outright, but a
  // listed one whose media never comes back is a silent downgrade to text-to-video.
  it('restores the destination frame a last-frame run ended on', async () => {
    expect(
      await recallLtx2({ generation_mode: 'ltx2_lf2v', last_frame_image: { image_name: 'last.png' } })
    ).toMatchObject({ lastFrameImage: { image_name: 'last.png' } });
  });

  it('restores the clip an extension continued, with the trim it ran at', async () => {
    const written = await recallLtx2({
      generation_mode: 'ltx2_extend_video',
      source_video: { video_name: 'source.mp4' },
      source_video_end_frame: 61,
      source_video_start_frame: 12,
    });

    // The trim is what decides WHERE the continuation picked up; the default would continue from
    // somewhere else entirely.
    expect(written).toMatchObject({
      sourceVideo: { endFrame: 61, startFrame: 12, video_name: 'source.mp4' },
    });
  });
});

describe('the LTX-2 conditioning clip on recall', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryVideos.metadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
  });

  // The role IS the mode: recalling an audio-to-video run as video-to-audio reproduces the
  // opposite generation. And the clip it is read from is an ordinary gallery video, whose own
  // default role is the other one -- so a dropped role does not fail loudly, it inverts.
  it('restores the recorded role rather than the one a fresh drop would default to', async () => {
    expect(await recallConditioningClip('audio')).toMatchObject({
      conditioningClip: { clip: { video_name: 'song.mp4' }, role: 'audio' },
    });
    expect(await recallConditioningClip('video')).toMatchObject({
      conditioningClip: { clip: { video_name: 'song.mp4' }, role: 'video' },
    });
  });

  it('clears the other conditioning slots the recalled run did not use', async () => {
    const written = await recallConditioningClip('audio', {
      startingValues: {
        firstFrameImage: { height: 704, image_name: 'held.png', width: 1248 },
        sourceVideo: {
          endFrame: 40,
          fps: 24,
          height: 704,
          numFrames: 41,
          startFrame: 0,
          video_name: 'old.mp4',
          width: 1248,
        },
      },
    });

    expect(written).toMatchObject({ firstFrameImage: null, references: [], sourceVideo: null });
  });

  it('clears a clip the panel is holding when the recalled run used none', async () => {
    galleryApi.galleryVideos.metadata.mockResolvedValue({
      ...ltx2ConditioningMetadata('audio'),
      ltx2_conditioning_video: undefined,
      ltx2_conditioning_role: undefined,
    });
    galleryApi.galleryItems.resolve.mockResolvedValue({ ...item, durationSeconds: 4, fps: 24, name: 'song.mp4' });
    const { commands, patchValues } = createCommands();

    await executeVideoRecall({
      commands,
      getVideoValues: () => ({
        ...(createDefaultVideoWidgetValues([ltx2Model]) as unknown as Record<string, unknown>),
        conditioningClip: {
          clip: { fps: 24, height: 704, numFrames: 96, video_name: 'stale.mp4', width: 1248 },
          fpsKnown: true,
          role: 'audio',
        },
      }),
      item: { ...item, name: 'ltx2-output-cleared.mp4' },
      kind: 'all',
      models: [ltx2Model],
    });

    expect(patchValues.mock.calls.at(-1)?.[1]).toMatchObject({ conditioningClip: null });
  });

  it('drops a half-recorded clip rather than guessing the missing half', async () => {
    expect(await recallConditioningClip(undefined)).toMatchObject({ conditioningClip: null });
    expect(await recallConditioningClip('audio', { videoName: null })).toMatchObject({ conditioningClip: null });
    expect(await recallConditioningClip('nonsense')).toMatchObject({ conditioningClip: null });
  });
});

describe('ref2va reference conditioning on recall', () => {
  beforeEach(() => {
    accountLifecycle.activate('test-account');
    galleryApi.galleryVideos.metadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
  });

  // Preserve all recorded conditioning modes, including video_audio on wrapped audio clips.
  it('keeps a recorded video_audio, even on a clip marked as a wrapped audio upload', async () => {
    expect(await recallReferences('video_audio')).toMatchObject([{ conditioning: 'video_audio' }]);
  });

  it('keeps a recorded video and a recorded audio', async () => {
    expect(await recallReferences('video')).toMatchObject([{ conditioning: 'video' }]);
    expect(await recallReferences('audio')).toMatchObject([{ conditioning: 'audio' }]);
  });

  it('falls back to the clip default only when nothing usable was recorded', async () => {
    expect(await recallReferences(undefined)).toMatchObject([{ conditioning: 'audio' }]);
    expect(await recallReferences(42)).toMatchObject([{ conditioning: 'audio' }]);
    expect(await recallReferences('nonsense')).toMatchObject([{ conditioning: 'audio' }]);
  });
});

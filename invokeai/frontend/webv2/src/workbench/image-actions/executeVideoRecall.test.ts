import type { GalleryVideoItem } from '@features/gallery';
import type * as GalleryBackendModule from '@features/gallery/data/backend';
import type { ModelConfig } from '@features/models';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { accountLifecycle } from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';
import { QueryClient } from '@tanstack/react-query';
import { createInstance } from 'i18next';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const galleryApi = vi.hoisted(() => ({
  galleryImages: { metadata: vi.fn(), resolveMany: vi.fn() },
  galleryItems: { resolve: vi.fn() },
}));
const videoMetadata = vi.hoisted(() => vi.fn());

vi.mock('@features/gallery', () => galleryApi);
// The transport under the gallery's shared metadata query.
vi.mock('@features/gallery/data/backend', async (importOriginal) => ({
  ...(await importOriginal<typeof GalleryBackendModule>()),
  getGalleryVideoMetadata: videoMetadata,
}));

import { galleryVideoMetadataOptions } from '@features/gallery/queries';
import { createDefaultVideoWidgetValues } from '@features/video';

import { applyVideoRecallMetadata, executeVideoRecall } from './executeVideoRecall';

const t = ((key: string) => key) as unknown as TFunction;

const englishCatalog = Object.values(
  import.meta.glob('../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;

/** The English copy, for asserting what a notice shows rather than which key it asked for. */
const createEnglishTranslator = async (): Promise<TFunction> => {
  const instance = createInstance();

  await instance.init({
    initAsync: false,
    interpolation: { escapeValue: false },
    lng: 'en',
    resources: { en: { translation: englishCatalog } },
  });

  return ((key: string, values?: Record<string, unknown>) => instance.t(key, values)) as TFunction;
};

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

// The shared metadata entry answers once per video name, so each case recalls a distinct output video rather than
// re-reading a cached answer.
let recallSeq = 0;
let queryClient: QueryClient;

const recallReferences = async (conditioning: unknown) => {
  recallSeq += 1;
  const recalledItem: GalleryVideoItem = { ...item, name: `output-${recallSeq}.mp4` };

  videoMetadata.mockResolvedValue(h3ReferenceMetadata(conditioning));
  galleryApi.galleryItems.resolve.mockResolvedValue(wrappedAudioItem);
  const patchValues = vi.fn();
  const commands = {
    notifications: { add: vi.fn(), reportError: vi.fn() } as unknown as WorkbenchCommands['notifications'],
    widgets: { patchValues } as unknown as WorkbenchCommands['widgets'],
  };

  await executeVideoRecall({
    queryClient,
    t,
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
  {
    recorded,
    startingValues,
    videoName,
  }: { recorded?: Record<string, unknown>; startingValues?: Record<string, unknown>; videoName?: unknown | null } = {}
) => {
  recallSeq += 1;

  videoMetadata.mockResolvedValue({ ...ltx2ConditioningMetadata(role, videoName), ...recorded });
  // An ordinary gallery clip: `getDefaultConditioningRole` would take it for the VIDEO role, so a
  // recalled `audio` role can only come from the metadata.
  galleryApi.galleryItems.resolve.mockResolvedValue({ ...item, durationSeconds: 4, fps: 24, name: 'song.mp4' });
  const { commands, patchValues } = createCommands();

  await executeVideoRecall({
    queryClient,
    t,
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
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
  });

  it('writes only the prompt keys for a prompts-only recall', async () => {
    videoMetadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();
    // A prompts-only recall must not persist the model-family transition produced by resnapshotting an uninstalled
    // main.
    const videoValues = { ...createDefaultVideoWidgetValues([wanModel]), numFrames: 41, steps: 4 };

    const didRecall = await executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues: () => videoValues as unknown as Record<string, unknown>,
      item,
      kind: 'prompts',
      // The panel's Wan main is not installed; only H3 is.
      models: [h3Model],
    });

    expect(didRecall).toBe(true);
    expect(patchValues).toHaveBeenCalledTimes(1);
    expect(patchValues).toHaveBeenCalledWith(
      'video',
      { negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a fox running' },
      undefined
    );
  });

  it('commits only the prompt keys of a prompts recall while the model catalog is empty', async () => {
    videoMetadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();

    const didRecall = await executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
      item,
      kind: 'prompts',
      models: [],
    });

    expect(didRecall).toBe(true);
    expect(patchValues).toHaveBeenCalledExactlyOnceWith(
      'video',
      { negativePrompt: 'blurry', negativePromptEnabled: true, positivePrompt: 'a fox running' },
      undefined
    );
  });

  it.each(['all', 'remix', 'seed', 'size'] as const)(
    'writes nothing for %s while the model catalog is empty',
    async (kind) => {
      videoMetadata.mockResolvedValue(metadata);
      const { commands, patchValues } = createCommands();

      const didRecall = await executeVideoRecall({
        queryClient,
        t,
        commands,
        getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
        item,
        kind,
        models: [],
      });

      expect(didRecall).toBe(false);
      expect(patchValues).not.toHaveBeenCalled();
      expect(commands.notifications.add).toHaveBeenCalledExactlyOnceWith({
        kind: 'info',
        message: 'widgets.gallery.itemActions.recall.modelsUnavailableForVideoRecall',
        title: 'widgets.gallery.itemActions.recall.cannotRecallVideo',
      });
    }
  );

  it('writes the whole values object when the recall carries more than prompts', async () => {
    videoMetadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();
    const videoValues = createDefaultVideoWidgetValues([wanModel]);

    const didRecall = await executeVideoRecall({
      queryClient,
      t,
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

  it('recalls only the size onto the panel model, reading none of the media the record names', async () => {
    videoMetadata.mockResolvedValue({
      ...metadata,
      first_frame_image: { image_name: 'frame.png' },
      height: 1280,
      model: { base: 'minimax-h3', key: 'h3-main', name: 'MiniMax H3', type: 'main' },
      width: 720,
    });
    const { commands, patchValues } = createCommands();
    const videoValues = createDefaultVideoWidgetValues([wanModel]);

    const didRecall = await executeVideoRecall({
      queryClient,
      t: await createEnglishTranslator(),
      commands,
      getVideoValues: () => videoValues as unknown as Record<string, unknown>,
      item,
      kind: 'size',
      models: [wanModel, h3Model],
    });

    expect(didRecall).toBe(true);
    expect(galleryApi.galleryImages.resolveMany).not.toHaveBeenCalled();
    expect(patchValues).toHaveBeenCalledWith(
      'video',
      { ...videoValues, aspectRatioId: '9:16', targetResolution: '720p' },
      undefined
    );
    expect(commands.notifications.add).toHaveBeenCalledWith({
      kind: 'success',
      message: 'Applied size.',
      title: 'Recalled size',
    });
  });

  it('says the recorded size has no preset on the panel model, not that the metadata is missing', async () => {
    videoMetadata.mockResolvedValue({ ...metadata, height: 464, width: 848 });
    const { commands, patchValues } = createCommands();

    const didRecall = await executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
      item,
      kind: 'size',
      models: [wanModel],
    });

    expect(didRecall).toBe(false);
    expect(patchValues).not.toHaveBeenCalled();
    expect(commands.notifications.add).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: 'widgets.gallery.itemActions.recall.videoSizeUnavailable',
      title: 'widgets.gallery.itemActions.recall.nothingRecalledVideoSize',
    });
  });

  it('names the panel media when it has no usable frame on the model, and shows a refused run its reason', async () => {
    const recallSize = async (model: ModelConfig, values: Record<string, unknown>, width: number, height: number) => {
      videoMetadata.mockResolvedValue({ ...metadata, height, width });
      queryClient.clear();
      const { commands, patchValues } = createCommands();

      await executeVideoRecall({
        queryClient,
        t,
        commands,
        getVideoValues: () => ({ ...createDefaultVideoWidgetValues([model]), ...values }),
        item,
        kind: 'size',
        models: [model],
      });
      expect(patchValues).not.toHaveBeenCalled();

      return commands.notifications.add;
    };

    const narrowFrame = { firstFrameImage: { height: 500, image_name: 'held.png', width: 100 } };
    const h3Checkpoint = { ...h3Model, format: 'checkpoint' } as ModelConfig;
    expect(await recallSize(h3Checkpoint, narrowFrame, 1344, 768)).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: 'widgets.gallery.itemActions.recall.videoSizeMediaUnresolvable',
      title: 'widgets.gallery.itemActions.recall.nothingRecalledVideoSize',
    });

    const audioClip = {
      conditioningClip: {
        clip: { fps: 24, height: 1024, numFrames: 96, video_name: 'song.mp4', width: 1792 },
        fpsKnown: true,
        role: 'audio',
      },
    };
    expect(
      await recallSize(ltx2Model, { ...audioClip, targetResolution: '720p' }, 1792, 1024)
    ).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: expect.stringContaining('does not run a two-stage target resolution'),
      title: 'widgets.gallery.itemActions.recall.nothingRecalledVideoSize',
    });
  });

  it('keeps the unsupported-metadata notice for a size key on a record that is not video generation', async () => {
    videoMetadata.mockResolvedValue({ generation_mode: 'sdxl_txt2img', height: 720, width: 1280 });
    const { commands } = createCommands();

    await executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
      item,
      kind: 'size',
      models: [wanModel],
    });

    expect(commands.notifications.add).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: 'widgets.gallery.itemActions.recall.videoMetadataUnsupported',
      title: 'widgets.gallery.itemActions.recall.nothingRecalledVideo',
    });
  });

  it('says media in the panel overrules the recorded size, and leaves the panel alone', async () => {
    videoMetadata.mockResolvedValue({ ...metadata, height: 1280, width: 720 });
    const { commands, patchValues } = createCommands();
    const message = vi.fn((key: string) => key);
    const videoValues = {
      ...createDefaultVideoWidgetValues([wanModel]),
      firstFrameImage: { height: 720, image_name: 'held.png', width: 1280 },
    };

    const didRecall = await executeVideoRecall({
      queryClient,
      t: message as unknown as TFunction,
      commands,
      getVideoValues: () => videoValues as unknown as Record<string, unknown>,
      item,
      kind: 'size',
      models: [wanModel],
    });

    expect(didRecall).toBe(false);
    expect(patchValues).not.toHaveBeenCalled();
    expect(message).toHaveBeenCalledWith('widgets.gallery.itemActions.recall.videoSizeMediaConflict', {
      effectiveHeight: 720,
      effectiveWidth: 1280,
      height: 1280,
      width: 720,
    });
    expect(commands.notifications.add).toHaveBeenCalledExactlyOnceWith({
      kind: 'info',
      message: 'widgets.gallery.itemActions.recall.videoSizeMediaConflict',
      title: 'widgets.gallery.itemActions.recall.nothingRecalledVideoSize',
    });
  });

  it('recalls from the metadata the menu already read, without reading it again', async () => {
    videoMetadata.mockResolvedValue(metadata);
    const { commands, patchValues } = createCommands();

    // The menu derives its recall verbs from the same entry before the user picks one.
    await queryClient.fetchQuery(galleryVideoMetadataOptions(accountLifecycle.capture(), item.name));
    const didRecall = await executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
      item,
      kind: 'prompts',
      models: [wanModel],
    });

    expect(didRecall).toBe(true);
    expect(videoMetadata).toHaveBeenCalledOnce();
    expect(patchValues).toHaveBeenCalledWith(
      'video',
      expect.objectContaining({ positivePrompt: 'a fox running' }),
      undefined
    );
  });

  it('reports a failed read and reads again on the next recall', async () => {
    const { commands, patchValues } = createCommands();
    const recall = () =>
      executeVideoRecall({
        queryClient,
        t,
        commands,
        getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
        item,
        kind: 'prompts',
        models: [wanModel],
      });

    videoMetadata.mockRejectedValueOnce(new ApiError('Internal Server Error', 500));
    await expect(recall()).resolves.toBe(false);
    expect(commands.notifications.reportError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'video-recall', message: 'Internal Server Error' })
    );

    videoMetadata.mockResolvedValueOnce(metadata);
    await expect(recall()).resolves.toBe(true);
    expect(videoMetadata).toHaveBeenCalledTimes(2);
    expect(patchValues).toHaveBeenCalledOnce();
  });
});

describe('the target resolution of a canvas the recorded media sized', () => {
  const wanI2v = { ...wanModel, key: 'wan-i2v', name: 'Wan 2.2 i2v_a14b', variant: 'i2v_a14b' } as ModelConfig;
  const h3Fl2va = { ...h3Model, key: 'h3-fl2va', name: 'MiniMax H3 FL2VA', variant: 'fl2va' } as ModelConfig;

  beforeEach(() => {
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  /** Recalls a first-frame run onto a panel on the same model, the gallery holding `frames`. */
  const recallFirstFrame = async (
    model: ModelConfig,
    panel: Record<string, unknown>,
    recorded: Record<string, unknown>,
    frames: { height: number; imageName: string; width: number }[]
  ) => {
    recallSeq += 1;
    videoMetadata.mockResolvedValue({
      first_frame_image: { image_name: 'first.png' },
      model: { base: model.base, key: model.key, name: model.name, type: 'main' },
      num_frames: 81,
      positive_prompt: 'a red fox',
      seed: 7,
      ...recorded,
    });
    galleryApi.galleryImages.resolveMany.mockResolvedValue(frames);
    const { commands, patchValues } = createCommands();

    await executeVideoRecall({
      queryClient,
      t: await createEnglishTranslator(),
      commands,
      getVideoValues: () => ({ ...createDefaultVideoWidgetValues([model]), ...panel }),
      item: { ...item, name: `media-sized-${recallSeq}.mp4` },
      kind: 'all',
      models: [model],
    });

    const [notice] = vi.mocked(commands.notifications.add).mock.calls.at(-1) ?? [];

    return { message: notice?.message, values: patchValues.mock.calls.at(-1)?.[1] as Record<string, unknown> };
  };

  it.each([
    // 1800x1000 at 768 highres: over the area cap, scaled to 1363.1x757.3 and snapped to 1376x768. The canvas fed
    // back would give 1344x768, and 768 lowres gives the frame 768x416.
    ['an area-capped H3 canvas', h3Fl2va, 'minimax_h3_i2v', 1376, 768, 1800, 1000, '768 lowres', '768 highres'],
    // 1000x700 at 1080p: 1542.9x1080, snapped to 1536x1088. The canvas fed back would give 1520x1088.
    ['a Wan 1080p canvas', wanI2v, 'wan_i2v', 1536, 1088, 1000, 700, '720p', '1080p'],
  ] as const)(
    'restores the target resolution of %s from the frame as the gallery returns it',
    async (_, model, generationMode, width, height, frameWidth, frameHeight, panelResolution, recalled) => {
      const { message, values } = await recallFirstFrame(
        model,
        { targetResolution: panelResolution },
        { generation_mode: generationMode, height, width },
        [{ height: frameHeight, imageName: 'first.png', width: frameWidth }]
      );

      expect(values).toMatchObject({
        firstFrameImage: { height: frameHeight, image_name: 'first.png', width: frameWidth },
        targetResolution: recalled,
      });
      expect(message).toContain('size');
    }
  );

  it('leaves the panel resolution, and claims no size, when the frame is gone or reaches the canvas nowhere', async () => {
    // 688x480 is the frame's 480p canvas, which fed back would give itself: only the frame missing refuses it. A 16:9
    // frame gives Wan 848x480, 1280x720 or 1920x1088, never 1536x1088.
    for (const [size, frames] of [
      [{ height: 480, width: 688 }, []],
      [{ height: 1088, width: 1536 }, [{ height: 720, imageName: 'first.png', width: 1280 }]],
    ] as const) {
      const { message, values } = await recallFirstFrame(
        wanI2v,
        { targetResolution: '720p' },
        { generation_mode: 'wan_i2v', ...size },
        [...frames]
      );

      expect(values.targetResolution).toBe('720p');
      expect(message).not.toContain('size');
    }
  });
});

describe('the LTX-2 keyframe and extend modes on recall', () => {
  beforeEach(() => {
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  const recallLtx2 = async (metadata: Record<string, unknown>) => {
    recallSeq += 1;
    videoMetadata.mockResolvedValue({
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
      queryClient,
      t,
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
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
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

  it("restores the frames that anchored a soundtrack's picture, and only for a soundtrack", async () => {
    const recorded = {
      first_frame_image: { image_name: 'first.png' },
      last_frame_image: { image_name: 'last.png' },
    };
    galleryApi.galleryImages.resolveMany.mockResolvedValue([
      { height: 704, imageName: 'first.png', width: 1248 },
      { height: 704, imageName: 'last.png', width: 1248 },
    ]);

    expect(await recallConditioningClip('audio', { recorded })).toMatchObject({
      conditioningClip: { role: 'audio' },
      firstFrameImage: { image_name: 'first.png' },
      lastFrameImage: { image_name: 'last.png' },
    });
    // A picture-role run cannot have held frames; a record claiming both keeps the clip, not them.
    expect(await recallConditioningClip('video', { recorded })).toMatchObject({
      conditioningClip: { role: 'video' },
      firstFrameImage: null,
      lastFrameImage: null,
    });
  });

  it('clears a clip the panel is holding when the recalled run used none', async () => {
    videoMetadata.mockResolvedValue({
      ...ltx2ConditioningMetadata('audio'),
      ltx2_conditioning_video: undefined,
      ltx2_conditioning_role: undefined,
    });
    galleryApi.galleryItems.resolve.mockResolvedValue({ ...item, durationSeconds: 4, fps: 24, name: 'song.mp4' });
    const { commands, patchValues } = createCommands();

    await executeVideoRecall({
      queryClient,
      t,
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
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
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

describe('reading the media a recall names', () => {
  beforeEach(() => {
    queryClient = new QueryClient();
    accountLifecycle.activate('test-account');
    videoMetadata.mockReset();
    galleryApi.galleryItems.resolve.mockReset();
    galleryApi.galleryImages.resolveMany.mockReset();
  });

  const videoReference = (name: string) => ({ conditioning: 'video', kind: 'video', video_name: name });
  const galleryVideo = (name: string): GalleryVideoItem => ({ ...item, durationSeconds: 4, fps: 24, name });

  const recallH3 = (recorded: Record<string, unknown>, getVideoValues?: () => Record<string, unknown>) => {
    recallSeq += 1;
    videoMetadata.mockResolvedValue({ ...h3ReferenceMetadata('video'), ...recorded });
    const { commands, patchValues } = createCommands();
    const recall = executeVideoRecall({
      queryClient,
      t,
      commands,
      getVideoValues:
        getVideoValues ?? (() => createDefaultVideoWidgetValues([h3Model]) as unknown as Record<string, unknown>),
      item: { ...item, name: `h3-output-${recallSeq}.mp4` },
      kind: 'all',
      models: [h3Model],
    });

    return {
      commands,
      recall,
      written: () => patchValues.mock.calls.at(-1)?.[1] as Record<string, unknown> | undefined,
    };
  };

  it('reads a clip that is both the source video and a reference once, and lands it in both slots', async () => {
    galleryApi.galleryItems.resolve.mockImplementation(({ name }: { name: string }) =>
      Promise.resolve(galleryVideo(name))
    );

    const { recall, written } = recallH3({
      minimax_h3_references: [videoReference('long.mp4')],
      source_video: { video_name: 'long.mp4' },
    });

    await expect(recall).resolves.toBe(true);
    expect(galleryApi.galleryItems.resolve).toHaveBeenCalledOnce();
    expect(written()).toMatchObject({
      references: [{ clip: { video_name: 'long.mp4' }, kind: 'video' }],
      sourceVideo: { video_name: 'long.mp4' },
    });
  });

  it('keeps at most four video reads in flight', async () => {
    const pending: { name: string; resolve: () => void }[] = [];
    let inFlight = 0;
    let maxInFlight = 0;

    galleryApi.galleryItems.resolve.mockImplementation(
      ({ name }: { name: string }) =>
        new Promise((resolve) => {
          inFlight += 1;
          maxInFlight = Math.max(maxInFlight, inFlight);
          pending.push({
            name,
            resolve: () => {
              inFlight -= 1;
              resolve(galleryVideo(name));
            },
          });
        })
    );

    const names = ['a.mp4', 'b.mp4', 'c.mp4', 'd.mp4', 'e.mp4', 'f.mp4'];
    const { recall } = recallH3({ minimax_h3_references: names.map(videoReference) });

    await vi.waitFor(() => expect(pending).toHaveLength(4));
    expect(galleryApi.galleryItems.resolve).toHaveBeenCalledTimes(4);
    for (let answered = 0; answered < names.length; answered += 1) {
      await vi.waitFor(() => expect(pending.length).toBeGreaterThan(0));
      pending.shift()!.resolve();
    }

    await expect(recall).resolves.toBe(true);
    expect(maxInFlight).toBe(4);
  });

  it('drops a deleted video and applies the rest, but reports any other failed read and applies nothing', async () => {
    galleryApi.galleryItems.resolve.mockImplementation(({ name }: { name: string }) =>
      name === 'deleted.mp4' ? Promise.reject(new ApiError('Not Found', 404)) : Promise.resolve(galleryVideo(name))
    );

    const deleted = recallH3({ minimax_h3_references: [videoReference('deleted.mp4'), videoReference('kept.mp4')] });

    await expect(deleted.recall).resolves.toBe(true);
    expect(deleted.written()).toMatchObject({ references: [{ clip: { video_name: 'kept.mp4' } }] });
    expect(deleted.written()?.references).toHaveLength(1);

    galleryApi.galleryItems.resolve.mockImplementation(({ name }: { name: string }) =>
      name === 'broken.mp4'
        ? Promise.reject(new ApiError('Internal Server Error', 500))
        : Promise.resolve(galleryVideo(name))
    );

    const failed = recallH3({ minimax_h3_references: [videoReference('broken.mp4'), videoReference('kept.mp4')] });

    await expect(failed.recall).resolves.toBe(false);
    expect(failed.written()).toBeUndefined();
    expect(failed.commands.notifications.reportError).toHaveBeenCalledWith(
      expect.objectContaining({ area: 'video-recall', message: 'Internal Server Error' })
    );
  });

  it('drops a video on a board the caller can no longer read and applies the rest', async () => {
    galleryApi.galleryItems.resolve.mockImplementation(({ name }: { name: string }) =>
      name === 'unreadable.mp4' ? Promise.reject(new ApiError('Forbidden', 403)) : Promise.resolve(galleryVideo(name))
    );

    const { commands, recall, written } = recallH3({
      minimax_h3_references: [videoReference('unreadable.mp4'), videoReference('kept.mp4')],
    });

    await expect(recall).resolves.toBe(true);
    expect(written()).toMatchObject({ references: [{ clip: { video_name: 'kept.mp4' } }] });
    expect(written()?.references).toHaveLength(1);
    expect(commands.notifications.reportError).not.toHaveBeenCalled();
  });

  it('applies on top of an edit made while the media loaded', async () => {
    let finishRead: (() => void) | undefined;
    let panel = { ...createDefaultVideoWidgetValues([h3Model]), seedMode: 'random' } as Record<string, unknown>;
    const getVideoValues = () => panel;

    galleryApi.galleryItems.resolve.mockImplementation(
      ({ name }: { name: string }) =>
        new Promise((resolve) => {
          finishRead = () => resolve(galleryVideo(name));
        })
    );

    const { recall, written } = recallH3(
      { minimax_h3_references: [videoReference('kept.mp4')], seed: undefined },
      getVideoValues
    );

    await vi.waitFor(() => expect(finishRead).toBeDefined());
    panel = { ...panel, seed: 99, seedMode: 'fixed' };
    finishRead!();

    await expect(recall).resolves.toBe(true);
    expect(written()).toMatchObject({
      references: [{ clip: { video_name: 'kept.mp4' } }],
      seed: 99,
      seedMode: 'fixed',
    });
  });

  it("never reads a medium a partial recall would drop for the panel's model, so its failure cannot fail the recall", async () => {
    galleryApi.galleryItems.resolve.mockRejectedValue(new ApiError('Internal Server Error', 500));
    const { commands, patchValues } = createCommands();

    // No recorded model: the panel's text-to-video Wan stays, and it cannot extend a clip.
    const didRecall = await applyVideoRecallMetadata({
      commands,
      emptyNotice: { message: 'nothing', title: 'nothing' },
      getVideoValues: () => createDefaultVideoWidgetValues([wanModel]) as unknown as Record<string, unknown>,
      kind: 'all',
      metadata: { positive_prompt: 'a heron', source_video: { video_name: 'source.mp4' } },
      models: [wanModel],
      owner: accountLifecycle.capture(),
      partial: true,
      requireGenerationMode: false,
      t,
    });

    expect(didRecall).toBe(true);
    expect(galleryApi.galleryItems.resolve).not.toHaveBeenCalled();
    expect(commands.notifications.reportError).not.toHaveBeenCalled();
    // The dropped clip leaves only the prompt to apply.
    expect(patchValues.mock.calls.at(-1)?.[1]).toEqual(expect.objectContaining({ positivePrompt: 'a heron' }));
    expect(patchValues.mock.calls.at(-1)?.[1]).not.toHaveProperty('sourceVideo');
  });
});

import type { GalleryVideoItem } from '@features/gallery';
import type { ModelConfig } from '@features/models';
import type { VideoReferenceItem, VideoWidgetValues } from '@features/video';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { galleryImages, galleryItems, galleryVideos } from '@features/gallery';
import {
  createDefaultVideoWidgetValues,
  createVideoConditioningClip,
  createVideoReferenceEntry,
  createVideoSourceClip,
  getDefaultReferenceConditioning,
  getInitialVideoPatch,
  getReferencesPatch,
  getVideoModelPolicy,
  isVideoReferenceConditioning,
  normalizeVideoWidgetValues,
  syncVideoWidgetValuesWithModels,
} from '@features/video';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';

import {
  buildVideoRecallSettings,
  getVideoRecallMessage,
  getVideoRecallTitle,
  type VideoRecallKind,
} from './videoRecall';

const videoMetadataRequests = new Map<string, { owner: AccountScope; promise: Promise<unknown> }>();

registerAccountOwnedResource({
  clear: () => {
    videoMetadataRequests.clear();
  },
  name: 'video-recall-metadata',
});

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const loadVideoMetadata = (videoName: string, owner: AccountScope): Promise<unknown> => {
  const cachedRequest = videoMetadataRequests.get(videoName);

  if (cachedRequest?.owner === owner) {
    return cachedRequest.promise;
  }

  const request = galleryVideos
    .metadata(videoName, owner.signal)
    .then((metadata) => {
      assertAccountScopeCurrent(owner);

      return metadata;
    })
    .catch((error: unknown) => {
      if (videoMetadataRequests.get(videoName)?.promise === request) {
        videoMetadataRequests.delete(videoName);
      }
      throw error;
    });

  videoMetadataRequests.set(videoName, { owner, promise: request });
  return request;
};

export const getCurrentVideoValues = ({
  models,
  videoValues,
}: {
  models: readonly ModelConfig[];
  videoValues: Record<string, unknown>;
}) => {
  const normalized = normalizeVideoWidgetValues(videoValues) ?? createDefaultVideoWidgetValues(models);

  return syncVideoWidgetValuesWithModels(normalized, models);
};

interface VideoRecallNotice {
  message: string;
  title: string;
}

const reportVideoRecallError = (
  commands: Pick<WorkbenchCommands, 'notifications'>,
  owner: AccountScope,
  error: unknown,
  projectId: string | undefined
): false => {
  if (isAccountScopeCurrent(owner)) {
    commands.notifications.reportError({
      area: 'video-recall',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
    });
  }
  return false;
};

export const executeVideoRecall = async ({
  commands,
  getVideoValues,
  item,
  kind,
  models,
  owner: callerOwner,
  projectId,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  getVideoValues: () => Record<string, unknown>;
  item: GalleryVideoItem;
  kind: VideoRecallKind;
  models: ModelConfig[];
  /** Caller-captured identity lifetime; direct synchronous callers may omit it. */
  owner?: AccountScope;
  projectId?: string;
}): Promise<boolean> => {
  const owner = callerOwner ?? captureAccountScope();

  if (!isAccountScopeCurrent(owner)) {
    return false;
  }

  let metadata: unknown;

  try {
    metadata = await loadVideoMetadata(item.name, owner);
  } catch (error: unknown) {
    return reportVideoRecallError(commands, owner, error, projectId);
  }

  return applyVideoRecallMetadata({
    commands,
    emptyNotice: {
      message: 'This video does not include supported Video metadata.',
      title: 'No recallable video data',
    },
    getVideoValues,
    kind,
    metadata,
    models,
    owner,
    projectId,
  });
};

/**
 * Apply a video metadata record — read from a gallery video, or sent by the external recall API — to the Video
 * panel: build the values, hydrate the media it names from the gallery, drop media the resulting model cannot use,
 * and commit. Resolves whether anything was applied.
 */
export const applyVideoRecallMetadata = async ({
  commands,
  emptyNotice,
  getVideoValues,
  kind,
  metadata,
  models,
  owner,
  partial = false,
  projectId,
  requireGenerationMode = true,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  /** Shown when nothing in the record applies. */
  emptyNotice: VideoRecallNotice;
  getVideoValues: () => Record<string, unknown>;
  kind: VideoRecallKind;
  metadata: unknown;
  models: readonly ModelConfig[];
  owner: AccountScope;
  /** See `buildVideoRecallSettings`. */
  partial?: boolean;
  projectId?: string;
  /** See `buildVideoRecallSettings`. */
  requireGenerationMode?: boolean;
}): Promise<boolean> => {
  try {
    assertAccountScopeCurrent(owner);
    // Snapshot the panel AFTER the fetch: an edit made while the metadata
    // loaded must survive into the base the recall applies on top of.
    const currentValues = getCurrentVideoValues({ models, videoValues: getVideoValues() });
    const result = buildVideoRecallSettings({
      currentValues,
      kind,
      metadata,
      models,
      partial,
      requireGenerationMode,
    });

    if (!result) {
      commands.notifications.add({ kind: 'info', ...emptyNotice });
      return false;
    }

    // A partial recall names media on top of the panel's own. One the panel's model cannot use would still displace
    // the held media it conflicts with before the mode check below dropped it, so it is dropped up front instead.
    if (partial && result.values.model) {
      const policy = getVideoModelPolicy(result.values.model, result.values);
      const modes = policy.modes;
      const names = result.mediaNames;

      if (!modes.includes('reference')) {
        names.references = [];
      }
      if (!modes.includes('first-frame') && !modes.includes('first-last')) {
        names.firstFrameName = null;
      }
      if (!modes.includes('last-frame') && !modes.includes('first-last')) {
        names.lastFrameName = null;
      }
      if (!modes.includes('extend') && !policy.references?.extend) {
        names.sourceVideoName = null;
        names.sourceVideoTrim = null;
      }
      if (
        names.conditioningClip &&
        !modes.includes(names.conditioningClip.role === 'audio' ? 'audio-to-video' : 'video-to-audio')
      ) {
        names.conditioningClip = null;
      }
    }

    // Resolve media names against the gallery for dimensions/probe data and drop deleted references.
    if (result.fields.includes('media')) {
      // A full recall that cleared held media changed the panel even if nothing named below hydrates.
      const clearedMedia = (
        ['conditioningClip', 'firstFrameImage', 'lastFrameImage', 'references', 'sourceVideo'] as const
      ).some((slot) => result.values[slot] !== currentValues[slot]);
      let recalledMedia = false;
      const { firstFrameName, lastFrameName, sourceVideoName } = result.mediaNames;
      const frameNames = [firstFrameName, lastFrameName].filter((name): name is string => name !== null);
      const resolvedFrames =
        frameNames.length > 0 ? await galleryImages.resolveMany([...new Set(frameNames)], owner.signal) : [];

      assertAccountScopeCurrent(owner);
      const framesByName = new Map(resolvedFrames.map((frame) => [frame.imageName, frame]));
      const firstFrame = firstFrameName ? framesByName.get(firstFrameName) : undefined;
      const lastFrame = lastFrameName ? framesByName.get(lastFrameName) : undefined;

      if (firstFrame) {
        result.values = {
          ...result.values,
          firstFrameImage: { height: firstFrame.height, image_name: firstFrame.imageName, width: firstFrame.width },
          sourceVideo: null,
        };
        recalledMedia = true;
      }
      if (lastFrame) {
        result.values = {
          ...result.values,
          lastFrameImage: { height: lastFrame.height, image_name: lastFrame.imageName, width: lastFrame.width },
        };
        recalledMedia = true;
      }

      const { conditioningClip } = result.mediaNames;

      if (conditioningClip) {
        try {
          const clipItem = await galleryItems.resolve({ kind: 'video', name: conditioningClip.name }, owner.signal);

          assertAccountScopeCurrent(owner);
          if (clipItem?.kind === 'video') {
            // The recorded role, not the one a fresh drop would default to: the run held that
            // modality clean, and the other role is a different generation entirely.
            result.values = {
              ...result.values,
              conditioningClip: {
                ...createVideoConditioningClip({
                  durationSeconds: clipItem.durationSeconds,
                  fps: clipItem.fps,
                  height: clipItem.height,
                  name: clipItem.name,
                  width: clipItem.width,
                }),
                role: conditioningClip.role,
              },
              firstFrameImage: null,
              lastFrameImage: null,
              references: [],
              sourceVideo: null,
            };
            recalledMedia = true;
          }
        } catch {
          assertAccountScopeCurrent(owner);
          // The clip is gone; the rest of the recall still applies.
        }
      }

      if (sourceVideoName) {
        try {
          const sourceItem = await galleryItems.resolve({ kind: 'video', name: sourceVideoName }, owner.signal);

          assertAccountScopeCurrent(owner);
          if (sourceItem?.kind === 'video') {
            const rebuiltClip = createVideoSourceClip({
              durationSeconds: sourceItem.durationSeconds,
              fps: sourceItem.fps,
              height: sourceItem.height,
              name: sourceItem.name,
              width: sourceItem.width,
            });
            // Restore recorded trim against the fresh estimate; default trim would select different frames.
            const trim = rebuiltClip.numFrames >= 2 ? result.mediaNames.sourceVideoTrim : null;
            const startFrame = trim
              ? Math.min(Math.max(trim.startFrame, 0), rebuiltClip.numFrames - 2)
              : rebuiltClip.startFrame;
            const endFrame = trim
              ? Math.min(Math.max(trim.endFrame, startFrame + 1), rebuiltClip.numFrames - 1)
              : rebuiltClip.endFrame;

            result.values = {
              ...result.values,
              firstFrameImage: null,
              sourceVideo: { ...rebuiltClip, endFrame, startFrame },
            };
            recalledMedia = true;
          }
        } catch {
          assertAccountScopeCurrent(owner);
          // The source clip is gone; the rest of the recall still applies.
        }
      }

      if (result.mediaNames.references.length > 0) {
        // Hydrate references in recorded order, dropping deleted media without reordering survivors.
        const imageNames = result.mediaNames.references
          .filter((reference): reference is typeof reference & { kind: 'image' } => reference.kind === 'image')
          .map((reference) => reference.name);
        const resolvedImages =
          imageNames.length > 0 ? await galleryImages.resolveMany([...new Set(imageNames)], owner.signal) : [];

        assertAccountScopeCurrent(owner);
        const imagesByName = new Map(resolvedImages.map((image) => [image.imageName, image]));
        const references: VideoReferenceItem[] = [];

        for (const recorded of result.mediaNames.references) {
          if (recorded.kind === 'image') {
            const image = imagesByName.get(recorded.name);

            if (image) {
              references.push({
                detail: recorded.detail === 'match' ? 'match' : 'max',
                image: { height: image.height, image_name: image.imageName, width: image.width },
                kind: 'image',
              });
            }
            continue;
          }
          try {
            const item = await galleryItems.resolve({ kind: 'video', name: recorded.name }, owner.signal);

            assertAccountScopeCurrent(owner);
            if (item?.kind !== 'video') {
              continue;
            }
            const clip = createVideoSourceClip({
              durationSeconds: item.durationSeconds,
              fps: item.fps,
              height: item.height,
              name: item.name,
              width: item.width,
            });
            // Restore the recorded trim, clamped to the fresh frame-count estimate.
            const startFrame = recorded.trim ? Math.min(Math.max(recorded.trim.startFrame, 0), clip.numFrames - 1) : 0;
            const endFrame = recorded.trim
              ? Math.min(Math.max(recorded.trim.endFrame, startFrame), clip.numFrames - 1)
              : clip.numFrames - 1;
            // Any valid recorded conditioning wins; otherwise use add-path defaults, including soundtrack
            // conditioning for wrapped audio.
            const conditioning = isVideoReferenceConditioning(recorded.conditioning)
              ? recorded.conditioning
              : getDefaultReferenceConditioning(item.mediaOrigin);

            references.push({ clip: { ...clip, endFrame, startFrame }, conditioning, kind: 'video' });
          } catch {
            assertAccountScopeCurrent(owner);
            // The reference video is gone; the rest of the recall still applies.
          }
        }

        if (references.length > 0) {
          result.values = {
            ...result.values,
            firstFrameImage: null,
            lastFrameImage: null,
            references,
            // A source video recorded alongside references is reference-extend
            // state (hydrated above) — keep it; clear only a leftover. A partial
            // recall keeps the panel's own, subject to the mode check below.
            sourceVideo: result.mediaNames.sourceVideoName || partial ? result.values.sourceVideo : null,
          };
          recalledMedia = true;
        }
      }

      if (!recalledMedia && !clearedMedia) {
        result.fields = result.fields.filter((field) => field !== 'media');
      }

      // Reconcile hydrated media with the effective model's modes; deleted frames or unavailable models must not
      // leave an ungeneratable panel.
      const effectiveModel = result.values.model;

      if (effectiveModel) {
        const policy = getVideoModelPolicy(effectiveModel, result.values);
        const modes = policy.modes;
        const referenceExtend = Boolean(policy.references?.extend);

        // A partial recall can replace the references or the initial video alone, which on a reference-extend panel
        // would orphan the clip's continuity reference or leave it pointing at the previous clip. Relink it the way
        // the panel's Initial Video field does, dropping the clip when no reference slot is left for it.
        if (partial && referenceExtend && result.values.sourceVideo) {
          const linked = getInitialVideoPatch({
            maxVideos: policy.references?.maxVideos ?? 0,
            numFrames: result.values.numFrames,
            referenceExtend,
            references: result.values.references,
            sourceVideo: result.values.sourceVideo,
          });

          result.values = linked ? { ...result.values, ...linked } : { ...result.values, sourceVideo: null };
        }

        let { firstFrameImage, lastFrameImage, sourceVideo } = result.values;
        let references = result.values.references;

        if (references.length > 0 && !modes.includes('reference')) {
          // The recalled transformer (or the panel's surviving one) has no reference mode.
          references = [];
        }
        if (references.length > 0) {
          // References replace the frame slots; the source video survives only
          // on a reference-extend panel (the new clip is appended to it).
          firstFrameImage = null;
          lastFrameImage = null;
          if (!referenceExtend) {
            sourceVideo = null;
          }
        }

        if (sourceVideo && !modes.includes('extend') && !(references.length > 0 && referenceExtend)) {
          sourceVideo = null;
        }
        if (firstFrameImage && !modes.includes('first-frame') && !modes.includes('first-last')) {
          firstFrameImage = null;
        }
        if (lastFrameImage) {
          const lastFrameSupported =
            firstFrameImage || sourceVideo ? modes.includes('first-last') : modes.includes('last-frame');

          if (!lastFrameSupported) {
            lastFrameImage = null;
          }
        }

        if (
          firstFrameImage !== result.values.firstFrameImage ||
          lastFrameImage !== result.values.lastFrameImage ||
          sourceVideo !== result.values.sourceVideo ||
          references !== result.values.references
        ) {
          result.values = { ...result.values, firstFrameImage, lastFrameImage, references, sourceVideo };
        }
      }
    }

    if (result.fields.length === 0) {
      commands.notifications.add({ kind: 'info', ...emptyNotice });
      return false;
    }

    // Commit only prompt keys: resnapshotted values may contain an unrelated model-family transition and would
    // widen the lost-update window for concurrent media recall.
    if (result.fields.every((field) => field === 'prompts')) {
      commands.widgets.patchValues(
        'video',
        {
          negativePrompt: result.values.negativePrompt,
          negativePromptEnabled: result.values.negativePromptEnabled,
          positivePrompt: result.values.positivePrompt,
        },
        projectId
      );
    } else {
      commands.widgets.patchValues('video', { ...result.values }, projectId);
    }
    commands.notifications.add({
      kind: 'success',
      message: getVideoRecallMessage(result.fields),
      title: getVideoRecallTitle(kind),
    });
    return true;
  } catch (error: unknown) {
    return reportVideoRecallError(commands, owner, error, projectId);
  }
};

// Placing a gallery video in the Video panel. Kept in this module rather than a file of its own: the gallery's actions
// load it on every editor route, and a new module there is a new entry in the routes' pinned source-owner sets.

/** What placing a video needs to know about it; gallery items and the external recall event both carry it. */
export interface PlaceableVideo {
  durationSeconds: number;
  fps?: number;
  height: number;
  mediaOrigin?: string;
  name: string;
  width: number;
}

export type InitialVideoPlacement =
  | {
      patch: Partial<VideoWidgetValues>;
      status: 'placed';
      /** Whether the panel's model can generate from an initial video; the slot is filled either way. */
      usable: boolean;
    }
  | { status: 'full' };

/** Set the video as the Video panel's Initial Video, exactly as the panel's own Initial Video field would. */
export const placeInitialVideo = ({
  models,
  video,
  videoValues,
}: {
  models: readonly ModelConfig[];
  video: PlaceableVideo;
  videoValues: Record<string, unknown>;
}): InitialVideoPlacement => {
  const values = getCurrentVideoValues({ models, videoValues });
  const policy = values.model ? getVideoModelPolicy(values.model, values) : null;
  const referenceExtend = Boolean(policy?.references?.extend);
  const patch = getInitialVideoPatch({
    maxVideos: policy?.references?.maxVideos ?? 0,
    numFrames: values.numFrames,
    referenceExtend,
    references: values.references,
    sourceVideo: createVideoSourceClip(video),
  });

  return patch
    ? { patch, status: 'placed', usable: Boolean(policy && (policy.modes.includes('extend') || referenceExtend)) }
    : { status: 'full' };
};

export type ReferenceVideoPlacement =
  | { patch: Partial<VideoWidgetValues>; status: 'appended' }
  | { status: 'full' | 'unsupported' };

const getReferenceVideoRoom = (values: VideoWidgetValues): 'available' | 'full' | 'unsupported' => {
  const policy = values.model ? getVideoModelPolicy(values.model, values) : null;

  if (!policy?.references || !policy.modes.includes('reference')) {
    return 'unsupported';
  }

  return values.references.filter((entry) => entry.kind === 'video').length < policy.references.maxVideos
    ? 'available'
    : 'full';
};

/** Whether the Video panel's model takes reference videos and has room for another. */
export const canAppendReferenceVideo = ({
  models,
  videoValues,
}: {
  models: readonly ModelConfig[];
  videoValues: Record<string, unknown>;
}): boolean => getReferenceVideoRoom(getCurrentVideoValues({ models, videoValues })) === 'available';

/** Append the video to the Video panel's references with the defaults the References field gives a new video. */
export const appendReferenceVideo = ({
  models,
  video,
  videoValues,
}: {
  models: readonly ModelConfig[];
  video: PlaceableVideo;
  videoValues: Record<string, unknown>;
}): ReferenceVideoPlacement => {
  const values = getCurrentVideoValues({ models, videoValues });
  const room = getReferenceVideoRoom(values);

  if (room !== 'available') {
    return { status: room };
  }

  const referenceExtend = Boolean(values.model && getVideoModelPolicy(values.model, values).references?.extend);

  return {
    patch: getReferencesPatch({
      referenceExtend,
      references: [...values.references, createVideoReferenceEntry(video)],
    }),
    status: 'appended',
  };
};

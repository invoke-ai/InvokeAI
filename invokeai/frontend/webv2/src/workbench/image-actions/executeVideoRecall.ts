import type { GalleryImage, GalleryVideoItem } from '@features/gallery';
import type { ImageWithDims } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  VideoConditioningRole,
  VideoGenerationMode,
  VideoModelPolicy,
  VideoReferenceItem,
  VideoWidgetValues,
} from '@features/video';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { QueryClient } from '@tanstack/react-query';
import type { WorkbenchCommands } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { galleryImages, galleryItems } from '@features/gallery';
import { fetchGalleryVideoMetadata } from '@features/gallery/queries';
import {
  createDefaultVideoWidgetValues,
  createVideoConditioningClip,
  createVideoReferenceEntry,
  createVideoSourceClip,
  getConditioningClipPatch,
  getDefaultReferenceConditioning,
  DEFAULT_REFERENCE_IMAGE_DETAIL,
  getFrameImagePatch,
  getInitialVideoPatch,
  getReferencesPatch,
  getVideoModelPolicy,
  isConditioningClipExcludingFrames,
  isVideoConditioningClip,
  isVideoReferenceConditioning,
  normalizeVideoWidgetValues,
  syncVideoWidgetValuesWithModels,
} from '@features/video';
import { mapWithConcurrency } from '@platform/core/concurrency';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { ApiError } from '@platform/transport/http';

import {
  buildVideoRecallSettings,
  getVideoRecallMessage,
  getVideoRecallTitle,
  parseVideoRecallRecord,
  resolveVideoSizeRecall,
  settleMediaSizedRecall,
  type VideoRecallKind,
  type VideoRecallMediaNames,
  type VideoSizeRecallFit,
} from './videoRecall';

const toErrorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

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

/** There is no bulk video read. */
const VIDEO_RESOLVE_CONCURRENCY = 4;

interface ResolvedVideoRecallMedia {
  images: ReadonlyMap<string, GalleryImage>;
  videos: ReadonlyMap<string, GalleryVideoItem>;
}

const NO_RESOLVED_MEDIA: ResolvedVideoRecallMedia = { images: new Map(), videos: new Map() };

/** A 404 or 403 means the medium is gone for this caller and the recall goes on; any other failure rejects. */
const resolveVideoRecallMedia = async (
  names: VideoRecallMediaNames,
  signal: AbortSignal
): Promise<ResolvedVideoRecallMedia> => {
  const imageNames = new Set<string>();
  const videoNames = new Set<string>();

  for (const name of [names.firstFrameName, names.lastFrameName]) {
    if (name !== null) {
      imageNames.add(name);
    }
  }
  for (const name of [names.conditioningClip?.name ?? null, names.sourceVideoName]) {
    if (name !== null) {
      videoNames.add(name);
    }
  }
  for (const reference of names.references) {
    (reference.kind === 'image' ? imageNames : videoNames).add(reference.name);
  }

  const [images, videos] = await Promise.all([
    imageNames.size > 0 ? galleryImages.resolveMany([...imageNames], signal) : [],
    mapWithConcurrency(
      [...videoNames],
      VIDEO_RESOLVE_CONCURRENCY,
      async (name): Promise<[string, GalleryVideoItem] | null> => {
        try {
          const item = await galleryItems.resolve({ kind: 'video', name }, signal);

          return item.kind === 'video' ? [name, item] : null;
        } catch (error) {
          if (error instanceof ApiError && (error.status === 403 || error.status === 404)) {
            return null;
          }
          throw error;
        }
      },
      { signal }
    ),
  ]);

  return {
    images: new Map(images.map((image) => [image.imageName, image])),
    videos: new Map(videos.flatMap((entry) => (entry ? [entry] : []))),
  };
};

/** A partial recall's unusable medium would still displace held media before reconcile, so drop it up front. */
const pruneUnusablePartialMedia = ({
  mediaNames: names,
  values,
}: {
  mediaNames: VideoRecallMediaNames;
  values: VideoWidgetValues;
}): void => {
  if (!values.model) {
    return;
  }

  const policy = getVideoModelPolicy(values.model, values);
  const modes = policy.modes;

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
  if (names.conditioningClip && !modes.includes(CONDITIONING_ROLE_MODE[names.conditioningClip.role])) {
    names.conditioningClip = null;
  }
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

const getSizeRecallFailureMessage = (
  fit: VideoSizeRecallFit,
  size: { height: number; width: number },
  t: TFunction
): string => {
  switch (fit.outcome) {
    case 'invalid':
      return fit.reason;
    case 'media-conflict':
      return t('widgets.gallery.itemActions.recall.videoSizeMediaConflict', {
        ...size,
        effectiveHeight: fit.effective.height,
        effectiveWidth: fit.effective.width,
      });
    case 'media-unresolvable':
      return t('widgets.gallery.itemActions.recall.videoSizeMediaUnresolvable', size);
    default:
      return t('widgets.gallery.itemActions.recall.videoSizeUnavailable', size);
  }
};

export const executeVideoRecall = async ({
  commands,
  getVideoValues,
  item,
  kind,
  models,
  owner: callerOwner,
  projectId,
  queryClient,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  getVideoValues: () => Record<string, unknown>;
  item: GalleryVideoItem;
  kind: VideoRecallKind;
  models: ModelConfig[];
  owner?: AccountScope;
  projectId?: string;
  queryClient: QueryClient;
  t: TFunction;
}): Promise<boolean> => {
  const owner = callerOwner ?? captureAccountScope();

  if (!isAccountScopeCurrent(owner)) {
    return false;
  }

  // Values sync against the catalog: an empty one would drop the panel's model, LoRAs and components on commit.
  if (models.length === 0 && kind !== 'prompts') {
    commands.notifications.add({
      kind: 'info',
      message: t('widgets.gallery.itemActions.recall.modelsUnavailableForVideoRecall'),
      title: t('widgets.gallery.itemActions.recall.cannotRecallVideo'),
    });
    return false;
  }

  let metadata: unknown;

  try {
    metadata = await fetchGalleryVideoMetadata(queryClient, owner, item.name);
  } catch (error: unknown) {
    return reportVideoRecallError(commands, owner, error, projectId);
  }

  // A recorded size the panel cannot take is not missing metadata: say which size and why.
  const record = kind === 'size' ? parseVideoRecallRecord(metadata) : null;
  const size =
    record?.isVideoGeneration && record.width && record.height ? { height: record.height, width: record.width } : null;
  const emptyNotice = size
    ? {
        message: getSizeRecallFailureMessage(
          resolveVideoSizeRecall(
            getCurrentVideoValues({ models, videoValues: getVideoValues() }),
            size.width,
            size.height
          ),
          size,
          t
        ),
        title: t('widgets.gallery.itemActions.recall.nothingRecalledVideoSize'),
      }
    : {
        message: t('widgets.gallery.itemActions.recall.videoMetadataUnsupported'),
        title: t('widgets.gallery.itemActions.recall.nothingRecalledVideo'),
      };

  return applyVideoRecallMetadata({
    commands,
    emptyNotice,
    getVideoValues,
    kind,
    metadata,
    models,
    owner,
    projectId,
    t,
  });
};

/** From a gallery video or the external recall API. */
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
  t,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  emptyNotice: VideoRecallNotice;
  getVideoValues: () => Record<string, unknown>;
  kind: VideoRecallKind;
  metadata: unknown;
  models: readonly ModelConfig[];
  owner: AccountScope;
  partial?: boolean;
  projectId?: string;
  requireGenerationMode?: boolean;
  t: TFunction;
}): Promise<boolean> => {
  try {
    assertAccountScopeCurrent(owner);
    const build = (videoValues: Record<string, unknown>) => {
      const currentValues = getCurrentVideoValues({ models, videoValues });
      const result = buildVideoRecallSettings({
        currentValues,
        kind,
        metadata,
        models,
        partial,
        requireGenerationMode,
      });

      if (result && partial) {
        pruneUnusablePartialMedia(result);
      }
      return { currentValues, result };
    };
    // A first build only decides what to read, so a recall that ends early or drops a medium never reads it.
    const planned = build(getVideoValues()).result;
    const resolved = planned?.fields.includes('media')
      ? await resolveVideoRecallMedia(planned.mediaNames, owner.signal)
      : NO_RESOLVED_MEDIA;

    assertAccountScopeCurrent(owner);
    // Snapshot the panel after every read: an edit made while media loaded must survive. A medium only this build names
    // was not read and is left out like a deleted one.
    const { currentValues, result } = build(getVideoValues());

    if (!result) {
      commands.notifications.add({ kind: 'info', ...emptyNotice });
      return false;
    }

    if (result.fields.includes('media')) {
      // A full recall that cleared held media changed the panel even if nothing hydrates.
      const clearedMedia = (
        ['conditioningClip', 'firstFrameImage', 'lastFrameImage', 'references', 'sourceVideo'] as const
      ).some((slot) => result.values[slot] !== currentValues[slot]);
      let recalledMedia = false;
      const { firstFrameName, lastFrameName, sourceVideoName } = result.mediaNames;
      const firstFrame = firstFrameName ? resolved.images.get(firstFrameName) : undefined;
      const lastFrame = lastFrameName ? resolved.images.get(lastFrameName) : undefined;

      // Through the panel's own frame patch, so a frame displaces a held clip exactly as setting it by hand would.
      if (firstFrame) {
        result.values = {
          ...result.values,
          ...getFrameImagePatch(
            'firstFrameImage',
            { height: firstFrame.height, image_name: firstFrame.imageName, width: firstFrame.width },
            result.values.conditioningClip
          ),
        };
        recalledMedia = true;
      }
      if (lastFrame) {
        result.values = {
          ...result.values,
          ...getFrameImagePatch(
            'lastFrameImage',
            { height: lastFrame.height, image_name: lastFrame.imageName, width: lastFrame.width },
            result.values.conditioningClip
          ),
        };
        recalledMedia = true;
      }

      const { conditioningClip } = result.mediaNames;

      const clipItem = conditioningClip ? resolved.videos.get(conditioningClip.name) : undefined;

      if (conditioningClip && clipItem) {
        // The recorded role, not a fresh drop's default: the other role is a different generation entirely.
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
          // A soundtrack keeps its frames: they anchored the picture made for it.
          ...(isConditioningClipExcludingFrames(conditioningClip)
            ? { firstFrameImage: null, lastFrameImage: null }
            : {}),
          references: [],
          sourceVideo: null,
        };
        recalledMedia = true;
      }

      const sourceItem = sourceVideoName ? resolved.videos.get(sourceVideoName) : undefined;

      if (sourceItem) {
        const rebuiltClip = createVideoSourceClip({
          durationSeconds: sourceItem.durationSeconds,
          fps: sourceItem.fps,
          height: sourceItem.height,
          name: sourceItem.name,
          width: sourceItem.width,
        });
        // Default trim would select different frames.
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

      if (result.mediaNames.references.length > 0) {
        const references: VideoReferenceItem[] = [];

        for (const recorded of result.mediaNames.references) {
          if (recorded.kind === 'image') {
            const image = resolved.images.get(recorded.name);

            if (image) {
              references.push({
                detail: recorded.detail === 'match' ? 'match' : 'max',
                image: { height: image.height, image_name: image.imageName, width: image.width },
                kind: 'image',
              });
            }
            continue;
          }

          const item = resolved.videos.get(recorded.name);

          if (!item) {
            continue;
          }
          const clip = createVideoSourceClip({
            durationSeconds: item.durationSeconds,
            fps: item.fps,
            height: item.height,
            name: item.name,
            width: item.width,
          });
          const startFrame = recorded.trim ? Math.min(Math.max(recorded.trim.startFrame, 0), clip.numFrames - 1) : 0;
          const endFrame = recorded.trim
            ? Math.min(Math.max(recorded.trim.endFrame, startFrame), clip.numFrames - 1)
            : clip.numFrames - 1;
          const conditioning = isVideoReferenceConditioning(recorded.conditioning)
            ? recorded.conditioning
            : getDefaultReferenceConditioning(item.mediaOrigin);

          references.push({ clip: { ...clip, endFrame, startFrame }, conditioning, kind: 'video' });
        }

        if (references.length > 0) {
          result.values = {
            ...result.values,
            firstFrameImage: null,
            lastFrameImage: null,
            references,
            // A source video recorded alongside references is reference-extend state; clear only a leftover.
            sourceVideo: result.mediaNames.sourceVideoName || partial ? result.values.sourceVideo : null,
          };
          recalledMedia = true;
        }
      }

      if (!recalledMedia && !clearedMedia) {
        result.fields = result.fields.filter((field) => field !== 'media');
      }

      // Deleted frames or unavailable models must not leave an ungeneratable panel.
      const effectiveModel = result.values.model;

      if (effectiveModel) {
        const policy = getVideoModelPolicy(effectiveModel, result.values);
        const modes = policy.modes;
        const referenceExtend = Boolean(policy.references?.extend);

        // A partial recall could orphan a reference-extend clip's continuity reference; relink it like Initial Video.
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
          references = [];
        }
        if (references.length > 0) {
          // The source video survives only on a reference-extend panel.
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

    ({ fields: result.fields, values: result.values } = settleMediaSizedRecall(result));

    if (result.fields.length === 0) {
      commands.notifications.add({ kind: 'info', ...emptyNotice });
      return false;
    }

    // Resnapshotted values may hold an unrelated model-family transition and would widen the lost-update window.
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
      message: getVideoRecallMessage(result.fields, t),
      title: getVideoRecallTitle(kind, t),
    });
    return true;
  } catch (error: unknown) {
    return reportVideoRecallError(commands, owner, error, projectId);
  }
};

// Kept in this module: a new module on every editor route is a new entry in the routes' pinned source-owner sets.

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
      /** The slot is filled either way. */
      usable: boolean;
    }
  | { status: 'full' }
  | { status: 'unsupported' };

/** Exactly as the panel's own Initial Video field would. */
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
  if (values.hostedH3) {
    return { status: 'unsupported' };
  }
  const policy = !values.hostedH3 && values.model ? getVideoModelPolicy(values.model, values) : null;
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

const getReferenceVideoRoom = (
  values: VideoWidgetValues,
  policy = !values.hostedH3 && values.model ? getVideoModelPolicy(values.model, values) : null
): 'available' | 'full' | 'unsupported' => {
  if (!policy?.references || !policy.modes.includes('reference')) {
    return 'unsupported';
  }

  return values.references.filter((entry) => entry.kind === 'video').length < policy.references.maxVideos
    ? 'available'
    : 'full';
};

/** With the defaults the References field gives a new video. */
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

export type VideoImagePlacement =
  | {
      displaced: boolean;
      patch: Partial<VideoWidgetValues>;
      slot: 'firstFrame' | 'lastFrame' | 'reference';
      status: 'placed';
    }
  | { status: 'full' | 'unsupported' };

/** Reference models get it as a reference image, frame models as the first frame; the panel's model never switches. */
export const placeVideoImage = ({
  append,
  image,
  models,
  videoValues,
}: {
  append: boolean;
  image: ImageWithDims;
  models: readonly ModelConfig[];
  videoValues: Record<string, unknown>;
}): VideoImagePlacement => {
  const values = getCurrentVideoValues({ models, videoValues });
  if (values.hostedH3) {
    if (values.hostedH3.input === 'references') {
      const kept = append ? values.references : [];
      if (kept.length >= 9) {
        return { status: 'full' };
      }
      return {
        displaced: Boolean(values.sourceVideo || values.conditioningClip),
        patch: getReferencesPatch({
          referenceExtend: false,
          references: [...kept, { detail: DEFAULT_REFERENCE_IMAGE_DETAIL, image, kind: 'image' }],
        }),
        slot: 'reference',
        status: 'placed',
      };
    }
    if (append && values.firstFrameImage && values.lastFrameImage) {
      return { status: 'full' };
    }
    const last = append && values.firstFrameImage !== null;
    return {
      displaced: Boolean(values.sourceVideo || values.conditioningClip || values.references.length),
      patch: {
        hostedH3: { ...values.hostedH3, input: 'frames' },
        references: [],
        sourceVideo: null,
        conditioningClip: null,
        firstFrameImage: last ? values.firstFrameImage : image,
        lastFrameImage: last ? image : append ? values.lastFrameImage : null,
      },
      slot: last ? 'lastFrame' : 'firstFrame',
      status: 'placed',
    };
  }
  const policy = !values.hostedH3 && values.model ? getVideoModelPolicy(values.model, values) : null;

  if (!policy) {
    return { status: 'unsupported' };
  }

  if (policy.references && policy.modes.includes('reference')) {
    const kept = append ? values.references : values.references.filter((entry) => entry.kind === 'video');

    if (kept.filter((entry) => entry.kind === 'image').length >= policy.references.maxImages) {
      return { status: 'full' };
    }
    const referenceExtend = Boolean(policy.references.extend);

    return {
      displaced: Boolean(values.conditioningClip || (values.sourceVideo && !referenceExtend)),
      patch: getReferencesPatch({
        referenceExtend,
        references: [...kept, { detail: DEFAULT_REFERENCE_IMAGE_DETAIL, image, kind: 'image' }],
      }),
      slot: 'reference',
      status: 'placed',
    };
  }

  const takesFirstFrame = policy.modes.includes('first-frame') || policy.modes.includes('first-last');

  if (!takesFirstFrame) {
    return { status: 'unsupported' };
  }

  // The stored clip, not the normalized one, which hides a picture-role clip behind a first frame.
  const storedClip = isVideoConditioningClip(videoValues.conditioningClip) ? videoValues.conditioningClip : null;
  const clipDisplaced = isConditioningClipExcludingFrames(values.conditioningClip);

  if (!append) {
    return {
      displaced: Boolean(clipDisplaced || values.sourceVideo),
      patch: {
        ...getFrameImagePatch('firstFrameImage', image, storedClip),
        lastFrameImage: null,
        references: [],
      },
      slot: 'firstFrame',
      status: 'placed',
    };
  }
  if (!values.firstFrameImage && !values.sourceVideo) {
    return {
      displaced: clipDisplaced,
      // A last frame hidden behind stale references is cleared rather than resurfaced.
      patch: {
        ...getFrameImagePatch('firstFrameImage', image, storedClip),
        lastFrameImage: values.lastFrameImage,
        references: [],
      },
      slot: 'firstFrame',
      status: 'placed',
    };
  }
  if (!values.lastFrameImage && policy.modes.includes('first-last')) {
    return {
      displaced: clipDisplaced,
      patch: { ...getFrameImagePatch('lastFrameImage', image, storedClip), references: [] },
      slot: 'lastFrame',
      status: 'placed',
    };
  }

  return { status: 'full' };
};

const CONDITIONING_ROLE_MODE = {
  audio: 'audio-to-video',
  video: 'video-to-audio',
} as const satisfies Record<VideoConditioningRole, VideoGenerationMode>;

export type ConditioningClipPlacement =
  | {
      displaced: boolean;
      patch: Partial<VideoWidgetValues>;
      status: 'placed';
    }
  /** A wrapped audio upload has only a placeholder picture, so it can only lend its soundtrack. */
  | { status: 'no-picture' | 'unsupported' };

const canConditionInRole = (policy: VideoModelPolicy | null, role: VideoConditioningRole): boolean =>
  Boolean(policy?.modes.includes(CONDITIONING_ROLE_MODE[role]));

/** Unlike the panel's field, clears conflicting media instead of refusing; declines a clip the model cannot use. */
export const placeConditioningClip = ({
  models,
  role,
  video,
  videoValues,
}: {
  models: readonly ModelConfig[];
  role: VideoConditioningRole;
  video: PlaceableVideo;
  videoValues: Record<string, unknown>;
}): ConditioningClipPlacement => {
  if (role === 'video' && video.mediaOrigin === 'audio_upload') {
    return { status: 'no-picture' };
  }

  const values = getCurrentVideoValues({ models, videoValues });

  if (values.hostedH3 || !canConditionInRole(values.model ? getVideoModelPolicy(values.model, values) : null, role)) {
    return { status: 'unsupported' };
  }

  const clip = { ...createVideoConditioningClip(video), role };

  return {
    displaced: Boolean(
      values.sourceVideo ||
      values.references.length > 0 ||
      (isConditioningClipExcludingFrames(clip) && (values.firstFrameImage || values.lastFrameImage))
    ),
    patch: getConditioningClipPatch(clip),
    status: 'placed',
  };
};

export interface VideoPlacementRoom {
  initialVideo: boolean;
  conditioningAudio: boolean;
  conditioningVideo: boolean;
  referenceVideo: boolean;
}

export const getVideoPlacementRoom = ({
  models,
  videoValues,
}: {
  models: readonly ModelConfig[];
  videoValues: Record<string, unknown>;
}): VideoPlacementRoom => {
  const values = getCurrentVideoValues({ models, videoValues });
  const policy = !values.hostedH3 && values.model ? getVideoModelPolicy(values.model, values) : null;

  return {
    initialVideo: !values.hostedH3,
    conditioningAudio: canConditionInRole(policy, 'audio'),
    conditioningVideo: canConditionInRole(policy, 'video'),
    referenceVideo: getReferenceVideoRoom(values, policy) === 'available',
  };
};

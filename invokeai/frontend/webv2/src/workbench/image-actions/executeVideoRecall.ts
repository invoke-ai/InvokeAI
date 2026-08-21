import type { GalleryVideoItem } from '@features/gallery';
import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { WorkbenchCommands } from '@workbench/workbenchStore';

import { galleryImages, galleryItems, galleryVideos } from '@features/gallery';
import {
  createDefaultVideoWidgetValues,
  createVideoSourceClip,
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

export const executeVideoRecall = async ({
  commands,
  getVideoValues,
  item,
  kind,
  models,
  owner: callerOwner,
  projectId,
}: {
  commands: Pick<WorkbenchCommands, 'generation' | 'notifications' | 'widgets'>;
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

  try {
    const metadata = await loadVideoMetadata(item.name, owner);

    assertAccountScopeCurrent(owner);
    // Snapshot the panel AFTER the fetch: an edit made while the metadata
    // loaded must survive into the base the recall applies on top of.
    const currentValues = getCurrentVideoValues({ models, videoValues: getVideoValues() });
    const result = buildVideoRecallSettings({ currentValues, kind, metadata, models });

    if (!result) {
      commands.notifications.add({
        kind: 'info',
        message: 'This video does not include supported Video metadata.',
        title: 'No recallable video data',
      });
      return false;
    }

    // Conditioning media re-hydrates against the gallery: the metadata records
    // only names, and the panel needs dimensions/probe data — plus deleted
    // media must drop rather than resurrect as broken references.
    if (result.fields.includes('media')) {
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

      if (sourceVideoName) {
        try {
          const sourceItem = await galleryItems.resolve({ kind: 'video', name: sourceVideoName }, owner.signal);

          assertAccountScopeCurrent(owner);
          if (sourceItem?.kind === 'video') {
            result.values = {
              ...result.values,
              firstFrameImage: null,
              sourceVideo: createVideoSourceClip({
                durationSeconds: sourceItem.durationSeconds,
                fps: sourceItem.fps,
                height: sourceItem.height,
                name: sourceItem.name,
                width: sourceItem.width,
              }),
            };
            recalledMedia = true;
          }
        } catch {
          assertAccountScopeCurrent(owner);
          // The source clip is gone; the rest of the recall still applies.
        }
      }

      if (!recalledMedia) {
        result.fields = result.fields.filter((field) => field !== 'media');
      }
    }

    if (result.fields.length === 0) {
      commands.notifications.add({
        kind: 'info',
        message: 'This video does not include supported Video metadata.',
        title: 'No recallable video data',
      });
      return false;
    }

    // Prompts live in the shared project draft; everything else is widget values.
    if (result.promptPatch) {
      commands.generation.patchPromptDraft(result.promptPatch, 'video', projectId);
    }
    // A prompts-only recall changes no widget values — skip the whole-values
    // write (and the lost-update window it would widen).
    if (result.values !== currentValues || result.fields.some((field) => field !== 'prompts')) {
      commands.widgets.patchValues('video', { ...result.values }, projectId);
    }
    commands.notifications.add({
      kind: 'success',
      message: getVideoRecallMessage(result.fields),
      title: getVideoRecallTitle(kind),
    });
    return true;
  } catch (error: unknown) {
    if (!isAccountScopeCurrent(owner)) {
      return false;
    }

    commands.notifications.reportError({
      area: 'video-recall',
      message: toErrorMessage(error),
      namespace: 'generation',
      projectId,
    });
    return false;
  }
};

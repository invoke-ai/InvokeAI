import type { ModelConfig } from '@features/models';
import type { AccountScope } from '@platform/state/accountLifecycle';
import type { SocketHub } from '@platform/transport/socketHub';
import type { WorkbenchCommands, WorkbenchQueries } from '@workbench/workbenchStore';
import type { TFunction } from 'i18next';

import { getProjectWidgetValues } from '@workbench/widgetState';

import type { PlaceableVideo } from './index';
import type { PendingRecallEvent, RecallRevealContext, RecallRuntime } from './recallEventRuntime';

// Through the package entry, not the modules themselves: the gallery's actions share these modules, and importing
// them from this lazy runtime directly would split them out of the image-actions chunk into one more request on
// every editor route.
import { appendReferenceVideo, applyVideoRecallMetadata, placeInitialVideo } from './index';
import { bringRecallWidgetToFront, createRecallEventRuntime } from './recallEventRuntime';

/** The `video` of a `video_recall_requested` event: a gallery video, as the backend describes it. */
interface VideoRecallEventVideo {
  duration: number;
  fps: number | null;
  height: number;
  media_origin: string | null;
  video_name: string;
  width: number;
}

/** Wire payload of the backend's `video_recall_requested` socket event (`POST /api/v1/recall/video`). */
export type VideoRecallRequestedEvent = { user_id: string } & (
  | {
      action: 'parameters';
      mode: 'recall' | 'remix';
      /** Resolved fields keyed like the video metadata record. */
      parameters: Record<string, unknown>;
      strict: boolean;
    }
  | { action: 'initial_video' | 'reference_video'; video: VideoRecallEventVideo }
);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

const isEventVideo = (value: unknown): value is VideoRecallEventVideo =>
  isRecord(value) &&
  typeof value.video_name === 'string' &&
  isPositiveNumber(value.width) &&
  isPositiveNumber(value.height) &&
  typeof value.duration === 'number' &&
  Number.isFinite(value.duration) &&
  value.duration >= 0 &&
  (value.fps === null || isPositiveNumber(value.fps)) &&
  (value.media_origin === null || typeof value.media_origin === 'string');

export const isVideoRecallRequestedEvent = (payload: unknown): payload is VideoRecallRequestedEvent => {
  if (!isRecord(payload) || typeof payload.user_id !== 'string') {
    return false;
  }
  if (payload.action === 'parameters') {
    return (
      (payload.mode === 'recall' || payload.mode === 'remix') &&
      typeof payload.strict === 'boolean' &&
      isRecord(payload.parameters)
    );
  }

  return (payload.action === 'initial_video' || payload.action === 'reference_video') && isEventVideo(payload.video);
};

const toPlaceableVideo = (video: VideoRecallEventVideo): PlaceableVideo => ({
  durationSeconds: video.duration,
  fps: video.fps ?? undefined,
  height: video.height,
  mediaOrigin: video.media_origin ?? undefined,
  name: video.video_name,
  width: video.width,
});

/**
 * Apply `video_recall_requested` events to their arrival-time project's Video panel: recall or remix parameters,
 * set the Initial Video, or append a reference video. The panel's model is never switched to make a video fit; a
 * placement it cannot take is declined with a notice. The Video widget is revealed when the project is still the
 * one on screen.
 */
export const createVideoRecallRuntime = ({
  commands,
  getSessionUserId,
  hub,
  queries,
  replay,
  reveal,
  t,
}: {
  commands: Pick<WorkbenchCommands, 'notifications' | 'widgets'>;
  getSessionUserId?: () => string | null;
  hub: Pick<SocketHub, 'on'>;
  queries: Pick<WorkbenchQueries, 'getProject' | 'getSnapshot'>;
  replay?: readonly PendingRecallEvent[];
  reveal: RecallRevealContext;
  /** Resolves against the current language at call time; captured once, at attach. */
  t: TFunction;
}): RecallRuntime => {
  const apply = async (
    event: VideoRecallRequestedEvent,
    { models, owner, projectId }: { models: ModelConfig[]; owner: AccountScope; projectId: string }
  ): Promise<void> => {
    const readVideoValues = () => {
      const project = queries.getProject(projectId);
      return project ? getProjectWidgetValues(project, 'video') : null;
    };
    const videoValues = readVideoValues();

    // The project closed while the event waited.
    if (!videoValues) {
      return;
    }

    let applied = false;

    if (event.action === 'parameters') {
      applied = await applyVideoRecallMetadata({
        commands,
        emptyNotice: {
          message: t('widgets.video.externalRecall.nothingAppliedDescription'),
          title: t('widgets.video.externalRecall.nothingApplied'),
        },
        getVideoValues: () => readVideoValues() ?? {},
        kind: event.mode === 'remix' ? 'remix' : 'all',
        metadata: event.parameters,
        models,
        owner,
        partial: !event.strict,
        projectId,
        requireGenerationMode: false,
      });
    } else if (event.action === 'initial_video') {
      const placement = placeInitialVideo({ models, video: toPlaceableVideo(event.video), videoValues });

      if (placement.status === 'full') {
        commands.notifications.add({
          kind: 'info',
          message: t('widgets.video.placement.initialVideoFull'),
          title: t('widgets.video.referenceExtendCapFull'),
        });
      } else {
        commands.widgets.patchValues('video', placement.patch, projectId);
        commands.notifications.add(
          placement.usable
            ? { kind: 'success', title: t('widgets.video.placement.initialVideoSet') }
            : {
                kind: 'info',
                message: t('widgets.video.placement.initialVideoUnused'),
                title: t('widgets.video.placement.initialVideoSet'),
              }
        );
        applied = true;
      }
    } else {
      const placement = appendReferenceVideo({ models, video: toPlaceableVideo(event.video), videoValues });

      if (placement.status === 'appended') {
        commands.widgets.patchValues('video', placement.patch, projectId);
        commands.notifications.add({ kind: 'success', title: t('widgets.video.placement.referenceAdded') });
        applied = true;
      } else {
        commands.notifications.add({
          kind: 'info',
          message: t(
            placement.status === 'full'
              ? 'widgets.video.placement.referenceFull'
              : 'widgets.video.placement.referenceUnsupported'
          ),
          title: t('widgets.video.placement.referenceNotAdded'),
        });
      }
    }

    // The value patch itself routes the project's invocation to Video, where the user lets edits route it.
    if (applied) {
      bringRecallWidgetToFront({ commands, owner, projectId, queries, reveal, typeId: 'video' });
    }
  };

  return createRecallEventRuntime({
    apply,
    area: 'video-recall',
    commands,
    eventName: 'video_recall_requested',
    getSessionUserId,
    hub,
    isEvent: isVideoRecallRequestedEvent,
    queries,
    replay,
  });
};

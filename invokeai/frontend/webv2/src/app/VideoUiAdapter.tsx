import type { VideoUiAdapter } from '@features/video';
import type { ReactNode } from 'react';

import { getGalleryAutoAddBoardId, toGalleryItemKey } from '@features/gallery/contracts';
import {
  canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations,
  invalidateGallery,
  invalidateGalleryBoardDestinations,
} from '@features/gallery/queries';
import { createDefaultVideoWidgetValues, normalizeVideoWidgetValues, VideoUiProvider } from '@features/video';
import { useQueryClient } from '@tanstack/react-query';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import {
  getVideoSpanPlaybackState,
  requestVideoSpanPlayback,
  subscribeVideoSpanPlaybackState,
} from '@workbench/widgets/preview/spanPlaybackRequest';
import { getProjectWidgetInstance, getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useMemo, useRef } from 'react';

const EMPTY_GALLERY_VALUES: Record<string, unknown> = Object.freeze({});

/** Video owns its prompt in widget values; do not join the draft shared by Generate/Upscale. */
export const VideoUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector(
    (activeProject) => {
      const instance = Object.values(activeProject.widgetInstances).find((candidate) => candidate.typeId === 'video');

      return {
        projectId: activeProject.id,
        rawValues: instance?.state.values ?? {},
      };
    },
    (left, right) => left.projectId === right.projectId && left.rawValues === right.rawValues
  );
  // Syntax highlighting is an account preference, not project data.
  const showPromptSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  // Panel uploads land where new results do: the auto-add board, resolved when the upload starts, against the board
  // listing as it is then.
  const galleryValues = useActiveProjectSelector(
    (activeProject) => getProjectWidgetInstance(activeProject, 'gallery')?.state?.values ?? EMPTY_GALLERY_VALUES
  );
  // Read the gallery values through a ref so board selection does not recreate project actions.
  const galleryValuesRef = useRef(galleryValues);
  // Async clip resolution can outlive a project switch; callbacks must check the live project.
  const activeProjectIdRef = useRef(project.projectId);
  useEffect(() => {
    galleryValuesRef.current = galleryValues;
    activeProjectIdRef.current = project.projectId;
  }, [galleryValues, project.projectId]);
  const commands = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const queryClient = useQueryClient();
  // Key actions by project, not values, to preserve callback identity while typing.
  const { projectId } = project;
  const patchValues = useCallback<VideoUiAdapter['patchValues']>(
    (values, origin) =>
      commands.widgets.patchValues(
        'video',
        typeof values === 'function'
          ? (current) => values(normalizeVideoWidgetValues(current) ?? createDefaultVideoWidgetValues())
          : values,
        projectId,
        origin
      ),
    [commands, projectId]
  );
  // The same project patchValues writes to, so an async commit reads and writes one panel.
  const readValues = useCallback<VideoUiAdapter['readValues']>(() => {
    const target = queries.getProject(projectId);

    return target ? getProjectWidgetValues(target, 'video') : {};
  }, [projectId, queries]);
  const reportError = useCallback<VideoUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'video', message, namespace: 'generation' }),
    [commands]
  );
  const touchGalleryImages = useCallback(() => void invalidateGallery(queryClient), [queryClient]);
  const refreshBoardDestinations = useCallback(
    () => void invalidateGalleryBoardDestinations(queryClient),
    [queryClient]
  );
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  // Raise Preview before changing selection; refusal must leave selection untouched. Request the span last and do
  // not reveal/scroll the gallery.
  const playVideoSpanInPreview = useCallback<VideoUiAdapter['playVideoSpanInPreview']>(
    ({ endSeconds, item, startSeconds }) => {
      // Discard a resolved clip if its originating project is no longer active.
      if (activeProjectIdRef.current !== projectId) {
        return null;
      }

      if (!openWorkbenchWidget('preview', { preferredRegions: ['center'], requireCenterView: true }).ok) {
        return null;
      }

      commands.gallery.selectItem(item, projectId);

      return requestVideoSpanPlayback({ endSeconds, itemKey: toGalleryItemKey(item), startSeconds });
    },
    [commands, openWorkbenchWidget, projectId]
  );
  // Judged from the listing of every readable board, ensured as the upload starts: an archived or revoked board is
  // decided from a listing, not for want of one, and no listing is read while nothing uploads. The board is read
  // before anything is awaited, so a later change to the panel cannot redirect an upload already under way.
  const getUploadBoardId = useCallback<VideoUiAdapter['getUploadBoardId']>(async () => {
    const galleryValues = galleryValuesRef.current;
    const pendingListing = ensureGalleryBoardDestinations(queryClient);

    if (pendingListing) {
      await pendingListing;
    }

    return (
      getGalleryAutoAddBoardId(galleryValues, (boardId) => canAddToCachedGalleryBoard(boardId, queryClient)) ?? 'none'
    );
  }, [queryClient]);
  const findInGallery = useFindGalleryItem();
  const adapter = useMemo<VideoUiAdapter>(
    () => ({
      ...project,
      findInGallery,
      getUploadBoardId,
      patchValues,
      playVideoSpanInPreview,
      readValues,
      refreshBoardDestinations,
      reportError,
      showPromptSyntaxHighlighting,
      touchGalleryImages,
      videoSpanPlayback: VIDEO_SPAN_PLAYBACK_PORT,
    }),
    [
      findInGallery,
      getUploadBoardId,
      patchValues,
      playVideoSpanInPreview,
      project,
      readValues,
      refreshBoardDestinations,
      reportError,
      showPromptSyntaxHighlighting,
      touchGalleryImages,
    ]
  );

  return <VideoUiProvider adapter={adapter}>{children}</VideoUiProvider>;
};

// The module-scoped player report permits one stable port across project switches.
const VIDEO_SPAN_PLAYBACK_PORT: VideoUiAdapter['videoSpanPlayback'] = {
  getState: getVideoSpanPlaybackState,
  subscribe: subscribeVideoSpanPlaybackState,
};

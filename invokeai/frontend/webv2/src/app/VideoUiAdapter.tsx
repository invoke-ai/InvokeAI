import type { VideoUiAdapter } from '@features/video';
import type { ReactNode } from 'react';

import { getGalleryAutoAddBoardId, toGalleryItemKey } from '@features/gallery/contracts';
import { invalidateGallery } from '@features/gallery/queries';
import { VideoUiProvider } from '@features/video';
import { useQueryClient } from '@tanstack/react-query';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import {
  getVideoSpanPlaybackState,
  requestVideoSpanPlayback,
  subscribeVideoSpanPlaybackState,
} from '@workbench/widgets/preview/spanPlaybackRequest';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useMemo, useRef } from 'react';

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
  // Panel uploads land where new results do: the auto-add board.
  const uploadBoardId = useActiveProjectSelector(
    (activeProject) => getGalleryAutoAddBoardId(getProjectWidgetValues(activeProject, 'gallery')) ?? 'none'
  );
  // Read the current board through a ref so board selection does not recreate project actions.
  const uploadBoardIdRef = useRef(uploadBoardId);
  // Async clip resolution can outlive a project switch; callbacks must check the live project.
  const activeProjectIdRef = useRef(project.projectId);
  useEffect(() => {
    uploadBoardIdRef.current = uploadBoardId;
    activeProjectIdRef.current = project.projectId;
  }, [project.projectId, uploadBoardId]);
  const commands = useWorkbenchCommands();
  const queryClient = useQueryClient();
  // Key actions by project, not values, to preserve callback identity while typing.
  const { projectId } = project;
  const patchValues = useCallback<VideoUiAdapter['patchValues']>(
    (values, origin) => commands.widgets.patchValues('video', values, projectId, origin),
    [commands, projectId]
  );
  const reportError = useCallback<VideoUiAdapter['reportError']>(
    (message) => commands.notifications.reportError({ area: 'video', message, namespace: 'generation' }),
    [commands]
  );
  const touchGalleryImages = useCallback(() => void invalidateGallery(queryClient), [queryClient]);
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
  const getUploadBoardId = useCallback(() => uploadBoardIdRef.current, []);
  const findInGallery = useFindGalleryItem();
  const adapter = useMemo<VideoUiAdapter>(
    () => ({
      ...project,
      findInGallery,
      getUploadBoardId,
      patchValues,
      playVideoSpanInPreview,
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

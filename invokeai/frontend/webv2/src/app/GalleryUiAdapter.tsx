import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryUiAdapter } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryUiProvider } from '@features/gallery/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useLivePreviewFollow } from '@workbench/widgets/preview/livePreviewFollow';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

const EMPTY_WIDGET_VALUES: Record<string, unknown> = Object.freeze({});

// Loaded on first reveal; the label cache is not part of the editor's initial graph.
const getItemLabel = (item: GalleryItemRef): Promise<string | null> =>
  import('@workbench/image-map/imageLabelCache')
    .then(({ getImageLabels }) => getImageLabels(item))
    .then((labels) => labels?.label ?? null)
    // A chunk that fails to load (stale deploy, dropped network) means no label, not an unhandled rejection.
    .catch(() => null);

const GalleryItemActionsAdapter = lazy(() =>
  import('./GalleryImageActionsBridge').then((module) => ({ default: module.GalleryItemActionsAdapter }))
);
const GalleryImageContextMenu = lazy(() =>
  import('./GalleryImageActionsBridge').then((module) => ({ default: module.GalleryImageContextMenu }))
);

export const GalleryUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const { projectId, projectName, galleryValues, generateValues, antialiasProgressImages, liveFollowEnabled } =
    useActiveProjectSelector((project) => ({
      projectId: project.id,
      projectName: project.name,
      galleryValues: getProjectWidgetInstance(project, 'gallery')?.state?.values ?? EMPTY_WIDGET_VALUES,
      generateValues: getProjectWidgetInstance(project, 'generate')?.state?.values ?? EMPTY_WIDGET_VALUES,
      antialiasProgressImages: project.settings.antialiasProgressImages,
      liveFollowEnabled: project.settings.showProgressImagesInViewer,
    }));
  const livePreview = useLivePreviewFollow();
  const { gallery, notifications, widgets } = useWorkbenchCommands();
  const queries = useWorkbenchQueries();
  const accountScope = captureAccountScope();
  const exportProject = useExportLibraryProject();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  // Preload row dependencies here to avoid a second fetch wave after the gallery mounts.
  useMountEffect(() => {
    void import('./GalleryImageActionsBridge');
  });
  const adapter = useMemo<GalleryUiAdapter>(
    () => ({
      antialiasProgressImages,
      exportProject,
      gallery: {
        ...gallery,
        updateSettings: (settings) => {
          if (isAccountScopeCurrent(accountScope) && queries.isActiveProject(projectId)) {
            gallery.updateSettings(settings, projectId);
          }
        },
      },
      galleryValues,
      generateValues,
      getItemLabel,
      ItemActionsProvider: GalleryItemActionsAdapter,
      ImageContextMenu: GalleryImageContextMenu,
      liveFollowEnabled,
      progressSessions: livePreview.gallerySessions,
      pinnedProgressSessionId: livePreview.pinnedSessionId,
      followedProgressSessionId: livePreview.followedSessionId,
      followProgressSession: (sessionId, { revealPreview }) => {
        if (!isAccountScopeCurrent(accountScope) || !queries.isActiveProject(projectId)) {
          return;
        }
        livePreview.follow(sessionId);
        if (revealPreview) {
          openWorkbenchWidget('preview');
        }
      },
      notifications,
      projectId,
      projectName,
      widgets: {
        openGallery: () => openWorkbenchWidget('gallery').ok,
        patchGalleryValues: (values) => widgets.patchValues('gallery', values),
      },
    }),
    [
      accountScope,
      antialiasProgressImages,
      exportProject,
      gallery,
      galleryValues,
      generateValues,
      liveFollowEnabled,
      livePreview,
      notifications,
      openWorkbenchWidget,
      projectId,
      projectName,
      queries,
      widgets,
    ]
  );

  return <GalleryUiProvider adapter={adapter}>{children}</GalleryUiProvider>;
};

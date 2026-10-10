import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryProjectRef, GalleryUiAdapter } from '@features/gallery/react';
import type { ProjectLibrarySnapshot } from '@workbench/projects/library';
import type { ReactNode } from 'react';

import { GalleryUiProvider } from '@features/gallery/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useProjectLibrarySelector } from '@workbench/projects/library';
import { useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { useLivePreviewFollow } from '@workbench/widgets/preview/livePreviewFollow';
import { getProjectWidgetInstance } from '@workbench/widgetState';
import {
  useActiveProjectSelector,
  useWorkbenchCommands,
  useWorkbenchInternalStore,
  useWorkbenchPersistenceService,
  useWorkbenchQueries,
} from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

const EMPTY_WIDGET_VALUES: Record<string, unknown> = Object.freeze({});

const selectGalleryProjects = (snapshot: ProjectLibrarySnapshot): GalleryProjectRef[] =>
  snapshot.summaries.map((summary) => ({ id: summary.id, name: summary.name }));

const areGalleryProjectsEqual = (left: GalleryProjectRef[], right: GalleryProjectRef[]): boolean =>
  left.length === right.length &&
  left.every((project, index) => project.id === right[index]?.id && project.name === right[index]?.name);

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
  // Only what the gallery needs to label other projects' boards, compared structurally: every autosave ack and
  // library refresh rebuilds the summaries, and the gallery must not re-render for cover or timestamp churn.
  const projects = useProjectLibrarySelector(selectGalleryProjects, areGalleryProjectsEqual);
  const store = useWorkbenchInternalStore();
  const persistence = useWorkbenchPersistenceService();
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
        selectItem: (item, selectionPage) =>
          selectionPage === undefined ? gallery.selectItem(item) : gallery.selectItem(item, undefined, selectionPage),
        setItemMultiSelection: (itemKeys, primaryItem, selectionPage) =>
          selectionPage === undefined
            ? gallery.setItemMultiSelection(itemKeys, primaryItem)
            : gallery.setItemMultiSelection(itemKeys, primaryItem, undefined, selectionPage),
        toggleItemSelection: (item, nextPrimaryItem, selectionPage) =>
          selectionPage === undefined
            ? gallery.toggleItemSelection(item, nextPrimaryItem)
            : gallery.toggleItemSelection(item, nextPrimaryItem, undefined, selectionPage),
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
      projects,
      // A board created in or moved into a project needs the project to exist on the server, which a new project
      // does only after its first save; the canvas path makes the same request before its first write.
      ensureProjectOnServer: async () => {
        const project = store.getState().projects.find((candidate) => candidate.id === projectId);

        if (project) {
          await persistence.ensureProjectOnServer(project);
        }
      },
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
      persistence,
      projectId,
      projectName,
      projects,
      queries,
      store,
      widgets,
    ]
  );

  return <GalleryUiProvider adapter={adapter}>{children}</GalleryUiProvider>;
};

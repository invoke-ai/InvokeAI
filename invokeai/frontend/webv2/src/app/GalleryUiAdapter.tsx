import type { GalleryItemRef } from '@features/gallery/contracts';
import type { GalleryProjectRef, GalleryUiAdapter } from '@features/gallery/react';
import type { ProjectLibrarySnapshot } from '@workbench/projects/library';
import type { ReactNode } from 'react';

import { getGalleryAutoAddBoardId, getGallerySettings } from '@features/gallery/contracts';
import { GalleryUiProvider } from '@features/gallery/react';
import { useAuthSession } from '@features/identity';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { useWorkbenchFocus } from '@workbench/focusRegions';
import { useProjectLibrarySelector } from '@workbench/projects/library';
import { useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import {
  areWorkbenchPreferencesReady,
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
} from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { useOpenWorkbenchWidget } from '@workbench/useOpenWorkbenchWidget';
import { revealPlacedWidget } from '@workbench/widgetPlacementCommands';
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
import { useTranslation } from 'react-i18next';

const EMPTY_WIDGET_VALUES: Record<string, unknown> = Object.freeze({});

/** What earlier builds stored for following the selected board, their default. */
const LEGACY_AUTO_ADD_FOLLOW = 'follow';

const getGalleryValues = (project: Parameters<typeof getProjectWidgetInstance>[0] | null | undefined) =>
  (project && getProjectWidgetInstance(project, 'gallery')?.state?.values) ?? EMPTY_WIDGET_VALUES;

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
      liveFollowEnabled: project.settings.viewerAutoSwitch === 'progress',
    }));
  const livePreview = useLivePreviewFollow();
  const boardSharing = useAuthSession().multiuserEnabled;
  const { gallery, notifications, widgets } = useWorkbenchCommands();
  const notify = useNotify();
  const { t } = useTranslation();
  const queries = useWorkbenchQueries();
  const accountScope = captureAccountScope();
  const exportProject = useExportLibraryProject();
  const openWorkbenchWidget = useOpenWorkbenchWidget();
  const { focusFloating, focusRegion } = useWorkbenchFocus();
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
      boardSharing,
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
        updateSettings: (settings, targetProjectId) => {
          if (!isAccountScopeCurrent(accountScope)) {
            return;
          }

          if (targetProjectId !== undefined) {
            if (queries.getProject(targetProjectId)) {
              gallery.updateSettings(settings, targetProjectId);
            }
          } else if (queries.isActiveProject(projectId)) {
            gallery.updateSettings(settings, projectId);
          }
        },
      },
      galleryValues,
      generateValues,
      getItemLabel,
      noteBoardBrowsed: (boardId) => {
        const values = getGalleryValues(queries.getProject(projectId));

        // Once per account, for a project that followed the selected board as earlier builds did by default (it
        // stored 'follow', now read as the default destination): say that browsing no longer moves results, the first
        // time the user browses away from where they go.
        if (
          values.autoAddBoardId === LEGACY_AUTO_ADD_FOLLOW &&
          !getGallerySettings(values).followSelectedBoard &&
          boardId !== (getGalleryAutoAddBoardId(values, () => true) ?? 'none') &&
          areWorkbenchPreferencesReady() &&
          !getWorkbenchPreferences().galleryDestinationNoticeShown
        ) {
          void patchWorkbenchPreferences({ galleryDestinationNoticeShown: true });
          notifications.add({
            kind: 'info',
            message: t('widgets.gallery.destinationNoticeMessage'),
            title: t('widgets.gallery.destinationNoticeTitle'),
          });
        }
      },
      getProjectGalleryValues: (targetProjectId) => {
        const project = isAccountScopeCurrent(accountScope) ? queries.getProject(targetProjectId) : null;

        return project ? getGalleryValues(project) : null;
      },
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
        if (!revealPreview) {
          return;
        }
        // Reveal Preview where it already is; the default open would pull it into the center over the Gallery.
        const revealed = revealPlacedWidget({
          project: queries.getSnapshot().activeProject,
          typeId: 'preview',
          widgets,
        });
        if (!revealed) {
          openWorkbenchWidget('preview');
        } else if (revealed.kind === 'floating') {
          focusFloating(revealed.instanceId);
        } else {
          focusRegion(revealed.region, 'preview');
        }
      },
      notifications: {
        add: notifications.add,
        reportError: notifications.reportError,
        undoable: (title, notice) => notify.undoable(title, { ...notice, projectId }),
      },
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
      boardSharing,
      exportProject,
      focusFloating,
      focusRegion,
      gallery,
      galleryValues,
      generateValues,
      liveFollowEnabled,
      livePreview,
      notifications,
      notify,
      openWorkbenchWidget,
      persistence,
      projectId,
      projectName,
      projects,
      queries,
      store,
      t,
      widgets,
    ]
  );

  return <GalleryUiProvider adapter={adapter}>{children}</GalleryUiProvider>;
};

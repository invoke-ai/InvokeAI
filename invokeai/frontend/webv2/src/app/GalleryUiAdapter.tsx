import type { GalleryUiAdapter } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryUiProvider } from '@features/gallery/react';
import { useActiveProgressTarget } from '@features/queue/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useExportLibraryProject } from '@workbench/projects/useProjectFileActions';
import { getProjectWidgetValues } from '@workbench/widgetState';
import {
  useActiveProjectId,
  useActiveProjectName,
  useActiveProjectSelector,
  useWidgetValuesSelector,
  useWorkbenchCommands,
} from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

const selectWidgetValues = (values: Record<string, unknown>): Record<string, unknown> => values;

const GalleryItemActionsAdapter = lazy(() =>
  import('./GalleryImageActionsBridge').then((module) => ({ default: module.GalleryItemActionsAdapter }))
);
const GalleryImageContextMenu = lazy(() =>
  import('./GalleryImageActionsBridge').then((module) => ({ default: module.GalleryImageContextMenu }))
);

/**
 * Production binding of Gallery's UI port: translates Gallery UI intents into
 * the Workbench aggregate. No second adapter is expected.
 */
export const GalleryUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const projectId = useActiveProjectId();
  const projectName = useActiveProjectName();
  const galleryValues = useActiveProjectSelector((project) => getProjectWidgetValues(project, 'gallery'));
  const generateValues = useWidgetValuesSelector('generate', selectWidgetValues);
  const queueItems = useActiveProjectSelector((project) => project.queue.items);
  const antialiasProgressImages = useActiveProjectSelector((project) => project.settings.antialiasProgressImages);
  const liveFollowEnabled = useActiveProjectSelector((project) => project.settings.showProgressImagesInViewer);
  const liveProgressTarget = useActiveProgressTarget();
  const { account, gallery, notifications, widgets } = useWorkbenchCommands();
  const exportProject = useExportLibraryProject();
  // These are `lazy()` children of an adapter that only ever mounts in the
  // editor, and the gallery widget needs them as soon as it renders a row.
  // Left to Suspense they were fetched at ~476ms — a full round trip after the
  // boot widget wave had already finished.
  useMountEffect(() => {
    void import('./GalleryImageActionsBridge');
  });
  const adapter = useMemo<GalleryUiAdapter>(
    () => ({
      account: {
        enableLiveFollow: () => account.updateProjectPreferences({ showProgressImagesInViewer: true }),
      },
      antialiasProgressImages,
      exportProject,
      gallery,
      galleryValues,
      generateValues,
      ItemActionsProvider: GalleryItemActionsAdapter,
      ImageContextMenu: GalleryImageContextMenu,
      liveFollowEnabled,
      liveProgressTarget,
      notifications,
      projectId,
      projectName,
      queueItems,
      widgets: { patchGalleryValues: (values) => widgets.patchValues('gallery', values) },
    }),
    [
      account,
      antialiasProgressImages,
      exportProject,
      gallery,
      galleryValues,
      generateValues,
      liveFollowEnabled,
      liveProgressTarget,
      notifications,
      projectId,
      projectName,
      queueItems,
      widgets,
    ]
  );

  return <GalleryUiProvider adapter={adapter}>{children}</GalleryUiProvider>;
};

import type { GalleryUiAdapter } from '@features/gallery/react';
import type { ReactNode } from 'react';

import { GalleryUiProvider } from '@features/gallery/react';
import { useProgressImage } from '@features/queue/react';
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

const GalleryImageActionsAdapter = lazy(() =>
  import('./GalleryImageActionsBridge').then((module) => ({ default: module.GalleryImageActionsAdapter }))
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
  const liveProgressTarget = useProgressImage()?.target ?? null;
  const { account, gallery, notifications, widgets } = useWorkbenchCommands();
  const adapter = useMemo<GalleryUiAdapter>(
    () => ({
      account: {
        enableLiveFollow: () => account.updateProjectPreferences({ showProgressImagesInViewer: true }),
      },
      antialiasProgressImages,
      gallery,
      galleryValues,
      generateValues,
      ImageActionsProvider: GalleryImageActionsAdapter,
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

import type { GenerationUiAdapter } from '@features/generation/react';
import type { ReactNode } from 'react';

import { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import { GenerationUiProvider } from '@features/generation/react';
import { ensureModelsLoaded, getModelBaseColorPalette, getModelBaseLabel, useModelsSelector } from '@features/models';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const GenerateCanvasCompositingSection = lazy(() =>
  import('@workbench/widgets/canvas/GenerateCanvasCompositingSection').then((module) => ({
    default: module.GenerateCanvasCompositingSection,
  }))
);

/**
 * Production binding of Generation's UI port: builds each sub-port from
 * Workbench, Models, and Gallery state. No second adapter is expected.
 */
export const GenerationUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector((activeProject) => ({
    activeProjectId: activeProject.id,
    generateValues: getProjectWidgetValues(activeProject, 'generate'),
    invocationSourceId: activeProject.invocation.sourceId,
    showPromptSyntaxHighlighting: activeProject.settings.showPromptSyntaxHighlighting,
  }));
  const promptHistoryItems = useActiveProjectSelector((activeProject) => activeProject.promptHistory);
  const selectedGalleryImage = useActiveProjectSelector((activeProject) =>
    getSelectedGalleryImageFromValues(getProjectWidgetValues(activeProject, 'gallery'))
  );
  const modelsCatalog = useModelsSelector((snapshot) => snapshot.models);
  const modelsError = useModelsSelector((snapshot) => snapshot.error);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const { gallery, generation, notifications } = useWorkbenchCommands();
  const notify = useNotify();

  const galleryGroup = useMemo<GenerationUiAdapter['gallery']>(
    () => ({
      selectedImage: selectedGalleryImage,
      touchImages: () => gallery.touchImages(),
    }),
    [gallery, selectedGalleryImage]
  );
  const modelsGroup = useMemo<GenerationUiAdapter['models']>(
    () => ({
      ModelSelect,
      catalog: modelsCatalog,
      ensureLoaded: ensureModelsLoaded,
      error: modelsError,
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      status: modelsStatus,
    }),
    [modelsCatalog, modelsError, modelsStatus]
  );
  const notificationsGroup = useMemo<GenerationUiAdapter['notifications']>(
    () => ({ error: notify.error, info: notify.info, reportError: notifications.reportError }),
    [notifications.reportError, notify.error, notify.info]
  );
  const promptHistoryGroup = useMemo<GenerationUiAdapter['promptHistory']>(
    () => ({
      clear: () => generation.clearPromptHistory(),
      items: promptHistoryItems,
      remove: generation.removePromptFromHistory,
    }),
    [generation, promptHistoryItems]
  );
  const settingsGroup = useMemo<GenerationUiAdapter['settings']>(
    () => ({ patchGenerateSettings: generation.patchSettings }),
    [generation]
  );

  const adapter = useMemo<GenerationUiAdapter>(
    () => ({
      CanvasCompositingSection: GenerateCanvasCompositingSection,
      gallery: galleryGroup,
      models: modelsGroup,
      notifications: notificationsGroup,
      project,
      promptHistory: promptHistoryGroup,
      settings: settingsGroup,
    }),
    [galleryGroup, modelsGroup, notificationsGroup, project, promptHistoryGroup, settingsGroup]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

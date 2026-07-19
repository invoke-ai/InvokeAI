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

export const GenerationUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  const project = useActiveProjectSelector((activeProject) => ({
    activeProjectId: activeProject.id,
    canvasValues: getProjectWidgetValues(activeProject, 'canvas'),
    generateValues: getProjectWidgetValues(activeProject, 'generate'),
    invocationSourceId: activeProject.invocation.sourceId,
    promptHistory: activeProject.promptHistory,
    selectedGalleryImage: getSelectedGalleryImageFromValues(getProjectWidgetValues(activeProject, 'gallery')),
    showPromptSyntaxHighlighting: activeProject.settings.showPromptSyntaxHighlighting,
  }));
  const models = useModelsSelector((snapshot) => snapshot.models);
  const modelsError = useModelsSelector((snapshot) => snapshot.error);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const { gallery, generation, notifications, widgets } = useWorkbenchCommands();
  const notify = useNotify();
  const adapter = useMemo<GenerationUiAdapter>(
    () => ({
      ...project,
      CanvasCompositingSection: GenerateCanvasCompositingSection,
      ModelSelect,
      clearPromptHistory: () => generation.clearPromptHistory(),
      ensureModelsLoaded,
      getModelBaseColorPalette,
      getModelBaseLabel,
      models,
      modelsError,
      modelsStatus,
      notifications: { error: notify.error, info: notify.info, reportError: notifications.reportError },
      patchCanvasValues: (values) => widgets.patchValues('canvas', values),
      patchGenerateSettings: generation.patchSettings,
      removePromptFromHistory: generation.removePromptFromHistory,
      touchGalleryImages: () => gallery.touchImages(),
    }),
    [
      gallery,
      generation,
      models,
      modelsError,
      modelsStatus,
      notifications.reportError,
      notify.error,
      notify.info,
      project,
      widgets,
    ]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

import type { GenerationUiAdapter } from '@features/generation/react';
import type { ReactNode } from 'react';

import { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import { invalidateGallery } from '@features/gallery/queries';
import { GenerationUiProvider } from '@features/generation/react';
import { normalizeRebalancePresets } from '@features/generation/settings';
import { useAuthSession, useCapabilities } from '@features/identity';
import { ensureModelsLoaded, getModelBaseColorPalette, getModelBaseLabel, useModelsSelector } from '@features/models';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQueryClient } from '@tanstack/react-query';
import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { lazy, useMemo } from 'react';

export const getGenerationSelectedGalleryImage = getSelectedGalleryImageFromValues;

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
  // The generate widget needs its model picker as soon as it renders. Left to
  // Suspense, `ModelSelect` was fetched in a second wave after the boot
  // widget wave had already finished. `GenerateCanvasCompositingSection` is
  // canvas-only and stays lazy — warming it would add bytes to every boot.
  useMountEffect(() => {
    void import('@features/models/react');
  });
  const projectState = useActiveProjectSelector((activeProject) => ({
    activeProjectId: activeProject.id,
    generateValues: getProjectWidgetValues(activeProject, 'generate'),
    invocationSourceId: activeProject.invocation.sourceId,
  }));
  // Syntax highlighting is a per-user preference, not a property of the
  // project, so it is joined here rather than read off the document.
  const showPromptSyntaxHighlighting = useWorkbenchPreferenceSelector(
    (preferences) => preferences.showPromptSyntaxHighlighting
  );
  const project = useMemo<GenerationUiAdapter['project']>(
    () => ({ ...projectState, showPromptSyntaxHighlighting }),
    [projectState, showPromptSyntaxHighlighting]
  );
  const promptHistoryItems = useActiveProjectSelector((activeProject) => activeProject.promptHistory);
  const selectedGalleryImage = useActiveProjectSelector((activeProject) =>
    getGenerationSelectedGalleryImage(getProjectWidgetValues(activeProject, 'gallery'))
  );
  const modelsCatalog = useModelsSelector((snapshot) => snapshot.models);
  const modelsError = useModelsSelector((snapshot) => snapshot.error);
  const modelsStatus = useModelsSelector((snapshot) => snapshot.status);
  const { generation, notifications } = useWorkbenchCommands();
  const session = useAuthSession();
  const queryClient = useQueryClient();
  const notify = useNotify();
  const galleryGroup = useMemo<GenerationUiAdapter['gallery']>(
    () => ({
      selectedImage: selectedGalleryImage,
      touchImages: () => void invalidateGallery(queryClient),
    }),
    [queryClient, selectedGalleryImage]
  );
  const modelsGroup = useMemo<GenerationUiAdapter['models']>(
    () => ({
      ModelSelect,
      catalog: modelsCatalog,
      ensureLoaded: ensureModelsLoaded,
      error: modelsError,
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      // Hash navigation, matching the app.selectModelsTab command. Going through
      // useNavigate or the models UI store would pull either the router hooks or
      // the store into the editor/launchpad initial bundles (architecture budget).
      // The manager opens on Add Models by default, which is where this link wants to land.
      openManager: () => {
        window.location.hash = `#/models?project=${encodeURIComponent(project.activeProjectId)}`;
      },
      status: modelsStatus,
    }),
    [modelsCatalog, modelsError, modelsStatus, project.activeProjectId]
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
  const { canManagePromptTemplates } = useCapabilities();
  const capabilitiesGroup = useMemo<GenerationUiAdapter['capabilities']>(
    () => ({ canManagePromptTemplates }),
    [canManagePromptTemplates]
  );
  const accountGroup = useMemo<GenerationUiAdapter['account']>(
    () => ({
      currentUserId: session.user?.user_id ?? null,
      multiuserEnabled: session.multiuserEnabled,
    }),
    [session.multiuserEnabled, session.user?.user_id]
  );
  const krea2RebalancePresets = useWorkbenchPreferenceSelector((preferences) => preferences.krea2RebalancePresets);
  const rebalancePresetsGroup = useMemo<GenerationUiAdapter['rebalancePresets']>(
    () => ({
      // Settings persists these shape-checked only; a curve whose weights no longer parse
      // (hand-edited storage, a future backend tap count) is dropped here rather than
      // handed to the picker.
      presets: normalizeRebalancePresets(krea2RebalancePresets),
      remove: (presetId) => {
        void patchWorkbenchPreferences({
          krea2RebalancePresets: getWorkbenchPreferences().krea2RebalancePresets.filter(
            (preset) => preset.id !== presetId
          ),
        });
      },
      rename: (presetId, label) => {
        void patchWorkbenchPreferences({
          krea2RebalancePresets: getWorkbenchPreferences().krea2RebalancePresets.map((preset) =>
            preset.id === presetId ? { ...preset, label } : preset
          ),
        });
      },
      save: (label, weights, multiplier) => {
        const preset = { id: crypto.randomUUID(), label, multiplier, weights };

        void patchWorkbenchPreferences({
          krea2RebalancePresets: [...getWorkbenchPreferences().krea2RebalancePresets, preset],
        });

        return preset;
      },
    }),
    [krea2RebalancePresets]
  );
  const generateSectionsOpen = useWorkbenchPreferenceSelector((preferences) => preferences.generateSectionsOpen);
  const sectionPreferencesGroup = useMemo<GenerationUiAdapter['sectionPreferences']>(
    () => ({
      sectionsOpen: generateSectionsOpen,
      setSectionOpen: (sectionId, open) => {
        void patchWorkbenchPreferences({
          generateSectionsOpen: { ...getWorkbenchPreferences().generateSectionsOpen, [sectionId]: open },
        });
      },
    }),
    [generateSectionsOpen]
  );

  const adapter = useMemo<GenerationUiAdapter>(
    () => ({
      CanvasCompositingSection: GenerateCanvasCompositingSection,
      account: accountGroup,
      capabilities: capabilitiesGroup,
      gallery: galleryGroup,
      models: modelsGroup,
      notifications: notificationsGroup,
      project,
      promptHistory: promptHistoryGroup,
      rebalancePresets: rebalancePresetsGroup,
      sectionPreferences: sectionPreferencesGroup,
      settings: settingsGroup,
    }),
    [
      accountGroup,
      capabilitiesGroup,
      galleryGroup,
      modelsGroup,
      notificationsGroup,
      project,
      promptHistoryGroup,
      rebalancePresetsGroup,
      sectionPreferencesGroup,
      settingsGroup,
    ]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

import type { GenerationUiAdapter } from '@features/generation/react';
import type { QueueItemReadModel, QueueReadModel } from '@features/queue/contracts';
import type { ReactNode } from 'react';

import { getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import { invalidateGallery } from '@features/gallery/queries';
import { galleryImageUrls } from '@features/gallery/utility';
import { GenerationUiProvider } from '@features/generation/react';
import { normalizeRebalancePresets } from '@features/generation/settings';
import { useAuthSession, useCapabilities } from '@features/identity';
import { ensureModelsLoaded, getModelBaseColorPalette, getModelBaseLabel, useModelsSelector } from '@features/models';
import {
  buildProjectQueueItemOriginPrefix,
  extractGenerationMeta,
  getResultImageName,
} from '@features/queue/contracts';
import { getQueueReadModelOptions } from '@features/queue/queries';
import { useMountEffect } from '@platform/react/useMountEffect';
import { useQuery, useQueryClient } from '@tanstack/react-query';
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

const RECENT_RUN_WINDOW = 10;
const SEED_HISTORY_LIMIT = 6;

const selectQueueItems = (model: QueueReadModel): QueueItemReadModel[] => model.items;

/**
 * Seeds and durations of recent completed Generate runs, derived from the
 * project-scoped backend queue read model. Backend items map to a local queue
 * item to prove they came from the Generate source (canvas, workflow, and
 * upscale runs share the same backend queue); the executed seed comes from the
 * backend item's session meta, which is authoritative for randomized runs.
 */
const useGenerationQueueInsights = (projectId: string): GenerationUiAdapter['queueInsights'] => {
  const localQueueItems = useActiveProjectSelector((activeProject) => activeProject.queue.items);
  const scope = useMemo(() => ({ originPrefix: buildProjectQueueItemOriginPrefix(projectId) }), [projectId]);
  const backendItems = useQuery({ ...getQueueReadModelOptions(scope), select: selectQueueItems }).data;

  return useMemo(() => {
    if (!backendItems) {
      return { secondsPerRun: null, seedHistory: [] };
    }

    const generateBackendIds = new Set<number>();

    for (const item of localQueueItems) {
      if (item.snapshot.sourceId === 'generate') {
        for (const backendId of item.backendItemIds ?? []) {
          generateBackendIds.add(backendId);
        }
      }
    }

    const completed = backendItems
      .filter((item) => item.status === 'completed' && generateBackendIds.has(item.id))
      .sort((a, b) => (b.completedAt ?? b.updatedAt).localeCompare(a.completedAt ?? a.updatedAt));

    const seenSeeds = new Set<number>();
    const seedHistory: GenerationUiAdapter['queueInsights']['seedHistory'][number][] = [];

    for (const item of completed) {
      const seed = extractGenerationMeta(item).seed;

      if (seed === undefined || seenSeeds.has(seed)) {
        continue;
      }

      seenSeeds.add(seed);
      const imageName = getResultImageName(item);
      seedHistory.push({ seed, thumbnailUrl: imageName ? galleryImageUrls.thumbnail(imageName) : null });

      if (seedHistory.length >= SEED_HISTORY_LIMIT) {
        break;
      }
    }

    const durations = completed
      .slice(0, RECENT_RUN_WINDOW)
      .map((item) =>
        item.startedAt && item.completedAt ? (Date.parse(item.completedAt) - Date.parse(item.startedAt)) / 1000 : null
      )
      .filter((seconds): seconds is number => seconds !== null && Number.isFinite(seconds) && seconds > 0);
    const secondsPerRun =
      durations.length === 0 ? null : durations.reduce((total, seconds) => total + seconds, 0) / durations.length;

    return { secondsPerRun, seedHistory };
  }, [backendItems, localQueueItems]);
};

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
  const generatePresets = useWorkbenchPreferenceSelector((preferences) => preferences.generatePresets);
  const presetsGroup = useMemo<GenerationUiAdapter['presets']>(
    () => ({
      // Settings persists these shape-checked only; the feature re-normalizes the
      // snapshot against the current model catalog when a preset is applied.
      presets: generatePresets,
      remove: (presetId) => {
        void patchWorkbenchPreferences({
          generatePresets: getWorkbenchPreferences().generatePresets.filter((preset) => preset.id !== presetId),
        });
      },
      rename: (presetId, label) => {
        void patchWorkbenchPreferences({
          generatePresets: getWorkbenchPreferences().generatePresets.map((preset) =>
            preset.id === presetId ? { ...preset, label } : preset
          ),
        });
      },
      save: (label, values) => {
        const preset = { id: crypto.randomUUID(), label, values };

        void patchWorkbenchPreferences({
          generatePresets: [...getWorkbenchPreferences().generatePresets, preset],
        });

        return preset;
      },
    }),
    [generatePresets]
  );
  const queueInsightsGroup = useGenerationQueueInsights(project.activeProjectId);
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
      presets: presetsGroup,
      project,
      promptHistory: promptHistoryGroup,
      queueInsights: queueInsightsGroup,
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
      presetsGroup,
      project,
      promptHistoryGroup,
      queueInsightsGroup,
      rebalancePresetsGroup,
      sectionPreferencesGroup,
      settingsGroup,
    ]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

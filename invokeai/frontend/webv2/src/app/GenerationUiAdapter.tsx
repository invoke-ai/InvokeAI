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
import { getQueueReadModelOptions } from '@features/queue';
import {
  buildProjectQueueItemOriginPrefix,
  extractGenerationMeta,
  getResultImageName,
} from '@features/queue/contracts';
import { createUuid } from '@platform/browser/randomUuid';
import { useMountEffect } from '@platform/react/useMountEffect';
import { createProjectedExternalStore } from '@platform/state/projectedExternalStore';
import { shallowEqual } from '@platform/state/selectors';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { lazy, useCallback, useMemo } from 'react';

export const getGenerationSelectedGalleryImage = getSelectedGalleryImageFromValues;

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const GenerateCanvasSections = lazy(() =>
  import('@workbench/widgets/canvas/GenerateCanvasSections').then((module) => ({
    default: module.GenerateCanvasSections,
  }))
);

const RECENT_RUN_WINDOW = 10;
const SEED_HISTORY_LIMIT = 6;

const selectQueueItems = (model: QueueReadModel): QueueItemReadModel[] => model.items;

/**
 * Join backend items to local Generate items to exclude other queue sources; use executed session-meta seeds for
 * randomized runs.
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

export const GenerationUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  // Preload the always-needed model picker; leave canvas-only sections lazy.
  useMountEffect(() => {
    void import('@features/models/react');
  });
  const store = useWorkbenchInternalStore();
  const generateValues = useMemo(
    () =>
      createProjectedExternalStore({
        isEqual: shallowEqual<Record<string, unknown>>,
        select: (snapshot) => getProjectWidgetValues(snapshot.activeProject, 'generate'),
        source: store,
      }),
    [store]
  );
  const projectState = useActiveProjectSelector((activeProject) => ({
    activeProjectId: activeProject.id,
    invocationSourceId: activeProject.invocation.sourceId,
  }));
  // Syntax highlighting is an account preference, not project data.
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
  const findGalleryItem = useFindGalleryItem();
  // Keep this handler stable across gallery selection changes so memoized reference cards do not all rerender.
  const findImage = useCallback<GenerationUiAdapter['gallery']['findImage']>(
    (imageName) => findGalleryItem({ kind: 'image', name: imageName }),
    [findGalleryItem]
  );
  const galleryGroup = useMemo<GenerationUiAdapter['gallery']>(
    () => ({
      findImage,
      selectedImage: selectedGalleryImage,
      touchImages: () => void invalidateGallery(queryClient),
    }),
    [findImage, queryClient, selectedGalleryImage]
  );
  const modelsGroup = useMemo<GenerationUiAdapter['models']>(
    () => ({
      ModelSelect,
      catalog: modelsCatalog,
      ensureLoaded: ensureModelsLoaded,
      error: modelsError,
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      // Use hash navigation and lazy filter seeding to keep router/manager code out of initial bundles; set the
      // filter before navigation.
      openManager: (options) => {
        const navigateToManager = () => {
          window.location.hash = `#/models?project=${encodeURIComponent(project.activeProjectId)}`;
        };
        const modelType = options?.modelType;

        if (modelType === undefined) {
          navigateToManager();
          return;
        }

        void import('@features/models/launchpad').then(({ requestAddModelsTypeFilter }) => {
          requestAddModelsTypeFilter(modelType);
          navigateToManager();
        });
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
  const { canManagePromptTemplates, canManageSharedSystemPrompts } = useCapabilities();
  const capabilitiesGroup = useMemo<GenerationUiAdapter['capabilities']>(
    () => ({ canManagePromptTemplates, canManageSharedSystemPrompts }),
    [canManagePromptTemplates, canManageSharedSystemPrompts]
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
      // Stored curves are only shape-checked; discard weights incompatible with the current parser.
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
        const preset = { id: createUuid(), label, multiplier, weights };

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
      // Presets are shape-checked in storage and normalized against current models on application.
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
        const preset = { id: createUuid(), label, values };

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
      CanvasGenerationSections: GenerateCanvasSections,
      account: accountGroup,
      capabilities: capabilitiesGroup,
      gallery: galleryGroup,
      generateValues,
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
      generateValues,
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

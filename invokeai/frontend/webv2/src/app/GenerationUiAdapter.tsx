import type { GenerationUiAdapter } from '@features/generation/react';
import type { QueueItem, QueueItemReadModel } from '@features/queue/contracts';
import type { ReadableExternalStore } from '@platform/state/projectedExternalStore';
import type { ReactNode } from 'react';

import { getGalleryAutoAddBoardId, getSelectedGalleryImageFromValues } from '@features/gallery/contracts';
import {
  canAddToCachedGalleryBoard,
  ensureGalleryBoardDestinations,
  invalidateGallery,
} from '@features/gallery/queries';
import { galleryImageUrls } from '@features/gallery/utility';
import { clearGenerateModelSwitchConfirmation, GenerationUiProvider } from '@features/generation/react';
import { normalizeRebalancePresets } from '@features/generation/settings';
import { useAuthSession, useCapabilities } from '@features/identity';
import {
  ensureModelsLoaded,
  refreshModels,
  getModelBaseColorPalette,
  getModelBaseLabel,
  getModelImageUrl,
  openModelManager,
  useModelsSelector,
  useOpenModelInManager,
} from '@features/models';
import { getQueueReadModelOptions } from '@features/queue';
import {
  buildProjectQueueItemOriginPrefix,
  extractGenerationMeta,
  getResultImageName,
} from '@features/queue/contracts';
import { createUuid } from '@platform/browser/randomUuid';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { createProjectedExternalStore } from '@platform/state/projectedExternalStore';
import { shallowEqual } from '@platform/state/selectors';
import { createActionToast } from '@platform/ui/toaster';
import { type QueryClient, QueryObserver, useQueryClient } from '@tanstack/react-query';
import { useFindGalleryItem } from '@workbench/image-actions/useFindGalleryItem';
import {
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  useWorkbenchPreferenceSelector,
} from '@workbench/settings/store';
import { useNotify } from '@workbench/useNotify';
import { getProjectWidgetValues, getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands, useWorkbenchInternalStore } from '@workbench/WorkbenchContext';
import { lazy, useCallback, useMemo } from 'react';

export const getGenerationSelectedGalleryImage = getSelectedGalleryImageFromValues;

const ModelSelect = lazy(() => import('@features/models/react').then((module) => ({ default: module.ModelSelect })));
const loadCanvasGenerateSlots = () => import('@workbench/widgets/canvas/GenerateCanvasSections');
const GenerateCanvasSections = lazy(() =>
  loadCanvasGenerateSlots().then((module) => ({ default: module.GenerateCanvasSections }))
);
const GenerateCanvasRenderSize = lazy(() =>
  loadCanvasGenerateSlots().then((module) => ({ default: module.GenerateCanvasRenderSize }))
);
const GenerateDenoisingStrength = lazy(() =>
  loadCanvasGenerateSlots().then((module) => ({ default: module.GenerateDenoisingStrength }))
);

const RECENT_RUN_WINDOW = 10;
const SEED_HISTORY_LIMIT = 6;

type GenerationQueueInsights = ReturnType<GenerationUiAdapter['queueInsights']['getSnapshot']>;

/** A removed preset put back at its old position (or last), or null while one with its id is still saved. */
const reinsertPreset = <Preset extends { id: string }>(
  presets: readonly Preset[],
  preset: Preset,
  index: number
): Preset[] | null => {
  if (presets.some((candidate) => candidate.id === preset.id)) {
    return null;
  }

  const next = [...presets];

  next.splice(Math.min(Math.max(index, 0), next.length), 0, preset);
  return next;
};

const EMPTY_QUEUE_INSIGHTS: GenerationQueueInsights = { secondsPerRun: null, seedHistory: [] };

/**
 * Join backend items to local Generate items to exclude other queue sources; use executed session-meta seeds for
 * randomized runs.
 */
const getQueueInsights = (
  backendItems: readonly QueueItemReadModel[],
  localQueueItems: readonly QueueItem[]
): GenerationQueueInsights => {
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
  const seedHistory: GenerationQueueInsights['seedHistory'][number][] = [];

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
};

/**
 * The project's queue read model is observed only while a Generate control subscribes, so a workbench without one
 * neither fetches it nor refetches it on every queue invalidation. Unsubscribed reads use whatever is cached.
 *
 * Insights derive from the read model's items and the local queue alone, keyed on their identities: structural
 * sharing keeps `items` stable across status-only refetches, so those leave the snapshot untouched. Every source
 * change notifies every subscriber; the subscribers' selectors drop the ones that change nothing they read. Comparing
 * against the cached snapshot here instead would miss a change that a render had already pulled into the cache.
 */
const createQueueInsightsStore = (
  queryClient: QueryClient,
  localQueueItems: ReadableExternalStore<readonly QueueItem[]>,
  projectId: string
): ReadableExternalStore<GenerationQueueInsights> => {
  const scope = { originPrefix: buildProjectQueueItemOriginPrefix(projectId) };
  const { queryKey } = getQueueReadModelOptions(scope);
  const listeners = new Set<() => void>();
  let stopObserving: (() => void) | null = null;
  let inputs: { backendItems: readonly QueueItemReadModel[] | undefined; local: readonly QueueItem[] } | null = null;
  let snapshot = EMPTY_QUEUE_INSIGHTS;

  const getSnapshot = (): GenerationQueueInsights => {
    const backendItems = queryClient.getQueryData(queryKey)?.items;
    const local = localQueueItems.getSnapshot();

    if (inputs === null || inputs.backendItems !== backendItems || inputs.local !== local) {
      inputs = { backendItems, local };
      snapshot = backendItems ? getQueueInsights(backendItems, local) : EMPTY_QUEUE_INSIGHTS;
    }

    return snapshot;
  };
  const notify = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };

  return {
    getSnapshot,
    subscribe: (listener) => {
      listeners.add(listener);

      if (!stopObserving) {
        // Options are built per observation so the read captures the account scope current at subscription.
        const stopQuery = new QueryObserver(queryClient, getQueueReadModelOptions(scope)).subscribe(notify);
        const stopLocal = localQueueItems.subscribe(notify);

        stopObserving = () => {
          stopQuery();
          stopLocal();
        };
      }

      return () => {
        listeners.delete(listener);

        if (listeners.size === 0) {
          stopObserving?.();
          stopObserving = null;
        }
      };
    },
  };
};

export const GenerationUiAdapterProvider = ({ children }: { children: ReactNode }) => {
  // Preload the always-needed model picker; leave canvas-only sections lazy.
  useMountEffect(() => {
    void import('@features/models/react');
  });
  const store = useWorkbenchInternalStore();
  // A model switch waiting for confirmation belongs to the project it was asked in; leaving that project drops it, so
  // the dialog never pops up later in another context.
  useMountEffect(() => {
    let activeProjectId = store.getSnapshot().activeProject.id;

    return store.subscribe(() => {
      const next = store.getSnapshot().activeProject.id;

      if (next !== activeProjectId) {
        activeProjectId = next;
        clearGenerateModelSwitchConfirmation();
      }
    });
  });
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
  const queueInsights = useMemo(
    () =>
      createQueueInsightsStore(
        queryClient,
        createProjectedExternalStore({ select: (snapshot) => snapshot.activeProject.queue.items, source: store }),
        project.activeProjectId
      ),
    [project.activeProjectId, queryClient, store]
  );
  const notify = useNotify();
  const findGalleryItem = useFindGalleryItem();
  // Keep this handler stable across gallery selection changes so memoized reference cards do not all rerender.
  const findImage = useCallback<GenerationUiAdapter['gallery']['findImage']>(
    (imageName) => findGalleryItem({ kind: 'image', name: imageName }),
    [findGalleryItem]
  );
  // Save to Uploads lands where new results do: the auto-add board, judged against the listing of every readable
  // board, ensured as the upload starts so an archived or revoked board is decided from a listing, not for want of
  // one, and no listing is read while nothing uploads; a board the listing refuses falls back to the project's inbox.
  // The board is read from the project before anything is awaited, so a later change cannot redirect an upload
  // already under way.
  const getUploadBoardId = useCallback<GenerationUiAdapter['gallery']['getUploadBoardId']>(async () => {
    const { activeProject, scopedDefaults } = store.getSnapshot();
    const galleryValues = getResolvedProjectWidgetValues(activeProject, 'gallery', scopedDefaults);
    const pendingListing = ensureGalleryBoardDestinations(queryClient);

    if (pendingListing) {
      await pendingListing;
    }

    return (
      getGalleryAutoAddBoardId(galleryValues, (boardId) => canAddToCachedGalleryBoard(boardId, queryClient)) ?? 'none'
    );
  }, [queryClient, store]);
  const galleryGroup = useMemo<GenerationUiAdapter['gallery']>(
    () => ({
      findImage,
      getUploadBoardId,
      selectedImage: selectedGalleryImage,
      touchImages: () => void invalidateGallery(queryClient),
    }),
    [findImage, getUploadBoardId, queryClient, selectedGalleryImage]
  );
  const openInModelManager = useOpenModelInManager();
  const modelsGroup = useMemo<GenerationUiAdapter['models']>(
    () => ({
      ModelSelect,
      catalog: modelsCatalog,
      ensureLoaded: ensureModelsLoaded,
      refresh: () => refreshModels(),
      error: modelsError,
      getBaseColorPalette: getModelBaseColorPalette,
      getBaseLabel: getModelBaseLabel,
      getImageUrl: getModelImageUrl,
      openInModelManager: openInModelManager ?? undefined,
      openManager: (options) => openModelManager({ modelType: options?.modelType, projectId: project.activeProjectId }),
      status: modelsStatus,
    }),
    [modelsCatalog, modelsError, modelsStatus, openInModelManager, project.activeProjectId]
  );
  const notificationsGroup = useMemo<GenerationUiAdapter['notifications']>(
    () => ({
      error: notify.error,
      info: notify.info,
      offer: (title, { actions, message }) =>
        void createActionToast({
          actions,
          description: message,
          duration: Number.POSITIVE_INFINITY,
          signal: captureAccountScope().signal,
          title,
          type: 'info',
        }),
      reportError: notifications.reportError,
      undoable: notify.undoable,
    }),
    [notifications.reportError, notify.error, notify.info, notify.undoable]
  );
  const promptHistoryGroup = useMemo<GenerationUiAdapter['promptHistory']>(
    () => ({
      clear: generation.clearPromptHistory,
      items: promptHistoryItems,
      remove: generation.removePromptFromHistory,
      restore: generation.restorePromptHistory,
    }),
    [generation, promptHistoryItems]
  );
  const settingsGroup = useMemo<GenerationUiAdapter['settings']>(
    () => ({
      patchGenerateSettings: generation.patchSettings,
      readGenerateValues: (projectId) => {
        const project = store.getSnapshot().projects.find((candidate) => candidate.id === projectId);

        return project ? getProjectWidgetValues(project, 'generate') : null;
      },
    }),
    [generation, store]
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
      restore: (preset, index) => {
        const restored = reinsertPreset(getWorkbenchPreferences().krea2RebalancePresets, preset, index);

        if (restored) {
          void patchWorkbenchPreferences({ krea2RebalancePresets: restored });
        }
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
  const generatePresetsIncludePrompts = useWorkbenchPreferenceSelector(
    (preferences) => preferences.generatePresetsIncludePrompts
  );
  const presetsGroup = useMemo<GenerationUiAdapter['presets']>(
    () => ({
      includePrompts: generatePresetsIncludePrompts,
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
      restore: (preset, index) => {
        const restored = reinsertPreset(getWorkbenchPreferences().generatePresets, preset, index);

        if (restored) {
          void patchWorkbenchPreferences({ generatePresets: restored });
        }
      },
      save: (label, values) => {
        const preset = { id: createUuid(), label, values };

        void patchWorkbenchPreferences({
          generatePresets: [...getWorkbenchPreferences().generatePresets, preset],
        });

        return preset;
      },
      setIncludePrompts: (includePrompts) => {
        void patchWorkbenchPreferences({ generatePresetsIncludePrompts: includePrompts });
      },
    }),
    [generatePresets, generatePresetsIncludePrompts]
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
      CanvasDenoisingStrength: GenerateDenoisingStrength,
      CanvasGenerationSections: GenerateCanvasSections,
      CanvasRenderSize: GenerateCanvasRenderSize,
      account: accountGroup,
      capabilities: capabilitiesGroup,
      gallery: galleryGroup,
      generateValues,
      models: modelsGroup,
      notifications: notificationsGroup,
      presets: presetsGroup,
      project,
      promptHistory: promptHistoryGroup,
      queueInsights,
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
      queueInsights,
      rebalancePresetsGroup,
      sectionPreferencesGroup,
      settingsGroup,
    ]
  );

  return <GenerationUiProvider adapter={adapter}>{children}</GenerationUiProvider>;
};

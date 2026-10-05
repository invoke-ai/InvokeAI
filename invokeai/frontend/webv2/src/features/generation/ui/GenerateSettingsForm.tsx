import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import { Separator, Stack } from '@chakra-ui/react';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import {
  type ComponentType,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';
import { GenerateCanvasSections } from './GenerateCanvasSections';
import { GenerateComponentsSection } from './GenerateComponentsSection';
import {
  applyGenerateSettingsPatch,
  applyGenerateSettingsUpdate,
  createDraftTracker,
  type GenerateDraftStore,
  getChangedGenerateSettingsPatch,
  type GenerateSettingsUpdate,
  isDraftView,
  mergeGenerateSettingsUpdate,
  type PendingGenerateSettingsUpdate,
  reuseEqualGenerateSettingsValues,
} from './generateDebounce';
import { GenerateDimensionFields } from './GenerateDimensionFields';
import { useRegisterGenerateDraftFlusher } from './generateDraftRegistry';
import { getSettingsWithLatestPromptFields } from './generateFormViewModel';
import { GenerateGuidanceSection } from './GenerateGuidanceSection';
import { GenerateModelCard } from './GenerateModelCard';
import { GenerateRenderSection } from './GenerateRenderSection';
import { GeneratePromptFields } from './promptFields';

const GENERATE_INPUT_DEBOUNCE_MS = 250;

type DraftSectionProps<Props extends { settings: GenerateSettings }> = Omit<Props, 'settings'> & {
  draft: GenerateDraftStore;
  section: ComponentType<Props>;
};

/**
 * Scopes draft subscriptions per section so a scrub in one section does not re-render the others. The section's
 * `settings` is a live draft view; see `createDraftTracker` for what it may and may not be used for.
 */
const DraftSection = <Props extends { settings: GenerateSettings }>({
  draft,
  section: Section,
  ...props
}: DraftSectionProps<Props>) => {
  const [getView] = useState(() => createDraftTracker(draft));
  const settings = useSyncExternalStore(draft.subscribe, getView, getView);

  return <Section {...(props as unknown as Props)} settings={settings} />;
};

interface GenerateSettingsFormProps {
  isLoadingModels: boolean;
  loadError: string | null;
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  models: readonly ModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  supportedModels: GenerateModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
  /** Stable; takes the project the pending edits belong to, which can differ from a stale closure's. */
  onPatchSettings: (patch: Partial<GenerateSettings>, projectId: string) => void;
}

export const GenerateSettingsForm = ({
  isLoadingModels,
  loadError,
  loraModels,
  models,
  onCommitSettings,
  onPatchSettings,
  projectId,
  selectedModel,
  settings,
  supportedModels,
}: GenerateSettingsFormProps) => {
  const [draft] = useState(() => createExternalStoreCore(settings));
  const latestSettingsRef = useRef(settings);
  const pendingUpdateRef = useRef<PendingGenerateSettingsUpdate>(null);
  const projectIdRef = useRef(projectId);
  const timeoutRef = useRef<number | null>(null);

  const clearPendingUpdate = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    pendingUpdateRef.current = null;
  };

  useLayoutEffect(() => {
    if (projectIdRef.current !== projectId) {
      projectIdRef.current = projectId;
      clearPendingUpdate();
      latestSettingsRef.current = settings;
      draft.setSnapshot(settings);
      return;
    }

    latestSettingsRef.current = settings;
    draft.setSnapshot(
      reuseEqualGenerateSettingsValues(
        draft.getSnapshot(),
        applyGenerateSettingsUpdate(settings, pendingUpdateRef.current)
      )
    );
  }, [draft, projectId, settings]);

  const flushPendingUpdate = useCallback(
    (shouldUpdateDraft = true) => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      const updateToCommit = pendingUpdateRef.current;

      pendingUpdateRef.current = null;

      if (updateToCommit) {
        const previousSettings = latestSettingsRef.current;
        const settingsToCommit = applyGenerateSettingsUpdate(latestSettingsRef.current, updateToCommit);

        latestSettingsRef.current = settingsToCommit;

        if (shouldUpdateDraft) {
          draft.setSnapshot(reuseEqualGenerateSettingsValues(draft.getSnapshot(), settingsToCommit));
        }

        onPatchSettings(getChangedGenerateSettingsPatch(previousSettings, settingsToCommit), projectIdRef.current);
      }
    },
    [draft, onPatchSettings]
  );

  useRegisterGenerateDraftFlusher(flushPendingUpdate);

  // The patch port is stable, so this flushes on unmount.
  useEffect(
    () => () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }

      flushPendingUpdate(false);
    },
    [flushPendingUpdate]
  );

  const scheduleCommitUpdate = useCallback(
    (update: GenerateSettingsUpdate) => {
      pendingUpdateRef.current = mergeGenerateSettingsUpdate(pendingUpdateRef.current, update);

      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = window.setTimeout(() => {
        timeoutRef.current = null;
        flushPendingUpdate();
      }, GENERATE_INPUT_DEBOUNCE_MS);
    },
    [flushPendingUpdate]
  );

  const commit = useCallback(
    (update: GenerateSettingsUpdate) => {
      const nextSettings = applyGenerateSettingsUpdate(draft.getSnapshot(), mergeGenerateSettingsUpdate(null, update));

      draft.setSnapshot(nextSettings);
      scheduleCommitUpdate(update);
    },
    [draft, scheduleCommitUpdate]
  );

  const commitDebouncedDraftUpdate = useCallback(
    (update: GenerateSettingsUpdate) => {
      const pendingUpdate = mergeGenerateSettingsUpdate(pendingUpdateRef.current, update);
      const previousSettings = latestSettingsRef.current;
      const nextSettings = applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdate);

      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      pendingUpdateRef.current = null;
      latestSettingsRef.current = nextSettings;
      draft.setSnapshot(nextSettings);
      onPatchSettings(getChangedGenerateSettingsPatch(previousSettings, nextSettings), projectIdRef.current);
    },
    [draft, onPatchSettings]
  );

  const commitPromptDraftPatch = useCallback(
    (patch: Partial<GenerateSettings>) => {
      const pendingUpdate = mergeGenerateSettingsUpdate(pendingUpdateRef.current, patch);
      const previousSettings = latestSettingsRef.current;
      const nextSettings = applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdate);

      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      pendingUpdateRef.current = null;
      latestSettingsRef.current = nextSettings;
      onPatchSettings(getChangedGenerateSettingsPatch(previousSettings, nextSettings), projectIdRef.current);
    },
    [onPatchSettings]
  );

  const commitSettingsImmediately = useCallback(
    (requestedSettings: GenerateSettings) => {
      // A section may hand back its draft view unchanged; commit the plain draft it reads from.
      const nextSettings = isDraftView(requestedSettings) ? draft.getSnapshot() : requestedSettings;
      const previousSettings = latestSettingsRef.current;
      const settingsToCommit = getSettingsWithLatestPromptFields(
        nextSettings,
        applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdateRef.current)
      );

      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      pendingUpdateRef.current = null;
      latestSettingsRef.current = settingsToCommit;
      draft.setSnapshot(settingsToCommit);

      if (!Object.is(previousSettings, settingsToCommit)) {
        onCommitSettings(settingsToCommit);
      }
    },
    [draft, onCommitSettings]
  );

  const commitPatchImmediately = useCallback(
    (patch: Partial<GenerateSettings>) => {
      const previousSettings = latestSettingsRef.current;
      const nextSettings = applyGenerateSettingsPatch(
        applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdateRef.current),
        patch
      );

      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      pendingUpdateRef.current = null;
      latestSettingsRef.current = nextSettings;
      draft.setSnapshot(nextSettings);
      onPatchSettings(getChangedGenerateSettingsPatch(previousSettings, nextSettings), projectIdRef.current);
    },
    [draft, onPatchSettings]
  );

  return (
    <Stack gap={1} p={1}>
      <DraftSection
        draft={draft}
        section={GenerateModelCard}
        isLoadingModels={isLoadingModels}
        loadError={loadError}
        models={models}
        selectedModel={selectedModel}
        supportedModels={supportedModels}
        onCommitSettings={commitSettingsImmediately}
      />

      {/* The same hairline the collapsible sections draw between one another. */}
      <Separator />

      <DraftSection
        draft={draft}
        section={GeneratePromptFields}
        projectId={projectId}
        selectedModel={selectedModel}
        onCommit={commitPromptDraftPatch}
        onCommitImmediate={commitPatchImmediately}
      />

      <DraftSection
        draft={draft}
        section={GenerateDimensionFields}
        projectId={projectId}
        selectedModel={selectedModel}
        onCommit={commit}
      />

      <DraftSection
        draft={draft}
        section={GenerateGuidanceSection}
        loraModels={loraModels}
        models={models}
        projectId={projectId}
        selectedModel={selectedModel}
        onCommitImmediate={commitPatchImmediately}
        onConceptCommit={commitDebouncedDraftUpdate}
        onReferenceCommit={commit}
      />

      <DraftSection
        draft={draft}
        section={GenerateRenderSection}
        selectedModel={selectedModel}
        onCommit={commit}
        onCommitImmediate={commitPatchImmediately}
      />

      <DraftSection
        draft={draft}
        section={GenerateComponentsSection}
        selectedModel={selectedModel}
        onCommit={commitPatchImmediately}
      />

      <DraftSection
        draft={draft}
        section={GenerateAdvancedFields}
        selectedModel={selectedModel}
        onCommit={commit}
        onCommitImmediate={commitPatchImmediately}
      />

      <GenerateCanvasSections />
    </Stack>
  );
};

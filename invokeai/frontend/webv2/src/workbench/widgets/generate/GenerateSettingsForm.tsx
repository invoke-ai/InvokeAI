/* eslint-disable react/react-compiler */
import type {
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
  VaeModelConfig,
} from '@workbench/generation/types';
import type { ModelConfig } from '@workbench/models/types';

import { Stack, Text } from '@chakra-ui/react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { GenerateAdvancedFields } from './GenerateAdvancedFields';
import { GenerateCanvasCompositingSection } from './GenerateCanvasCompositingSection';
import { GenerateComponentsSection } from './GenerateComponentsSection';
import { GenerateConceptsSection } from './GenerateConceptsSection';
import {
  applyGenerateSettingsPatch,
  applyGenerateSettingsUpdate,
  getChangedGenerateSettingsPatch,
  type GenerateSettingsUpdate,
  mergeGenerateSettingsUpdate,
  type PendingGenerateSettingsUpdate,
} from './generateDebounce';
import { GenerateDimensionFields } from './GenerateDimensionFields';
import { useRegisterGenerateDraftFlusher } from './generateDraftRegistry';
import { getSettingsWithLatestPromptFields } from './generateFormViewModel';
import { GenerateModelFields } from './GenerateModelFields';
import { GeneratePromptFields } from './promptFields';
import { GenerateReferenceImagesSection } from './reference-images/GenerateReferenceImagesSection';

const GENERATE_INPUT_DEBOUNCE_MS = 250;

interface GenerateSettingsFormProps {
  isLoadingModels: boolean;
  loadError: string | null;
  settings: GenerateSettings;
  loraModels: LoraModelConfig[];
  models: readonly ModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  supportedModels: GenerateModelConfig[];
  vaeModels: VaeModelConfig[];
  onCommitSettings: (nextSettings: GenerateSettings) => void;
  onPatchSettings: (patch: Partial<GenerateSettings>) => void;
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
  vaeModels,
}: GenerateSettingsFormProps) => {
  const [draftSettings, setDraftSettings] = useState(settings);
  const draftSettingsRef = useRef(settings);
  const latestSettingsRef = useRef(settings);
  const pendingUpdateRef = useRef<PendingGenerateSettingsUpdate>(null);
  const projectIdRef = useRef(projectId);
  const timeoutRef = useRef<number | null>(null);
  const onCommitSettingsRef = useRef(onCommitSettings);
  const onPatchSettingsRef = useRef(onPatchSettings);

  onCommitSettingsRef.current = onCommitSettings;
  onPatchSettingsRef.current = onPatchSettings;

  const setDraft = (nextSettings: GenerateSettings) => {
    draftSettingsRef.current = nextSettings;
    setDraftSettings(nextSettings);
  };

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
      setDraft(settings);
      return;
    }

    latestSettingsRef.current = settings;
    setDraft(applyGenerateSettingsUpdate(settings, pendingUpdateRef.current));
  }, [projectId, settings]);

  const flushPendingUpdate = (shouldUpdateDraft = true) => {
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
        setDraft(settingsToCommit);
      }

      onPatchSettingsRef.current(getChangedGenerateSettingsPatch(previousSettings, settingsToCommit));
    }
  };

  useRegisterGenerateDraftFlusher(flushPendingUpdate);

  useEffect(
    () => () => {
      if (timeoutRef.current !== null) {
        window.clearTimeout(timeoutRef.current);
      }

      flushPendingUpdate(false);
    },
    []
  );

  const scheduleCommitUpdate = (update: GenerateSettingsUpdate) => {
    pendingUpdateRef.current = mergeGenerateSettingsUpdate(pendingUpdateRef.current, update);

    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = window.setTimeout(() => {
      timeoutRef.current = null;
      flushPendingUpdate();
    }, GENERATE_INPUT_DEBOUNCE_MS);
  };

  const commit = useCallback((update: GenerateSettingsUpdate) => {
    const nextSettings = applyGenerateSettingsUpdate(
      draftSettingsRef.current,
      mergeGenerateSettingsUpdate(null, update)
    );

    setDraft(nextSettings);
    scheduleCommitUpdate(update);
  }, []);

  const commitDebouncedDraftUpdate = useCallback((update: GenerateSettingsUpdate) => {
    const pendingUpdate = mergeGenerateSettingsUpdate(pendingUpdateRef.current, update);
    const previousSettings = latestSettingsRef.current;
    const nextSettings = applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdate);

    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    pendingUpdateRef.current = null;
    latestSettingsRef.current = nextSettings;
    setDraft(nextSettings);
    onPatchSettingsRef.current(getChangedGenerateSettingsPatch(previousSettings, nextSettings));
  }, []);

  const commitPromptDraftPatch = useCallback((patch: Partial<GenerateSettings>) => {
    const pendingUpdate = mergeGenerateSettingsUpdate(pendingUpdateRef.current, patch);
    const previousSettings = latestSettingsRef.current;
    const nextSettings = applyGenerateSettingsUpdate(latestSettingsRef.current, pendingUpdate);

    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    pendingUpdateRef.current = null;
    latestSettingsRef.current = nextSettings;
    onPatchSettingsRef.current(getChangedGenerateSettingsPatch(previousSettings, nextSettings));
  }, []);

  const commitSettingsImmediately = useCallback(
    (nextSettings: GenerateSettings) => {
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
      setDraft(settingsToCommit);

      if (!Object.is(previousSettings, settingsToCommit)) {
        onCommitSettings(settingsToCommit);
      }
    },
    [onCommitSettings]
  );

  const commitPatchImmediately = useCallback((patch: Partial<GenerateSettings>) => {
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
    setDraft(nextSettings);
    onPatchSettingsRef.current(getChangedGenerateSettingsPatch(previousSettings, nextSettings));
  }, []);

  return (
    <Stack gap="1" px={1}>
      <GeneratePromptFields
        projectId={projectId}
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commitPromptDraftPatch}
        onCommitImmediate={commitPatchImmediately}
      />

      <GenerateReferenceImagesSection
        models={models}
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commit}
        onCommitImmediate={commitPatchImmediately}
      />

      <GenerateDimensionFields
        projectId={projectId}
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commit}
      />

      <GenerateCanvasCompositingSection />

      <GenerateModelFields
        models={models}
        selectedModel={selectedModel}
        settings={draftSettings}
        vaeModels={vaeModels}
        onCommit={commit}
        onCommitSettings={commitSettingsImmediately}
      />

      <GenerateConceptsSection
        loraModels={loraModels}
        projectId={projectId}
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commitDebouncedDraftUpdate}
        onCommitImmediate={commitPatchImmediately}
      />

      <GenerateComponentsSection
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commitPatchImmediately}
      />

      <GenerateAdvancedFields
        selectedModel={selectedModel}
        settings={draftSettings}
        onCommit={commit}
        onCommitImmediate={commitPatchImmediately}
      />

      {isLoadingModels ? (
        <Text color="fg.subtle" fontSize="2xs">
          Loading backend models...
        </Text>
      ) : loadError ? (
        <Text color="fg.error" fontSize="2xs">
          {loadError}
        </Text>
      ) : supportedModels.length === 0 ? (
        <Text color="fg.subtle" fontSize="2xs">
          No supported generation models were found on the backend.
        </Text>
      ) : selectedModel ? null : (
        <Text color="fg.error" fontSize="2xs">
          Select a model to enable invocation.
        </Text>
      )}
    </Stack>
  );
};

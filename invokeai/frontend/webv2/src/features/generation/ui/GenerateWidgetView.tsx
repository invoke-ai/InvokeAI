import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import { Box, HStack, Spinner, Stack, Text } from '@chakra-ui/react';
import {
  getDefaultGenerateSettings,
  isGenerateModelSelectable,
} from '@features/generation/core/baseGenerationPolicies';
import { isLoraModelConfig, normalizeGenerateSettings } from '@features/generation/core/settings';
import {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  useArchitectureCapabilitiesSelector,
} from '@features/generation/data/architectureCapabilitiesStore';
import { resolveGenerateWidgetValues } from '@features/generation/settings';
import { focusIfUnclaimed } from '@platform/react/focusIfUnclaimed';
import { Button } from '@platform/ui/Button';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { getGenerateFormCommitPatch } from './generateFormViewModel';
import { GenerateSettingsForm } from './GenerateSettingsForm';
import { useGenerateValues, useGenerationUi } from './GenerationUiContext';

export const GenerateWidgetView = () => {
  const { t } = useTranslation();
  const ui = useGenerationUi();
  const capabilitiesStatus = useArchitectureCapabilitiesSelector((snapshot) => snapshot.status);
  const capabilitiesError = useArchitectureCapabilitiesSelector((snapshot) => snapshot.error);
  const [hasRequestedRetry, setHasRequestedRetry] = useState(false);
  const projectId = ui.project.activeProjectId;
  const storedValues = useGenerateValues();
  const error = ui.models.error;
  const models = ui.models.catalog;
  const status = ui.models.status;

  const supportedModels = useMemo<GenerateModelConfig[]>(() => models.filter(isGenerateModelSelectable), [models]);
  const loraModels = useMemo(
    () => models.filter((model): model is ModelConfig & LoraModelConfig => isLoraModelConfig(model)),
    [models]
  );
  // Include capability revision/status so a cached null resolves again after loading.
  const resolved = useMemo(
    () => (capabilitiesStatus === 'loaded' ? resolveGenerateWidgetValues({ models, storedValues }) : null),
    [capabilitiesStatus, models, storedValues]
  );
  const settings =
    resolved?.values ?? normalizeGenerateSettings(storedValues) ?? getDefaultGenerateSettings(supportedModels[0]);
  const selectedModel = resolved?.values.model;
  // Keep the failure surface mounted during retry to preserve button focus.
  const isRetrying = hasRequestedRetry && capabilitiesStatus === 'loading';

  // Set by the click, consumed when the form the retry revealed mounts.
  const focusHandoffPending = useRef(false);

  const retryCapabilities = useCallback(() => {
    if (isRetrying) {
      return;
    }

    focusHandoffPending.current = true;
    setHasRequestedRetry(true);
    // Only the initiating retry owns focus; unrelated later reloads must not take it.
    void ensureArchitectureCapabilitiesLoaded().then(() => {
      focusHandoffPending.current &&= getArchitectureCapabilitiesSnapshot().status === 'loaded';
      setHasRequestedRetry(false);
    });
  }, [isRetrying]);

  const handOverFocus = useCallback((element: HTMLDivElement | null) => {
    if (element && focusHandoffPending.current) {
      focusHandoffPending.current = false;
      focusIfUnclaimed(element);
    }
  }, []);

  const commitSettings = useCallback(
    (nextSettings: GenerateSettings) => {
      const model = supportedModels.find((candidate) => candidate.key === nextSettings.modelKey);

      if (!model) {
        return;
      }

      const next = resolveGenerateWidgetValues({
        models,
        storedValues: { ...nextSettings, model },
      });

      if (next) {
        ui.settings.patchGenerateSettings(getGenerateFormCommitPatch(next.values), projectId);
      }
    },
    [models, projectId, supportedModels, ui.settings]
  );

  // Gate the form so edits cannot persist fallback architecture defaults.
  if (capabilitiesStatus !== 'loaded') {
    if (capabilitiesStatus === 'error' || isRetrying) {
      return (
        <Stack aria-busy={isRetrying} aria-live="polite" gap="2" justify="center" minH="8rem" p="1" role="alert">
          <Text color="fg.error" fontSize="xs" textWrap="pretty">
            {t('widgets.generate.capabilitiesLoadFailed')}
          </Text>
          {capabilitiesError ? (
            <Text color="fg.muted" fontSize="xs" textWrap="pretty">
              {capabilitiesError}
            </Text>
          ) : null}
          {/* `aria-disabled` rather than `disabled`: a disabled button drops the focus it holds. */}
          <Button
            alignSelf="flex-start"
            aria-busy={isRetrying}
            aria-disabled={isRetrying}
            variant="outline"
            onClick={retryCapabilities}
          >
            {isRetrying ? <Spinner /> : null}
            {t('widgets.generate.retry')}
          </Button>
        </Stack>
      );
    }

    return (
      <HStack
        align="center"
        aria-atomic="true"
        aria-busy="true"
        aria-live="polite"
        color="fg.muted"
        gap="1.5"
        justify="center"
        minH="8rem"
        p="1"
        role="status"
      >
        <Spinner />
        <Text fontSize="xs">{t('widgets.generate.loadingCapabilities')}</Text>
      </HStack>
    );
  }

  return (
    // A successful retry unmounts the button that held focus; hand it to the form the retry revealed.
    <Box ref={handOverFocus}>
      <GenerateSettingsForm
        isLoadingModels={status === 'idle' || status === 'loading'}
        loadError={error}
        loraModels={loraModels}
        models={models}
        projectId={projectId}
        selectedModel={selectedModel}
        settings={settings}
        supportedModels={supportedModels}
        onCommitSettings={commitSettings}
        onPatchSettings={ui.settings.patchGenerateSettings}
      />
    </Box>
  );
};

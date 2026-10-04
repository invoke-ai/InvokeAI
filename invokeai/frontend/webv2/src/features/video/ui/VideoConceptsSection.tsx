import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';

import { Stack, Text } from '@chakra-ui/react';
import {
  ConceptList,
  type ConceptModelPort,
  ConceptRow,
  GenerationSettingsSection,
} from '@features/generation/components';
import { getDefaultLoraWeight, isLoraCompatibleWithModel, isLoraModelConfig } from '@features/generation/settings';
import { getModelBaseColorPalette, getModelBaseLabel, getModelImageUrl, useOpenModelInManager } from '@features/models';
import { ModelSelect } from '@features/models/react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoLorasEquivalent, areVideoModelsEquivalent } from './videoComparators';

/** Graph compilation routes LoRAs to experts from their probed tags; the UI supplies no target override. */

const LORA_MODEL_TYPES: readonly ModelTaxonomyType[] = ['lora'];

export const VideoConceptsSection = memo(
  function VideoConceptsSection({
    loras,
    model,
    onChangeLoras,
    projectId,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    onChangeLoras: (update: (current: GenerateLora[]) => GenerateLora[]) => void;
    projectId: string;
  }) {
    const { t } = useTranslation();
    const openInModelManager = useOpenModelInManager();
    const conceptModels = useMemo<ConceptModelPort>(
      () => ({
        getBaseColorPalette: getModelBaseColorPalette,
        getBaseLabel: getModelBaseLabel,
        getImageUrl: getModelImageUrl,
        openInModelManager: openInModelManager ?? undefined,
      }),
      [openInModelManager]
    );
    const selectedLoraKeys = useMemo(() => new Set(loras.map((lora) => lora.model.key)), [loras]);
    const loraFilter = useCallback(
      (candidate: ModelConfig) =>
        Boolean(model && isLoraModelConfig(candidate) && isLoraCompatibleWithModel(candidate, model)),
      [model]
    );
    const addLora = useCallback(
      (candidate: ModelConfig | null) => {
        if (!model || !isLoraModelConfig(candidate) || !isLoraCompatibleWithModel(candidate, model)) {
          return;
        }

        onChangeLoras((current) => [
          ...current,
          { isEnabled: true, model: candidate, weight: getDefaultLoraWeight(candidate) },
        ]);
      },
      [model, onChangeLoras]
    );
    const updateLora = useCallback(
      (key: string, update: Partial<GenerateLora>) =>
        // A row removed mid-edit still flushes its draft; returning the list unchanged makes that a no-op.
        onChangeLoras((current) =>
          current.some((lora) => lora.model.key === key)
            ? current.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora))
            : current
        ),
      [onChangeLoras]
    );
    const removeLora = useCallback(
      (key: string) => onChangeLoras((current) => current.filter((candidate) => candidate.model.key !== key)),
      [onChangeLoras]
    );

    return (
      <GenerationSettingsSection label={t('widgets.video.concepts')} sectionId="video-concepts" defaultOpen>
        <Stack gap="2" p="2">
          <ModelSelect
            disabled={!model}
            excludeKeys={selectedLoraKeys}
            filter={loraFilter}
            modelTypes={LORA_MODEL_TYPES}
            placeholder={t('widgets.video.addLora')}
            scopeLabel={t('models.scopeConcepts')}
            value={null}
            onChange={addLora}
          />
          {loras.length === 0 ? (
            <Text color="fg.muted" fontSize="xs">
              {t('widgets.video.noLoras')}
            </Text>
          ) : (
            <ConceptList label={t('widgets.video.concepts')} projectId={projectId}>
              {loras.map((lora) => (
                <ConceptRow
                  key={lora.model.key}
                  models={conceptModels}
                  lora={lora}
                  onRemove={removeLora}
                  onUpdate={updateLora}
                />
              ))}
            </ConceptList>
          )}
        </Stack>
      </GenerationSettingsSection>
    );
  },
  (previous, next) =>
    previous.projectId === next.projectId &&
    previous.onChangeLoras === next.onChangeLoras &&
    areVideoModelsEquivalent(previous.model, next.model) &&
    areVideoLorasEquivalent(previous.loras, next.loras)
);

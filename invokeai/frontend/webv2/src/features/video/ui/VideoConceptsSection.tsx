import type { GenerateLora, MainModelConfig } from '@features/generation/contracts';
import type { ModelConfig, ModelTaxonomyType } from '@features/models';

import { Stack, Text } from '@chakra-ui/react';
import {
  ConceptList,
  type ConceptModelIdentity,
  ConceptRow,
  GenerationSettingsSection,
} from '@features/generation/components';
import { getDefaultLoraWeight, isLoraCompatibleWithModel, isLoraModelConfig } from '@features/generation/settings';
import { getModelBaseColorPalette, getModelBaseLabel, getModelImageUrl } from '@features/models';
import { ModelSelect } from '@features/models/react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { areVideoLorasEquivalent, areVideoModelsEquivalent } from './videoComparators';

/** Graph compilation routes LoRAs to experts from their probed tags; the UI supplies no target override. */

const LORA_MODEL_TYPES: readonly ModelTaxonomyType[] = ['lora'];
const CONCEPT_IDENTITY: ConceptModelIdentity = {
  getBaseColorPalette: getModelBaseColorPalette,
  getBaseLabel: getModelBaseLabel,
  getImageUrl: getModelImageUrl,
};

export const VideoConceptsSection = memo(
  function VideoConceptsSection({
    loras,
    model,
    onChangeLoras,
  }: {
    loras: GenerateLora[];
    model: MainModelConfig | null;
    onChangeLoras: (loras: GenerateLora[]) => void;
  }) {
    const { t } = useTranslation();
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

        onChangeLoras([...loras, { isEnabled: true, model: candidate, weight: getDefaultLoraWeight(candidate) }]);
      },
      [loras, model, onChangeLoras]
    );
    const updateLora = useCallback(
      (key: string, update: Partial<GenerateLora>) =>
        onChangeLoras(loras.map((lora) => (lora.model.key === key ? { ...lora, ...update } : lora))),
      [loras, onChangeLoras]
    );
    const removeLora = useCallback(
      (key: string) => onChangeLoras(loras.filter((candidate) => candidate.model.key !== key)),
      [loras, onChangeLoras]
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
            size="xs"
            value={null}
            onChange={addLora}
          />
          {loras.length === 0 ? (
            <Text color="fg.muted" fontSize="2xs">
              {t('widgets.video.noLoras')}
            </Text>
          ) : (
            <ConceptList>
              {loras.map((lora) => (
                <ConceptRow
                  key={lora.model.key}
                  identity={CONCEPT_IDENTITY}
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
    previous.onChangeLoras === next.onChangeLoras &&
    areVideoModelsEquivalent(previous.model, next.model) &&
    areVideoLorasEquivalent(previous.loras, next.loras)
);

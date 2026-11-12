import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';

import { Badge, Stack, Text } from '@chakra-ui/react';
import { getKrea2EditLoraState, isReferenceImageSupported } from '@features/generation/core/baseGenerationPolicies';
import {
  type GenerateDraft,
  type GenerateSettingsUpdate,
  pickGenerateSettings,
} from '@features/generation/ui/generateDebounce';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerateConceptsContent, isCompatibleLora, isConceptInstalled } from './GenerateConceptsSection';
import { useGenerationUi } from './GenerationUiContext';
import { GenerateReferenceImagesContent } from './reference-images/GenerateReferenceImagesSection';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateGuidanceSectionProps {
  draft: GenerateDraft;
  loraModels: LoraModelConfig[];
  models: readonly ModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  /** Debounced draft-update channel — reference-image edits ride the form's debounce. */
  onReferenceCommit: (update: GenerateSettingsUpdate) => void;
  /** Flushed update channel — concept rows debounce their own weight drafts first. */
  onConceptCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

const selectGuidanceSettings = pickGenerateSettings([
  'loras',
  'qwen3VLEncoderModel',
  'qwen3VLVisionTowerModel',
  'referenceImages',
]);

export const GenerateGuidanceSection = ({
  draft,
  loraModels,
  models,
  onCommitImmediate,
  onConceptCommit,
  onReferenceCommit,
  projectId,
  selectedModel,
}: GenerateGuidanceSectionProps) => {
  const { t } = useTranslation();
  const settings = useExternalStoreSelector(draft.subscribe, draft.getSnapshot, selectGuidanceSettings);
  const referenceImagesSupported = isReferenceImageSupported(selectedModel, settings);
  const referenceImageCount = settings.referenceImages.length;
  const activeReferenceImages = referenceImagesSupported
    ? settings.referenceImages.filter((image) => image.isEnabled).length
    : 0;
  const catalogStatus = useGenerationUi().models.status;
  const activeConcepts = settings.loras.filter(
    (lora) =>
      lora.isEnabled && isCompatibleLora(lora, selectedModel) && isConceptInstalled(lora, loraModels, catalogStatus)
  ).length;
  const hasIncompatible =
    (!referenceImagesSupported && referenceImageCount > 0) ||
    Boolean(selectedModel && settings.loras.some((lora) => !isCompatibleLora(lora, selectedModel)));
  const activeCount = activeReferenceImages + activeConcepts;
  const totalCount = referenceImageCount + settings.loras.length;

  // Whether something here cannot run, or is all switched off, holds open or folded; what is active is a summary.
  const status = (
    <>
      {hasIncompatible ? (
        <Badge colorPalette="orange" variant="surface">
          {t('widgets.generate.incompatible')}
        </Badge>
      ) : null}
      {activeCount === 0 && totalCount > 0 ? (
        <Badge variant="surface">{t('widgets.generate.offCount', { count: totalCount })}</Badge>
      ) : null}
    </>
  );
  const summary = useMemo(
    () =>
      activeCount === 0
        ? undefined
        : [
            activeCount === totalCount
              ? t('widgets.generate.activeCount', { count: activeCount })
              : t('widgets.generate.activeOfTotal', { active: activeCount, total: totalCount }),
          ],
    [activeCount, t, totalCount]
  );

  return (
    <GenerateCollapsibleSection
      label={t('widgets.generate.guidance')}
      defaultOpen
      sectionId="guidance"
      status={status}
      summary={summary}
    >
      <Stack gap="2.5">
        <GenerateReferenceImagesContent
          models={models}
          visionComponents={settings}
          referenceImages={settings.referenceImages}
          selectedModel={selectedModel}
          onCommit={onReferenceCommit}
          onCommitImmediate={onCommitImmediate}
        />
        {/* Not blocking: the run works, but without an edit LoRA it only copies the references. */}
        {getKrea2EditLoraState(selectedModel, settings) === 'noLora' ? (
          <Text color="fg.warning" fontSize="xs" role="status" textWrap="pretty">
            {t('widgets.generate.krea2EditReferencesNeedLora')}
          </Text>
        ) : null}
        <GenerateConceptsContent
          loraModels={loraModels}
          loras={settings.loras}
          projectId={projectId}
          selectedModel={selectedModel}
          onCommit={onConceptCommit}
        />
      </Stack>
    </GenerateCollapsibleSection>
  );
};

import type { GenerationModelCatalogItem as ModelConfig } from '@features/generation/contracts';
import type { GenerateModelConfig, GenerateSettings, LoraModelConfig } from '@features/generation/core/types';
import type { GenerateSettingsUpdate } from '@features/generation/ui/generateDebounce';

import { Badge, Separator, Stack } from '@chakra-ui/react';
import { isReferenceImageSupported } from '@features/generation/core/baseGenerationPolicies';
import { isLoraCompatibleWithModel } from '@features/generation/core/settings';
import { useTranslation } from 'react-i18next';

import { GenerateConceptsContent } from './GenerateConceptsSection';
import { GenerateReferenceImagesContent } from './reference-images/GenerateReferenceImagesSection';
import { GenerateCollapsibleSection } from './shared/GenerateCollapsibleSection';

interface GenerateGuidanceSectionProps {
  loraModels: LoraModelConfig[];
  models: readonly ModelConfig[];
  projectId: string;
  selectedModel: GenerateModelConfig | undefined;
  settings: GenerateSettings;
  /** Debounced draft-update channel — reference-image edits ride the form's debounce. */
  onReferenceCommit: (update: GenerateSettingsUpdate) => void;
  /** Flushed update channel — concept rows debounce their own weight drafts first. */
  onConceptCommit: (update: GenerateSettingsUpdate) => void;
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

/**
 * Creative conditioning in one place: reference images and concepts (LoRAs) are
 * the same kind of decision — a visual influence with a weight and a toggle —
 * so they share a section instead of two sibling accordions.
 */
export const GenerateGuidanceSection = ({
  loraModels,
  models,
  onCommitImmediate,
  onConceptCommit,
  onReferenceCommit,
  projectId,
  selectedModel,
  settings,
}: GenerateGuidanceSectionProps) => {
  const { t } = useTranslation();
  const referenceImagesSupported = isReferenceImageSupported(selectedModel);
  const referenceImageCount = settings.referenceImages.length;
  const activeReferenceImages = referenceImagesSupported
    ? settings.referenceImages.filter((image) => image.isEnabled).length
    : 0;
  const activeConcepts = selectedModel
    ? settings.loras.filter((lora) => lora.isEnabled && isLoraCompatibleWithModel(lora.model, selectedModel)).length
    : 0;
  const hasIncompatible =
    (!referenceImagesSupported && referenceImageCount > 0) ||
    settings.loras.some((lora) => selectedModel && !isLoraCompatibleWithModel(lora.model, selectedModel));
  const activeCount = activeReferenceImages + activeConcepts;
  const totalCount = referenceImageCount + settings.loras.length;

  const badges = (
    <>
      {hasIncompatible ? (
        <Badge colorPalette="orange" size="xs" variant="surface">
          {t('widgets.generate.incompatible')}
        </Badge>
      ) : null}
      {activeCount > 0 ? (
        <Badge size="xs" variant="surface">
          {t('widgets.generate.activeCount', { count: activeCount })}
        </Badge>
      ) : totalCount > 0 ? (
        <Badge size="xs" variant="surface">
          {t('widgets.generate.offCount', { count: totalCount })}
        </Badge>
      ) : null}
    </>
  );

  return (
    <GenerateCollapsibleSection label={t('widgets.generate.guidance')} defaultOpen badges={badges} sectionId="guidance">
      <Stack gap="2" p="2">
        <GenerateReferenceImagesContent
          models={models}
          selectedModel={selectedModel}
          settings={settings}
          onCommit={onReferenceCommit}
          onCommitImmediate={onCommitImmediate}
        />
        {referenceImagesSupported || referenceImageCount > 0 ? <Separator borderColor="bg.subtle" /> : null}
        <GenerateConceptsContent
          loraModels={loraModels}
          projectId={projectId}
          selectedModel={selectedModel}
          settings={settings}
          onCommit={onConceptCommit}
          onCommitImmediate={onCommitImmediate}
        />
      </Stack>
    </GenerateCollapsibleSection>
  );
};

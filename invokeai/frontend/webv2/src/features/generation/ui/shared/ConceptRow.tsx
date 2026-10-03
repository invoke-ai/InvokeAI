import type { GenerateLora } from '@features/generation/core/types';
import type { ReactNode } from 'react';

import { Avatar, Badge, HStack, Icon, Stack, StackSeparator, Text } from '@chakra-ui/react';
import { DEFAULT_LORA_WEIGHT_CONFIG, getDefaultLoraWeight } from '@features/generation/core/settings';
import { IconButton } from '@platform/ui/Button';
import { MiddleTruncate } from '@platform/ui/MiddleTruncate';
import { ScrubberField } from '@platform/ui/ScrubberField';
import { Tooltip } from '@platform/ui/Tooltip';
import { BoxIcon, Trash2Icon } from 'lucide-react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { GenerateFieldContextMenu } from './GenerateFieldContextMenu';
import { GenerateToggleSwitch } from './GenerateToggleSwitch';

const WEIGHT_MARKS = [-1, 0, 1, 2];

/** Resolves a model's badge and thumbnail; the caller supplies the model catalog's identity helpers. */
export interface ConceptModelIdentity {
  getBaseColorPalette(base: string): string;
  getBaseLabel(base: string): string;
  getImageUrl(key: string): string;
}

export type ConceptUpdate = Partial<Pick<GenerateLora, 'isEnabled' | 'weight'>>;

export interface ConceptRowProps {
  identity: ConceptModelIdentity;
  /** False dims the row, badges it, and locks it off; the caller decides compatibility against its main model. */
  isCompatible?: boolean;
  /** `lora.weight` is the displayed weight, so a caller that defers commits passes its draft here. */
  lora: GenerateLora;
  onRemove: (key: string) => void;
  /** Fires per weight step and on toggle; the caller owns any debouncing. */
  onUpdate: (key: string, update: ConceptUpdate) => void;
}

const LIST_SEPARATOR = <StackSeparator borderColor="border.subtle" />;

/** Stacks concept rows with dividers between them. */
export const ConceptList = ({ children }: { children: ReactNode }) => (
  <Stack gap="0" separator={LIST_SEPARATOR}>
    {children}
  </Stack>
);

/** One applied LoRA/concept: thumbnail and identity, enable toggle, remove, and a weight scrubber with copy/reset. */
export const ConceptRow = memo(function ConceptRow({
  identity,
  isCompatible = true,
  lora,
  onRemove,
  onUpdate,
}: ConceptRowProps) {
  const { t } = useTranslation();
  const { key, name, trigger_phrases: triggerPhrases } = lora.model;
  const isActive = lora.isEnabled && isCompatible;
  const defaultWeight = getDefaultLoraWeight(lora.model);
  const handleToggle = useCallback((isEnabled: boolean) => onUpdate(key, { isEnabled }), [key, onUpdate]);
  const handleWeightChange = useCallback((weight: number) => onUpdate(key, { weight }), [key, onUpdate]);
  const handleRemove = useCallback(() => onRemove(key), [key, onRemove]);
  const handleReset = useCallback(() => onUpdate(key, { weight: defaultWeight }), [defaultWeight, key, onUpdate]);
  const copyWeight = useCallback(() => String(lora.weight), [lora.weight]);

  return (
    // Naming the group gives each row's "Weight" slider and controls their concept's context.
    <Stack aria-label={name} gap="2" opacity={isCompatible ? 1 : 0.68} py="2.5" role="group">
      <HStack gap="2.5" minW="0">
        <Avatar.Root
          bg="bg.muted"
          borderColor="border"
          borderWidth="1px"
          color="fg.subtle"
          flexShrink={0}
          shape="rounded"
          size="sm"
        >
          {/* The fallback also covers a cover image that fails to load. */}
          <Avatar.Fallback>
            <Icon as={BoxIcon} boxSize="4" />
          </Avatar.Fallback>
          {lora.model.cover_image ? <Avatar.Image alt="" src={identity.getImageUrl(key)} /> : null}
        </Avatar.Root>
        <Stack flex="1" gap="0.5" minW="0">
          <HStack gap="1.5" minW="0">
            <MiddleTruncate
              color={isActive ? 'fg' : 'fg.muted'}
              fontSize="xs"
              fontWeight="medium"
              minW="0"
              text={name}
            />
            <Badge
              colorPalette={identity.getBaseColorPalette(lora.model.base)}
              flexShrink={0}
              size="xs"
              variant="surface"
            >
              {identity.getBaseLabel(lora.model.base)}
            </Badge>
            {isCompatible ? null : (
              <Badge colorPalette="orange" flexShrink={0} size="xs" variant="surface">
                {t('widgets.generate.incompatible')}
              </Badge>
            )}
          </HStack>
          {triggerPhrases?.length ? (
            <Text color="fg.muted" fontSize="2xs" truncate>
              {triggerPhrases.join(', ')}
            </Text>
          ) : null}
        </Stack>
        <HStack flexShrink="0" gap="1">
          <GenerateToggleSwitch
            checked={isActive}
            disabled={!isCompatible}
            label={
              isActive ? t('widgets.generate.disableConcept', { name }) : t('widgets.generate.enableConcept', { name })
            }
            onCheckedChange={handleToggle}
          />
          <Tooltip content={t('widgets.generate.removeConcept')}>
            <IconButton
              aria-label={t('widgets.generate.removeConceptNamed', { name })}
              color="fg.muted"
              size="2xs"
              variant="ghost"
              onClick={handleRemove}
            >
              <Trash2Icon />
            </IconButton>
          </Tooltip>
        </HStack>
      </HStack>
      <GenerateFieldContextMenu
        copyValue={copyWeight}
        isAtDefault={lora.weight === defaultWeight}
        resetLabel={t('widgets.generate.resetToConceptDefault')}
        onReset={handleReset}
      >
        <ScrubberField
          defaultValue={defaultWeight}
          disabled={!isActive}
          inputMax={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMax}
          inputMin={DEFAULT_LORA_WEIGHT_CONFIG.numberInputMin}
          label={t('widgets.generate.weight')}
          marks={WEIGHT_MARKS}
          max={DEFAULT_LORA_WEIGHT_CONFIG.sliderMax}
          min={DEFAULT_LORA_WEIGHT_CONFIG.sliderMin}
          step={DEFAULT_LORA_WEIGHT_CONFIG.coarseStep}
          value={lora.weight}
          onChange={handleWeightChange}
        />
      </GenerateFieldContextMenu>
    </Stack>
  );
});

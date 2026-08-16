/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-jsx-as-prop */
import type { RebalancePreset } from '@features/generation/core/conditioningRebalance';
import type { GenerateSettings } from '@features/generation/core/types';

import { createListCollection, HStack, Icon, Input, Menu, Portal, Stack, Switch, Text } from '@chakra-ui/react';
import {
  BUILTIN_REBALANCE_PRESETS,
  DEFAULT_KREA2_REBALANCE_MULTIPLIER,
  DEFAULT_KREA2_REBALANCE_WEIGHTS,
  KREA2_REBALANCE_WEIGHT_COUNT,
  matchRebalancePreset,
  parseRebalanceWeights,
  REBALANCE_PRESET_DEFAULT_ID,
  serializeRebalanceWeights,
} from '@features/generation/core/conditioningRebalance';
// Imported by subpath rather than from the `@platform/ui` barrel: the barrel is at its
// direct-importer budget, so each call site here would otherwise cost against it.
import { IconButton } from '@platform/ui/Button';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Field } from '@platform/ui/Field';
import { MenuContent } from '@platform/ui/Menu';
import { RenameDialog } from '@platform/ui/RenameDialog';
import { Select } from '@platform/ui/Select';
import { SliderNumberField } from '@platform/ui/SliderNumberField';
import { Tooltip } from '@platform/ui/Tooltip';
import { MoreHorizontalIcon, RotateCcwIcon } from 'lucide-react';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useGenerationUi } from './GenerationUiContext';
import { ConditioningRebalanceBars } from './shared/ConditioningRebalanceBars';
import { ConditioningRebalanceSparkline } from './shared/ConditioningRebalanceSparkline';

/** The gain slider's practical range. Typed values may go to the legacy schema's bound. */
const GAIN_SLIDER_MAX = 10;
const GAIN_INPUT_MAX = 20;

const SELECT_CONTENT_PROPS = { maxH: '16rem' } as const;
const READOUT_PROPS = {
  color: 'fg.subtle',
  fontFamily: 'mono',
  fontSize: '2xs',
  // Reserved so the header does not reflow as the readout appears and clears.
  minH: '4',
  whiteSpace: 'nowrap',
} as const;

const DEFAULT_WEIGHT_VECTOR: readonly number[] = parseRebalanceWeights(DEFAULT_KREA2_REBALANCE_WEIGHTS) ?? [];

type PresetDialog = { mode: 'save' } | { mode: 'rename'; preset: RebalancePreset };

interface GenerateConditioningRebalanceFieldProps {
  settings: GenerateSettings;
  /** Debounced; used for the continuous controls. */
  onCommit: (patch: Partial<GenerateSettings>) => void;
  /** Immediate; used for discrete edits that should not sit in the draft. */
  onCommitImmediate: (patch: Partial<GenerateSettings>) => void;
}

/**
 * Krea-2's per-tap conditioning gains.
 *
 * The twelve bars weight the twelve Qwen3-VL encoder layers stacked into the
 * conditioning, shallow (the prompt's literal wording) to deep (its meaning and
 * composition); Gain scales the result. Collapsed, the row is a switch and a preview of
 * the current curve.
 */
export const GenerateConditioningRebalanceField = ({
  onCommit,
  onCommitImmediate,
  settings,
}: GenerateConditioningRebalanceFieldProps) => {
  const { t } = useTranslation();
  const { rebalancePresets } = useGenerationUi();
  // The switch shares the Field.Root with the controls below; an explicit id keeps its
  // label bound to its own hidden input rather than the Field's control id.
  const switchInputId = useId();

  const [previewWeights, setPreviewWeights] = useState<number[] | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  /** Non-null while the text field is being typed into, so a half-typed vector survives. */
  const [weightsDraft, setWeightsDraft] = useState<string | null>(null);
  const [presetDialog, setPresetDialog] = useState<PresetDialog | null>(null);
  const [presetPendingDelete, setPresetPendingDelete] = useState<RebalancePreset | null>(null);

  const committedWeights = useMemo(
    () => parseRebalanceWeights(settings.krea2RebalanceWeights) ?? DEFAULT_WEIGHT_VECTOR,
    [settings.krea2RebalanceWeights]
  );
  const weights = previewWeights ?? committedWeights;
  const isWeightsValid = parseRebalanceWeights(weightsDraft ?? settings.krea2RebalanceWeights) !== null;

  const presets = useMemo<RebalancePreset[]>(
    () => [...BUILTIN_REBALANCE_PRESETS, ...rebalancePresets.presets],
    [rebalancePresets.presets]
  );
  const presetCollection = useMemo(
    () =>
      createListCollection({
        itemToString: (preset: RebalancePreset) => preset.label,
        itemToValue: (preset: RebalancePreset) => preset.id,
        items: presets,
      }),
    [presets]
  );
  const activePresetId = matchRebalancePreset(
    presets,
    settings.krea2RebalanceWeights,
    settings.krea2RebalanceMultiplier
  );
  const activePreset = presets.find((preset) => preset.id === activePresetId) ?? null;
  const isCustomPreset =
    activePreset !== null && !BUILTIN_REBALANCE_PRESETS.some((preset) => preset.id === activePreset.id);
  const isAtDefault = activePresetId === REBALANCE_PRESET_DEFAULT_ID;

  const applyWeights = useCallback(
    (nextWeights: string, multiplier?: number) => {
      setWeightsDraft(null);
      setPreviewWeights(null);
      onCommitImmediate(
        multiplier === undefined
          ? { krea2RebalanceWeights: nextWeights }
          : { krea2RebalanceMultiplier: multiplier, krea2RebalanceWeights: nextWeights }
      );
    },
    [onCommitImmediate]
  );

  const handleBarsCommit = useCallback(
    (nextWeights: number[]) => {
      setWeightsDraft(null);
      setPreviewWeights(null);
      onCommit({ krea2RebalanceWeights: serializeRebalanceWeights(nextWeights) });
    },
    [onCommit]
  );

  const handleWeightsTextChange = useCallback(
    (value: string) => {
      setWeightsDraft(value);

      // Only a vector the backend would accept reaches the settings; the draft holds
      // everything in between so the field does not fight the cursor.
      if (parseRebalanceWeights(value) !== null) {
        onCommitImmediate({ krea2RebalanceWeights: value });
      }
    },
    [onCommitImmediate]
  );

  const handleReset = useCallback(() => {
    applyWeights(DEFAULT_KREA2_REBALANCE_WEIGHTS, DEFAULT_KREA2_REBALANCE_MULTIPLIER);
  }, [applyWeights]);

  const handlePresetChange = useCallback(
    (presetIds: string[]) => {
      const preset = presets.find((entry) => entry.id === presetIds[0]);

      if (preset) {
        applyWeights(preset.weights, preset.multiplier);
      }
    },
    [applyWeights, presets]
  );

  const handlePresetDialogSubmit = useCallback(
    (label: string) => {
      if (presetDialog?.mode === 'rename') {
        rebalancePresets.rename(presetDialog.preset.id, label);

        return;
      }

      rebalancePresets.save(label, settings.krea2RebalanceWeights, settings.krea2RebalanceMultiplier);
    },
    [presetDialog, rebalancePresets, settings.krea2RebalanceMultiplier, settings.krea2RebalanceWeights]
  );

  const handleConfirmDelete = useCallback(() => {
    if (presetPendingDelete) {
      rebalancePresets.remove(presetPendingDelete.id);
    }
  }, [presetPendingDelete, rebalancePresets]);

  const tapLabel = useCallback(
    (tap: number, layer: number) => t('widgets.generate.krea2RebalanceTapLabel', { layer, tap }),
    [t]
  );

  const activeWeight = activeIndex === null ? undefined : weights[activeIndex];
  const isEnabled = settings.krea2RebalanceEnabled;

  return (
    <>
      <Field
        borderColor="border.subtle"
        borderTopWidth="1px"
        helpText={isEnabled ? undefined : t('widgets.generate.krea2RebalanceHelp')}
        hint="conditioningRebalance"
        label={t('widgets.generate.krea2Rebalance')}
        labelEnd={
          <HStack gap="2">
            <ConditioningRebalanceSparkline muted={!isEnabled} weights={committedWeights} />
            <Switch.Root
              checked={isEnabled}
              ids={{ hiddenInput: switchInputId, label: `${switchInputId}-label` }}
              size="sm"
              onCheckedChange={({ checked }) => onCommitImmediate({ krea2RebalanceEnabled: checked })}
            >
              <Switch.HiddenInput aria-label={t('widgets.generate.krea2Rebalance')} />
              <Switch.Control _checked={{ bg: 'accent.solid' }}>
                <Switch.Thumb />
              </Switch.Control>
            </Switch.Root>
          </HStack>
        }
        pt="2"
      >
        {isEnabled ? (
          <Stack gap="2" w="full">
            <HStack gap="1">
              <Select
                aria-label={t('widgets.generate.krea2RebalancePreset')}
                collection={presetCollection}
                contentProps={SELECT_CONTENT_PROPS}
                flex="1"
                minW="0"
                size="xs"
                value={activePresetId === null ? [] : [activePresetId]}
                valueText={activePreset?.label ?? t('widgets.generate.krea2RebalancePresetCustom')}
                onValueChange={({ value }) => handlePresetChange(value)}
              />
              <Tooltip content={t('widgets.generate.krea2RebalanceReset')}>
                <IconButton
                  aria-label={t('widgets.generate.krea2RebalanceReset')}
                  color="fg.muted"
                  disabled={isAtDefault}
                  size="2xs"
                  variant="ghost"
                  onClick={handleReset}
                >
                  <Icon as={RotateCcwIcon} boxSize="3" />
                </IconButton>
              </Tooltip>
              <Menu.Root lazyMount unmountOnExit>
                <Menu.Trigger asChild>
                  <IconButton
                    aria-label={t('widgets.generate.krea2RebalancePresetActions')}
                    color="fg.muted"
                    size="2xs"
                    variant="ghost"
                  >
                    <Icon as={MoreHorizontalIcon} boxSize="3" />
                  </IconButton>
                </Menu.Trigger>
                <Portal>
                  <Menu.Positioner>
                    <MenuContent minW="11rem">
                      <Menu.Item
                        disabled={!isWeightsValid}
                        value="save"
                        onClick={() => setPresetDialog({ mode: 'save' })}
                      >
                        <Menu.ItemText>{t('widgets.generate.krea2RebalancePresetSave')}</Menu.ItemText>
                      </Menu.Item>
                      <Menu.Item
                        disabled={!isCustomPreset}
                        value="rename"
                        onClick={() => activePreset && setPresetDialog({ mode: 'rename', preset: activePreset })}
                      >
                        <Menu.ItemText>{t('widgets.generate.krea2RebalancePresetRename')}</Menu.ItemText>
                      </Menu.Item>
                      <Menu.Separator />
                      <Menu.Item
                        color="fg.error"
                        disabled={!isCustomPreset}
                        value="delete"
                        onClick={() => activePreset && setPresetPendingDelete(activePreset)}
                      >
                        <Menu.ItemText>{t('widgets.generate.krea2RebalancePresetDelete')}</Menu.ItemText>
                      </Menu.Item>
                    </MenuContent>
                  </Menu.Positioner>
                </Portal>
              </Menu.Root>
            </HStack>

            <Stack gap="1">
              <HStack justify="flex-end" {...READOUT_PROPS}>
                <Text as="span">
                  {activeIndex !== null && activeWeight !== undefined
                    ? t('widgets.generate.krea2RebalanceTapReadout', {
                        tap: activeIndex + 1,
                        value: activeWeight.toFixed(2),
                      })
                    : ''}
                </Text>
              </HStack>
              <ConditioningRebalanceBars
                activeIndex={activeIndex}
                tapLabel={tapLabel}
                weights={weights}
                onActiveIndexChange={setActiveIndex}
                onCommit={handleBarsCommit}
                onPreview={setPreviewWeights}
              />
              <HStack color="fg.subtle" fontSize="2xs" justify="space-between">
                <Text as="span">{t('widgets.generate.krea2RebalanceAxisShallow')}</Text>
                <Text as="span">{t('widgets.generate.krea2RebalanceAxisDeep')}</Text>
              </HStack>
            </Stack>

            <Field label={t('widgets.generate.krea2RebalanceGain')}>
              <SliderNumberField
                ariaLabel={t('widgets.generate.krea2RebalanceGain')}
                defaultValue={DEFAULT_KREA2_REBALANCE_MULTIPLIER}
                max={GAIN_SLIDER_MAX}
                min={0}
                numberInputMax={GAIN_INPUT_MAX}
                resetLabel={t('widgets.generate.krea2RebalanceGainReset')}
                step={0.1}
                value={settings.krea2RebalanceMultiplier}
                onChange={(multiplier) => onCommit({ krea2RebalanceMultiplier: multiplier })}
              />
            </Field>

            <Field
              error={
                isWeightsValid
                  ? undefined
                  : t('widgets.generate.krea2RebalanceWeightsInvalid', { count: KREA2_REBALANCE_WEIGHT_COUNT })
              }
              label={t('widgets.generate.krea2RebalanceWeights')}
            >
              <Input
                aria-label={t('widgets.generate.krea2RebalanceWeights')}
                fontFamily="mono"
                size="xs"
                value={
                  weightsDraft ??
                  (previewWeights ? serializeRebalanceWeights(previewWeights) : settings.krea2RebalanceWeights)
                }
                onBlur={() => setWeightsDraft(null)}
                onChange={(event) => handleWeightsTextChange(event.target.value)}
              />
            </Field>
          </Stack>
        ) : null}
      </Field>

      <RenameDialog
        initialName={presetDialog?.mode === 'rename' ? presetDialog.preset.label : ''}
        isOpen={presetDialog !== null}
        label={t('widgets.generate.krea2RebalancePresetName')}
        submitLabel={
          presetDialog?.mode === 'rename'
            ? t('widgets.generate.krea2RebalancePresetRename')
            : t('widgets.generate.krea2RebalancePresetSaveAction')
        }
        title={
          presetDialog?.mode === 'rename'
            ? t('widgets.generate.krea2RebalancePresetRename')
            : t('widgets.generate.krea2RebalancePresetSave')
        }
        onClose={() => setPresetDialog(null)}
        onSubmit={handlePresetDialogSubmit}
      />
      <ConfirmDialog
        body={t('widgets.generate.krea2RebalancePresetDeleteBody', { name: presetPendingDelete?.label ?? '' })}
        confirmLabel={t('widgets.generate.krea2RebalancePresetDelete')}
        isOpen={presetPendingDelete !== null}
        title={t('widgets.generate.krea2RebalancePresetDelete')}
        onClose={() => setPresetPendingDelete(null)}
        onConfirm={handleConfirmDelete}
      />
    </>
  );
};

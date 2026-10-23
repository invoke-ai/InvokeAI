import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import { Box, Text } from '@chakra-ui/react';
import { getControlAdapterMinWeight } from '@features/generation/graph';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { Slider } from '@platform/ui/Slider';
import {
  FormNumberField,
  useNumberCommit,
  useSliderGesture,
} from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useId, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatPanelText, type ControlPanelModel, type ControlPriority } from './controlPanelModel';
import { CONTROL_WEIGHT_BOUNDS } from './layerOps';

const STEP_RANGE_STEP = 0.05;
const PRIORITIES: readonly ControlPriority[] = ['more_prompt', 'balanced', 'more_control'];

const formatWeight = (value: number): string => value.toFixed(2);
const formatPercent = (value: number): string => `${Math.round(value * 100)}%`;
const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

type AdapterPatch = Partial<Pick<CanvasControlAdapterContract, 'weight' | 'beginEndStepPct' | 'controlMode'>>;

export interface ControlAdapterEditing {
  /** Shows a gesture's value live; the commit that ends it records one undo step. */
  preview(patch: AdapterPatch): void;
  commit(patch: AdapterPatch, label: string): void;
  /** Ends a gesture that came back to where it began, restoring what it previewed. */
  cancel(): void;
}

/** Weight, Steps and Priority: the adapter's own values, edited the same way whatever drives the layer. */
export const ControlAdapterRows = ({
  adapter,
  disabled,
  editing,
  model,
}: {
  adapter: CanvasControlAdapterContract;
  disabled: boolean;
  editing: ControlAdapterEditing;
  model: ControlPanelModel['adapter'];
}) => {
  const { i18n, t } = useTranslation();
  const weightLabel = t('widgets.layers.control.weight');
  const stepsLabel = t('widgets.layers.control.rows.steps');
  const priorityLabel = t('widgets.layers.control.rows.priority');
  const priorityReasonId = useId();

  // Weight: the slider covers 0–2; typed values reach the kind's lower bound.
  const commitWeight = useCallback((weight: number) => editing.commit({ weight }, weightLabel), [editing, weightLabel]);
  const previewWeight = useCallback((weight: number) => editing.preview({ weight }), [editing]);
  const weightGesture = useSliderGesture(adapter.weight, commitWeight, previewWeight);
  const weightValue = useMemo(
    () => [clamp(weightGesture.value, CONTROL_WEIGHT_BOUNDS.sliderMin, CONTROL_WEIGHT_BOUNDS.sliderMax)],
    [weightGesture.value]
  );
  const weightAria = useMemo(() => [weightLabel], [weightLabel]);
  const handleWeightChange = useCallback(
    ({ value }: SliderValueChangeDetails) => weightGesture.onChange(value[0]!),
    [weightGesture]
  );
  const handleWeightChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      const next = value[0]!;
      weightGesture.onChangeEnd(next);
      if (next === adapter.weight) {
        editing.cancel();
      }
    },
    [adapter.weight, editing, weightGesture]
  );
  const minWeight = getControlAdapterMinWeight(model.kind);
  const handleWeightTyped = useNumberCommit(
    useCallback(
      (value: number) => commitWeight(clamp(value, minWeight, CONTROL_WEIGHT_BOUNDS.inputMax)),
      [commitWeight, minWeight]
    )
  );

  // Steps: the readout follows the thumbs; one step is recorded on release.
  const [rangeDraft, setRangeDraft] = useState<[number, number] | null>(null);
  const range = rangeDraft ?? adapter.beginEndStepPct;
  const rangeValue = useMemo(() => [range[0], range[1]], [range]);
  const rangeAria = useMemo(() => [t('widgets.layers.control.beginStep'), t('widgets.layers.control.endStep')], [t]);
  const handleRangeChange = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      if (value.length === 2) {
        const next: [number, number] = [value[0]!, value[1]!];
        setRangeDraft(next);
        editing.preview({ beginEndStepPct: next });
      }
    },
    [editing]
  );
  const handleRangeChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      setRangeDraft(null);
      if (value.length !== 2) {
        return;
      }
      const [begin, end] = adapter.beginEndStepPct;
      if (value[0] === begin && value[1] === end) {
        editing.cancel();
        return;
      }
      editing.commit({ beginEndStepPct: [value[0]!, value[1]!] }, stepsLabel);
    },
    [adapter.beginEndStepPct, editing, stepsLabel]
  );
  const stepsReadout = rangeDraft
    ? t('widgets.layers.control.stepsReadout', {
        begin: Math.round(rangeDraft[0] * 100),
        end: Math.round(rangeDraft[1] * 100),
      })
    : formatPanelText(t, model.stepsReadout, i18n.language);

  const priorityOptions = useMemo(
    () => PRIORITIES.map((value) => ({ label: t(`widgets.layers.control.priorityModes.${value}`), value })),
    [t]
  );
  const handlePriority = useCallback(
    (value: string) => {
      if (value !== model.priority.value) {
        editing.commit({ controlMode: value as ControlPriority }, priorityLabel);
      }
    },
    [editing, model.priority.value, priorityLabel]
  );

  return (
    <>
      <PropertyControlRow label={weightLabel}>
        <Slider
          aria-label={weightAria}
          disabled={disabled}
          formatValue={formatWeight}
          max={CONTROL_WEIGHT_BOUNDS.sliderMax}
          min={CONTROL_WEIGHT_BOUNDS.sliderMin}
          minW="0"
          step={CONTROL_WEIGHT_BOUNDS.step}
          value={weightValue}
          onValueChange={handleWeightChange}
          onValueChangeEnd={handleWeightChangeEnd}
        />
        <FormNumberField
          aria-label={weightLabel}
          disabled={disabled}
          max={CONTROL_WEIGHT_BOUNDS.inputMax}
          min={minWeight}
          step={CONTROL_WEIGHT_BOUNDS.step}
          value={formatWeight(weightGesture.value)}
          onValueCommit={handleWeightTyped}
        />
      </PropertyControlRow>
      <PropertyControlRow label={stepsLabel}>
        <Slider
          aria-label={rangeAria}
          disabled={disabled}
          formatValue={formatPercent}
          max={1}
          min={0}
          minW="0"
          step={STEP_RANGE_STEP}
          value={rangeValue}
          withThumbTooltip
          onValueChange={handleRangeChange}
          onValueChangeEnd={handleRangeChangeEnd}
        />
        <Text color="fg.muted" fontVariantNumeric="tabular-nums" minW="14" textAlign="end">
          {stepsReadout}
        </Text>
      </PropertyControlRow>
      <PropertyControlRow label={priorityLabel}>
        <Box gridColumn="2 / -1" minW="0" opacity={model.priority.disabled ? 0.45 : undefined}>
          <SegmentedControl
            aria-describedby={model.priority.disabled ? priorityReasonId : undefined}
            ariaLabel={priorityLabel}
            disabled={disabled || model.priority.disabled}
            options={priorityOptions}
            value={model.priority.value}
            onChange={handlePriority}
          />
          {model.priority.disabled ? (
            <span hidden id={priorityReasonId}>
              {t('widgets.layers.control.priorityUnavailable')}
            </span>
          ) : null}
        </Box>
      </PropertyControlRow>
    </>
  );
};

import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasControlAdapterContract } from '@workbench/canvas-engine/api';

import { Box, Text } from '@chakra-ui/react';
import { getControlAdapterMinWeight } from '@features/generation/graph';
import { SegmentedControl } from '@platform/ui/SegmentedControl';
import { Slider } from '@platform/ui/Slider';
import { Tooltip } from '@platform/ui/Tooltip';
import { FormNumberField, useNumberCommit } from '@workbench/widgets/canvas/tool-presentation/FormControls';
import { PropertyControlRow } from '@workbench/widgets/canvas/tool-presentation/PropertyPrimitives';
import { useCallback, useId, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { formatPanelText, type ControlPanelModel, type ControlPriority } from './controlPanelModel';
import { CONTROL_WEIGHT_BOUNDS } from './layerOps';

const STEP_RANGE_STEP = 0.05;
const PRIORITIES: readonly ControlPriority[] = ['more_prompt', 'balanced', 'more_control'];
// A layer that stored the retired `unbalanced` mode sees it as a fourth segment, so the value is visible and can be left.
const PRIORITIES_WITH_UNBALANCED: readonly (ControlPriority | 'unbalanced')[] = [...PRIORITIES, 'unbalanced'];

const formatWeight = (value: number): string => value.toFixed(2);
const formatPercent = (value: number): string => `${Math.round(value * 100)}%`;
const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

type AdapterPatch = Partial<Pick<CanvasControlAdapterContract, 'weight' | 'beginEndStepPct' | 'controlMode'>>;

export interface ControlAdapterEditing {
  /** Shows a gesture's value live; the commit that ends it records one undo step. */
  preview(patch: AdapterPatch): void;
  /** Records the patch from the value its gesture's preview replaced; one that changes nothing drops the preview. */
  commit(patch: AdapterPatch, label: string): void;
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

  // Previews reach the document, so `adapter` already carries a dragged value: every gesture end commits, and the
  // commit compares with the value the preview replaced, dropping a gesture that changed nothing. Drafts keep the
  // thumbs on the pointer between previews, which land at most once per frame.
  // The held gesture's last preview, committed as its release would if the rows unmount mid-gesture.
  const pendingRef = useRef<{ label: string; patch: AdapterPatch } | null>(null);
  const previewGesture = useCallback(
    (patch: AdapterPatch, label: string) => {
      pendingRef.current = { label, patch };
      editing.preview(patch);
    },
    [editing]
  );
  const commitGesture = useCallback(
    (patch: AdapterPatch, label: string) => {
      pendingRef.current = null;
      editing.commit(patch, label);
    },
    [editing]
  );
  const flushGestureOnUnmount = useCallback(
    (node: HTMLDivElement | null) =>
      node
        ? () => {
            const pending = pendingRef.current;
            if (pending) {
              commitGesture(pending.patch, pending.label);
            }
          }
        : undefined,
    [commitGesture]
  );

  // Weight: the slider covers 0–2; typed values reach the kind's lower bound.
  const commitWeight = useCallback((weight: number) => editing.commit({ weight }, weightLabel), [editing, weightLabel]);
  const [weightDraft, setWeightDraft] = useState<number | null>(null);
  const weight = weightDraft ?? adapter.weight;
  const weightValue = useMemo(
    () => [clamp(weight, CONTROL_WEIGHT_BOUNDS.sliderMin, CONTROL_WEIGHT_BOUNDS.sliderMax)],
    [weight]
  );
  const weightAria = useMemo(() => [weightLabel], [weightLabel]);
  const handleWeightChange = useCallback(
    ({ value: [next] }: SliderValueChangeDetails) => {
      if (next !== undefined) {
        setWeightDraft(next);
        previewGesture({ weight: next }, weightLabel);
      }
    },
    [previewGesture, weightLabel]
  );
  const handleWeightChangeEnd = useCallback(
    ({ value: [next] }: SliderValueChangeDetails) => {
      setWeightDraft(null);
      if (next !== undefined) {
        commitGesture({ weight: next }, weightLabel);
      }
    },
    [commitGesture, weightLabel]
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
        previewGesture({ beginEndStepPct: next }, stepsLabel);
      }
    },
    [previewGesture, stepsLabel]
  );
  const handleRangeChangeEnd = useCallback(
    ({ value }: SliderValueChangeDetails) => {
      setRangeDraft(null);
      if (value.length === 2) {
        commitGesture({ beginEndStepPct: [value[0]!, value[1]!] }, stepsLabel);
      }
    },
    [commitGesture, stepsLabel]
  );
  const stepsReadout = rangeDraft
    ? t('widgets.layers.control.stepsReadout', {
        begin: Math.round(rangeDraft[0] * 100),
        end: Math.round(rangeDraft[1] * 100),
      })
    : formatPanelText(t, model.stepsReadout, i18n.language);

  const isLegacyUnbalanced = model.priority.value === 'unbalanced';
  const priorityOptions = useMemo(
    () =>
      (isLegacyUnbalanced ? PRIORITIES_WITH_UNBALANCED : PRIORITIES).map((value) => ({
        label: t(`widgets.layers.control.priorityModes.${value}`),
        value,
      })),
    [isLegacyUnbalanced, t]
  );
  const handlePriority = useCallback(
    (value: string) => {
      if (value !== model.priority.value) {
        editing.commit(
          { controlMode: value as NonNullable<CanvasControlAdapterContract['controlMode']> },
          priorityLabel
        );
      }
    },
    [editing, model.priority.value, priorityLabel]
  );
  const priorityUnavailable = model.priority.unavailable
    ? t(
        model.priority.unavailable === 'flux'
          ? 'widgets.layers.control.priorityUnavailableFlux'
          : 'widgets.layers.control.priorityUnavailable'
      )
    : null;

  return (
    // `contents` keeps the rows in the panel's own layout; the box only marks when they unmount.
    <Box ref={flushGestureOnUnmount} display="contents">
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
          value={formatWeight(weight)}
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
        <Box gridColumn="2 / -1" minW="0" opacity={priorityUnavailable ? 0.45 : undefined}>
          {/* Disabled radios take no focus, so the radiogroup itself does while unavailable, showing its reason. */}
          <Tooltip content={priorityUnavailable} disabled={priorityUnavailable === null}>
            <SegmentedControl
              aria-describedby={priorityUnavailable ? priorityReasonId : undefined}
              ariaLabel={priorityLabel}
              disabled={disabled || priorityUnavailable !== null}
              focusVisibleRing="outside"
              options={priorityOptions}
              tabIndex={priorityUnavailable ? 0 : undefined}
              value={model.priority.value}
              onChange={handlePriority}
            />
          </Tooltip>
          {priorityUnavailable ? (
            <span hidden id={priorityReasonId}>
              {priorityUnavailable}
            </span>
          ) : null}
        </Box>
      </PropertyControlRow>
    </Box>
  );
};

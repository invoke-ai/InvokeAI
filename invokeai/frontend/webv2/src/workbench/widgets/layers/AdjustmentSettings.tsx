import type { SelectValueChangeDetails, SliderValueChangeDetails } from '@chakra-ui/react';
import type {
  CanvasAdjustmentCurves,
  CanvasAdjustmentEntry,
  CanvasRasterLayerContractV2,
} from '@workbench/canvas-engine/api';
import type { CanvasPreparedEngine } from '@workbench/widgets/canvas/useStructuralCommit';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { chakra, createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { Field, Select, Slider } from '@platform/ui';
import { buildCurveLut } from '@workbench/canvas-engine/api';
import { useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  CURVE_PADDING,
  CURVE_SIZE,
  curvePointFromSvg,
  curvePointToSvg,
  finishCurveDragResult,
  getCurveGridCoordinates,
} from './curveEditorMath';

/**
 * Edit adjustment subrows through whole-stack patch-config. Preview gestures live and commit once; tree rows own
 * enable/remove/reorder.
 */

export type AdjustmentsEngine = CanvasStructuralEngine & CanvasPreparedEngine;

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: true } as const;

const CurveSvg = chakra('svg');
const CurveRect = chakra('rect');
const CurveLine = chakra('line');
const CurveGroup = chakra('g');
const CurvePath = chakra('path');
const CurveHandle = chakra('circle');

const CURVE_SVG_CSS = {
  aspectRatio: '1',
  borderRadius: 'l2',
  maxWidth: `${CURVE_SIZE}px`,
  touchAction: 'none',
  userSelect: 'none',
  width: 'full',
};

const CURVE_EDITOR_CSS = { userSelect: 'none' };
const CURVE_HANDLE_CSS = { cursor: 'grab', _active: { cursor: 'grabbing' } };

const preventDefault = (event: { preventDefault: () => void }): void => event.preventDefault();

type CurveChannel = 'r' | 'g' | 'b';
const CURVE_CHANNELS: readonly CurveChannel[] = ['r', 'g', 'b'];

const IDENTITY_CURVE: [number, number][] = [
  [0, 0],
  [255, 255],
];

const formatSigned = (value: number): string => `${value > 0 ? '+' : ''}${Math.round(value * 100)}`;

/** An adjustment stack's owner: a raster layer or a raster-stack group. */
export type AdjustmentOwner = Pick<CanvasRasterLayerContractV2, 'id' | 'adjustments'> & {
  type: 'raster' | 'group';
};

export const AdjustmentSettings = ({
  engine,
  entryId,
  layer,
}: {
  engine: AdjustmentsEngine | null;
  entryId: string;
  layer: AdjustmentOwner;
}) => {
  const entry = layer.adjustments?.find((candidate) => candidate.id === entryId);
  if (!entry) {
    return null;
  }
  return <AdjustmentEntryEditor engine={engine} entry={entry} layer={layer} />;
};

const AdjustmentEntryEditor = ({
  engine,
  entry,
  layer,
}: {
  engine: AdjustmentsEngine | null;
  entry: CanvasAdjustmentEntry;
  layer: AdjustmentOwner;
}) => {
  const { cancel: cancelPreview, commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const { t } = useTranslation();
  const gestureBaselineRef = useRef<readonly CanvasAdjustmentEntry[] | null>(null);
  const configOf = useCallback(
    (adjustments: CanvasAdjustmentEntry[]) =>
      layer.type === 'group'
        ? { adjustments, layerType: 'group' as const }
        : { adjustments, layerType: 'raster' as const },
    [layer.type]
  );

  const patchLive = useCallback(
    (next: CanvasAdjustmentEntry) => {
      const entries = layer.adjustments ?? [];
      gestureBaselineRef.current ??= entries;
      previewStructural({
        config: configOf(entries.map((candidate) => (candidate.id === next.id ? next : candidate))),
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [configOf, previewStructural, layer.adjustments, layer.id]
  );

  const commitEntry = useCallback(
    (label: string, next: CanvasAdjustmentEntry) => {
      const entries = layer.adjustments ?? [];
      const baseline = gestureBaselineRef.current ?? entries;
      gestureBaselineRef.current = null;
      // The tree dot owns enablement; a toggle landing mid-gesture stays put.
      const committed = { ...next, isEnabled: entry.isEnabled };
      commitPrepared(label, (model) =>
        model.prepare({
          before: configOf([...baseline]),
          config: configOf(entries.map((candidate) => (candidate.id === committed.id ? committed : candidate))),
          id: layer.id,
          type: 'patch-config',
        })
      );
    },
    [commitPrepared, configOf, entry.isEnabled, layer.adjustments, layer.id]
  );

  const cancelGesture = useCallback(() => {
    const baseline = gestureBaselineRef.current;
    gestureBaselineRef.current = null;
    if (baseline) {
      cancelPreview({ config: configOf([...baseline]), id: layer.id, type: 'updateCanvasLayerConfig' });
    }
  }, [cancelPreview, configOf, layer.id]);

  const handleScalarLive = useCallback(
    (field: ScalarField, next: number) => patchLive({ ...entry, [field]: next } as CanvasAdjustmentEntry),
    [entry, patchLive]
  );
  const handleScalarCommit = useCallback(
    (field: ScalarField, next: number) =>
      commitEntry(t(SCALAR_LABEL_KEYS[field]), { ...entry, [field]: next } as CanvasAdjustmentEntry),
    [commitEntry, entry, t]
  );
  const handleLevelsLive = useCallback(
    (patch: Partial<LevelsEntry>) => patchLive({ ...entry, ...patch } as CanvasAdjustmentEntry),
    [entry, patchLive]
  );
  const handleLevelsCommit = useCallback(
    (patch: Partial<LevelsEntry>) =>
      commitEntry(t('widgets.layers.adjustments.levels'), { ...entry, ...patch } as CanvasAdjustmentEntry),
    [commitEntry, entry, t]
  );
  const handleCurvesLive = useCallback(
    (curves: CanvasAdjustmentCurves) => patchLive({ ...entry, curves } as CanvasAdjustmentEntry),
    [entry, patchLive]
  );
  const handleCurvesCommit = useCallback(
    (curves: CanvasAdjustmentCurves) =>
      commitEntry(t('widgets.layers.adjustments.curves'), { ...entry, curves } as CanvasAdjustmentEntry),
    [commitEntry, entry, t]
  );

  switch (entry.type) {
    case 'brightness-contrast':
      return (
        <Stack gap="3">
          <ScalarSlider
            field="brightness"
            label={t('widgets.layers.adjustments.brightness')}
            value={entry.brightness}
            onCommit={handleScalarCommit}
            onLive={handleScalarLive}
          />
          <ScalarSlider
            field="contrast"
            label={t('widgets.layers.adjustments.contrast')}
            value={entry.contrast}
            onCommit={handleScalarCommit}
            onLive={handleScalarLive}
          />
        </Stack>
      );
    case 'exposure':
      return (
        <ScalarSlider
          field="stops"
          formatValue={formatStops}
          label={t('widgets.layers.adjustments.exposure')}
          max={5}
          min={-5}
          step={0.05}
          value={entry.stops}
          onCommit={handleScalarCommit}
          onLive={handleScalarLive}
        />
      );
    case 'hsl':
      return (
        <ScalarSlider
          field="saturation"
          label={t('widgets.layers.adjustments.saturation')}
          value={entry.saturation}
          onCommit={handleScalarCommit}
          onLive={handleScalarLive}
        />
      );
    case 'hue':
      return (
        <ScalarSlider
          field="rotation"
          formatValue={formatDegrees}
          label={t('widgets.layers.adjustments.hue')}
          max={180}
          min={-180}
          step={1}
          value={entry.rotation}
          onCommit={handleScalarCommit}
          onLive={handleScalarLive}
        />
      );
    case 'levels':
      return <LevelsEditor entry={entry} onCommit={handleLevelsCommit} onLive={handleLevelsLive} />;
    case 'invert':
      return (
        <Text color="fg.muted" fontSize="md">
          {t('widgets.layers.adjustments.invertHint')}
        </Text>
      );
    case 'curves':
      return (
        <CurvesEditor
          curves={entry.curves}
          onCancel={cancelGesture}
          onCommit={handleCurvesCommit}
          onLive={handleCurvesLive}
        />
      );
  }
};

type ScalarField = 'brightness' | 'contrast' | 'saturation' | 'rotation' | 'stops';
type LevelsEntry = Extract<CanvasAdjustmentEntry, { type: 'levels' }>;

const SCALAR_LABEL_KEYS: Record<ScalarField, string> = {
  brightness: 'widgets.layers.adjustments.brightness',
  contrast: 'widgets.layers.adjustments.contrast',
  rotation: 'widgets.layers.adjustments.hue',
  saturation: 'widgets.layers.adjustments.saturation',
  stops: 'widgets.layers.adjustments.exposure',
};

const formatDegrees = (value: number): string => `${Math.round(value)}°`;
const formatStops = (value: number): string => `${value > 0 ? '+' : ''}${value.toFixed(2)} EV`;

const LEVELS_CHANNELS = ['rgb', 'r', 'g', 'b'] as const;

const ScalarSlider = ({
  field,
  formatValue = formatSigned,
  label,
  max = 1,
  min = -1,
  onCommit,
  onLive,
  step = 0.01,
  value,
}: {
  field: ScalarField;
  formatValue?: (value: number) => string;
  label: string;
  max?: number;
  min?: number;
  step?: number;
  value: number;
  onLive: (field: ScalarField, next: number) => void;
  onCommit: (field: ScalarField, next: number) => void;
}) => {
  const sliderValue = useMemo(() => [value], [value]);
  const aria = useMemo(() => [label], [label]);

  const handleChange = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const next = v[0];
      if (next !== undefined && Number.isFinite(next)) {
        onLive(field, next);
      }
    },
    [field, onLive]
  );

  const handleChangeEnd = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const next = v[0];
      if (next !== undefined && Number.isFinite(next)) {
        onCommit(field, next);
      }
    },
    [field, onCommit]
  );

  return (
    <Field label={label}>
      <Slider
        aria-label={aria}
        formatValue={formatValue}
        max={max}
        min={min}
        step={step}
        value={sliderValue}
        withThumbTooltip
        onValueChange={handleChange}
        onValueChangeEnd={handleChangeEnd}
      />
    </Field>
  );
};

const formatGamma = (value: number): string => value.toFixed(2);

/** A pair-of-thumbs range plus a gamma midtone: the classic input/output levels remap. */
const LevelsEditor = ({
  entry,
  onCommit,
  onLive,
}: {
  entry: LevelsEntry;
  onLive: (patch: Partial<LevelsEntry>) => void;
  onCommit: (patch: Partial<LevelsEntry>) => void;
}) => {
  const { t } = useTranslation();
  const inputValue = useMemo(() => [entry.inBlack, entry.inWhite], [entry.inBlack, entry.inWhite]);
  const gammaValue = useMemo(() => [entry.gamma], [entry.gamma]);
  const outputValue = useMemo(() => [entry.outBlack, entry.outWhite], [entry.outBlack, entry.outWhite]);
  const inputAria = useMemo(
    () => [t('widgets.layers.adjustments.inputBlack'), t('widgets.layers.adjustments.inputWhite')],
    [t]
  );
  const gammaAria = useMemo(() => [t('widgets.layers.adjustments.gamma')], [t]);
  const outputAria = useMemo(
    () => [t('widgets.layers.adjustments.outputBlack'), t('widgets.layers.adjustments.outputWhite')],
    [t]
  );

  const channelCollection = useMemo(
    () =>
      createListCollection({
        items: LEVELS_CHANNELS.map((channel) => ({
          label: t(`widgets.layers.adjustments.channels.${channel}`),
          value: channel,
        })),
      }),
    [t]
  );
  const channelValue = useMemo(() => [entry.channel ?? 'rgb'], [entry.channel]);
  const handleChannelChange = useCallback(
    ({ value: v }: SelectValueChangeDetails) => {
      const channel = v[0] as LevelsEntry['channel'] | undefined;
      if (channel && channel !== (entry.channel ?? 'rgb')) {
        onCommit({ channel });
      }
    },
    [entry.channel, onCommit]
  );

  const rangePatch = useCallback(
    (v: number[], black: 'inBlack' | 'outBlack', white: 'inWhite' | 'outWhite'): Partial<LevelsEntry> | null =>
      v[0] !== undefined && v[1] !== undefined ? { [black]: v[0], [white]: v[1] } : null,
    []
  );
  const handleInputChange = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const patch = rangePatch(v, 'inBlack', 'inWhite');
      if (patch) {
        onLive(patch);
      }
    },
    [onLive, rangePatch]
  );
  const handleInputCommit = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const patch = rangePatch(v, 'inBlack', 'inWhite');
      if (patch) {
        onCommit(patch);
      }
    },
    [onCommit, rangePatch]
  );
  const handleOutputChange = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const patch = rangePatch(v, 'outBlack', 'outWhite');
      if (patch) {
        onLive(patch);
      }
    },
    [onLive, rangePatch]
  );
  const handleOutputCommit = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const patch = rangePatch(v, 'outBlack', 'outWhite');
      if (patch) {
        onCommit(patch);
      }
    },
    [onCommit, rangePatch]
  );
  const handleGammaChange = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      if (v[0] !== undefined && Number.isFinite(v[0])) {
        onLive({ gamma: v[0] });
      }
    },
    [onLive]
  );
  const handleGammaCommit = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      if (v[0] !== undefined && Number.isFinite(v[0])) {
        onCommit({ gamma: v[0] });
      }
    },
    [onCommit]
  );

  return (
    <Stack gap="3">
      <Field label={t('widgets.layers.adjustments.channel')}>
        <Select
          aria-label={t('widgets.layers.adjustments.channel')}
          collection={channelCollection}
          positioning={SELECT_POSITIONING}
          value={channelValue}
          valueText={t(`widgets.layers.adjustments.channels.${entry.channel ?? 'rgb'}`)}
          onValueChange={handleChannelChange}
        />
      </Field>
      <Field label={t('widgets.layers.adjustments.inputLevels')}>
        <Slider
          aria-label={inputAria}
          max={255}
          min={0}
          minStepsBetweenThumbs={1}
          step={1}
          value={inputValue}
          withThumbTooltip
          onValueChange={handleInputChange}
          onValueChangeEnd={handleInputCommit}
        />
      </Field>
      <Field label={t('widgets.layers.adjustments.gamma')}>
        <Slider
          aria-label={gammaAria}
          formatValue={formatGamma}
          max={4}
          min={0.1}
          step={0.01}
          value={gammaValue}
          withThumbTooltip
          onValueChange={handleGammaChange}
          onValueChangeEnd={handleGammaCommit}
        />
      </Field>
      <Field label={t('widgets.layers.adjustments.outputLevels')}>
        <Slider
          aria-label={outputAria}
          max={255}
          min={0}
          step={1}
          value={outputValue}
          withThumbTooltip
          onValueChange={handleOutputChange}
          onValueChangeEnd={handleOutputCommit}
        />
      </Field>
    </Stack>
  );
};

const sameChannel = (left: CanvasAdjustmentCurves, right: CanvasAdjustmentCurves, channel: CurveChannel): boolean =>
  JSON.stringify(left[channel] ?? IDENTITY_CURVE) === JSON.stringify(right[channel] ?? IDENTITY_CURVE);

const CurvesEditor = ({
  curves,
  onCancel,
  onCommit,
  onLive,
}: {
  curves: CanvasAdjustmentCurves;
  onLive: (curves: CanvasAdjustmentCurves) => void;
  onCancel: () => void;
  onCommit: (curves: CanvasAdjustmentCurves) => void;
}) => {
  const { t } = useTranslation();
  const [channel, setChannel] = useState<CurveChannel>('r');
  const dragIndexRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const beforeRef = useRef<CanvasAdjustmentCurves | null>(null);
  const latestPointsRef = useRef<[number, number][] | null>(null);
  const dragTargetRef = useRef<Element | null>(null);

  const points = useMemo<[number, number][]>(() => {
    const raw = curves[channel];
    return raw && raw.length >= 2 ? [...raw].map(([x, y]) => [x, y] as [number, number]) : [...IDENTITY_CURVE];
  }, [channel, curves]);

  const channelCollection = useMemo(
    () =>
      createListCollection({
        items: CURVE_CHANNELS.map((c) => ({ label: t(`widgets.layers.adjustments.channels.${c}`), value: c })),
      }),
    [t]
  );

  const svgPointFromEvent = useCallback((event: { clientX: number; clientY: number }): { px: number; py: number } => {
    const svg = svgRef.current;
    if (!svg) {
      return { px: 0, py: 0 };
    }
    const rect = svg.getBoundingClientRect();
    return {
      px: ((event.clientX - rect.left) / rect.width) * CURVE_SIZE,
      py: ((event.clientY - rect.top) / rect.height) * CURVE_SIZE,
    };
  }, []);

  const lutPath = useMemo(() => {
    const lut = buildCurveLut(points);
    let d = '';
    for (let i = 0; i < 256; i += 4) {
      const { cx, cy } = curvePointToSvg(i, lut[i]);
      d += `${i === 0 ? 'M' : 'L'}${cx.toFixed(1)},${cy.toFixed(1)} `;
    }
    return d.trim();
  }, [points]);
  const gridCoordinates = getCurveGridCoordinates();

  const handleChannelChange = useCallback(
    ({ value }: { value: string[] }) => setChannel((value[0] as CurveChannel) ?? 'r'),
    []
  );

  const handlePointDown = useCallback(
    (event: ReactPointerEvent<SVGCircleElement>) => {
      event.stopPropagation();
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      dragIndexRef.current = Number(event.currentTarget.dataset.index);
      dragTargetRef.current = event.currentTarget;
      beforeRef.current = curves;
      latestPointsRef.current = null;
    },
    [curves]
  );

  const handleMove = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      const index = dragIndexRef.current;
      if (index === null) {
        return;
      }
      const { px, py } = svgPointFromEvent(event);
      const [nx, ny] = curvePointFromSvg(px, py);
      const isEndpoint = index === 0 || index === points.length - 1;
      const next = points.map((p, i) => {
        if (i !== index) {
          return p;
        }
        return isEndpoint ? ([p[0], ny] as [number, number]) : ([nx, ny] as [number, number]);
      });
      if (!isEndpoint) {
        const lo = next[index - 1][0] + 1;
        const hi = next[index + 1][0] - 1;
        next[index] = [Math.max(lo, Math.min(hi, next[index][0])), next[index][1]];
      }
      latestPointsRef.current = next;
      onLive({ ...curves, [channel]: next });
    },
    [channel, curves, onLive, points, svgPointFromEvent]
  );

  const finishDrag = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>, cancelled: boolean) => {
      const wasDragging = dragIndexRef.current !== null;
      const dragTarget = dragTargetRef.current;
      if (dragTarget?.hasPointerCapture(event.pointerId)) {
        dragTarget.releasePointerCapture(event.pointerId);
      }
      dragIndexRef.current = null;
      dragTargetRef.current = null;
      const before = beforeRef.current;
      const finalPoints = latestPointsRef.current;
      beforeRef.current = null;
      latestPointsRef.current = null;
      if (wasDragging && before && finalPoints) {
        const current = { ...before, [channel]: finalPoints };
        finishCurveDragResult({
          before,
          cancelled: cancelled || sameChannel(current, before, channel),
          current,
          onCommit,
          onPreview: () => onCancel(),
        });
      }
    },
    [channel, onCancel, onCommit]
  );

  const handleUp = useCallback((event: ReactPointerEvent<SVGSVGElement>) => finishDrag(event, false), [finishDrag]);
  const handleCancel = useCallback((event: ReactPointerEvent<SVGSVGElement>) => finishDrag(event, true), [finishDrag]);

  const handleAdd = useCallback(
    (event: ReactPointerEvent<SVGSVGElement>) => {
      if (dragIndexRef.current !== null) {
        return;
      }
      const { px, py } = svgPointFromEvent(event);
      const [nx, ny] = curvePointFromSvg(px, py);
      if (nx <= 0 || nx >= 255) {
        return;
      }
      const next = [...points, [nx, ny] as [number, number]].sort((a, b) => a[0] - b[0]);
      onCommit({ ...curves, [channel]: next });
    },
    [channel, curves, onCommit, points, svgPointFromEvent]
  );

  const handleRemove = useCallback(
    (event: ReactPointerEvent<SVGCircleElement>) => {
      event.stopPropagation();
      const index = Number(event.currentTarget.dataset.index);
      if (index === 0 || index === points.length - 1 || points.length <= 2) {
        return;
      }
      onCommit({ ...curves, [channel]: points.filter((_, i) => i !== index) });
    },
    [channel, curves, onCommit, points]
  );

  const channelValue = useMemo(() => [channel], [channel]);

  return (
    <Stack css={CURVE_EDITOR_CSS} gap="2">
      <HStack justify="space-between">
        <Text fontSize="md" fontWeight="medium">
          {t('widgets.layers.adjustments.curves')}
        </Text>
        <Select
          aria-label={t('widgets.layers.adjustments.channel')}
          collection={channelCollection}
          positioning={SELECT_POSITIONING}
          value={channelValue}
          valueText={t(`widgets.layers.adjustments.channels.${channel}`)}
          w="6rem"
          onValueChange={handleChannelChange}
        />
      </HStack>
      <CurveSvg
        bg="bg.inset"
        css={CURVE_SVG_CSS}
        onDoubleClick={handleAdd}
        onPointerCancel={handleCancel}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        ref={svgRef}
        viewBox={`0 0 ${CURVE_SIZE} ${CURVE_SIZE}`}
      >
        <CurveRect
          fill="bg.inset"
          height={CURVE_SIZE - CURVE_PADDING * 2}
          width={CURVE_SIZE - CURVE_PADDING * 2}
          x={CURVE_PADDING}
          y={CURVE_PADDING}
        />
        <CurveGroup stroke="fg.grid">
          {gridCoordinates.map((coordinate) => (
            <g key={coordinate}>
              <line
                vectorEffect="non-scaling-stroke"
                x1={coordinate}
                x2={coordinate}
                y1={CURVE_PADDING}
                y2={CURVE_SIZE - CURVE_PADDING}
              />
              <line
                vectorEffect="non-scaling-stroke"
                x1={CURVE_PADDING}
                x2={CURVE_SIZE - CURVE_PADDING}
                y1={coordinate}
                y2={coordinate}
              />
            </g>
          ))}
        </CurveGroup>
        <CurveRect
          fill="none"
          height={CURVE_SIZE - CURVE_PADDING * 2}
          stroke="border.emphasized"
          vectorEffect="non-scaling-stroke"
          width={CURVE_SIZE - CURVE_PADDING * 2}
          x={CURVE_PADDING}
          y={CURVE_PADDING}
        />
        <CurveLine
          stroke="fg.subtle"
          strokeDasharray="4 4"
          vectorEffect="non-scaling-stroke"
          x1={CURVE_PADDING}
          x2={CURVE_SIZE - CURVE_PADDING}
          y1={CURVE_SIZE - CURVE_PADDING}
          y2={CURVE_PADDING}
        />
        <CurvePath
          d={lutPath}
          fill="none"
          stroke="accent.solid"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        {points.map((p, i) => {
          const { cx, cy } = curvePointToSvg(p[0], p[1]);
          return (
            <CurveHandle
              cx={cx}
              cy={cy}
              css={CURVE_HANDLE_CSS}
              data-index={i}
              fill="accent.solid"
              key={i}
              onContextMenu={preventDefault}
              onDoubleClick={handleRemove}
              onPointerDown={handlePointDown}
              r={5}
              stroke="bg.inset"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </CurveSvg>
      <Text color="fg.muted" fontSize="xs">
        {t('widgets.layers.adjustments.curvesHint')}
      </Text>
    </Stack>
  );
};

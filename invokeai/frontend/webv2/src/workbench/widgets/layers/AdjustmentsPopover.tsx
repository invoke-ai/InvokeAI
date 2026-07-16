import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/types';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { DEFAULT_ADJUSTMENTS, buildCurveLut } from '@workbench/canvas-engine/render/adjustments';
import { Button, Field, Select, Slider } from '@workbench/components/ui';
import { useCanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
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
import { applyStructural, applyStructuralPreview } from './layerOps';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

type CurveChannel = 'r' | 'g' | 'b';
const CURVE_CHANNELS: readonly CurveChannel[] = ['r', 'g', 'b'];

/** The identity curve control points (diagonal). */
const IDENTITY_CURVE: [number, number][] = [
  [0, 0],
  [255, 255],
];

const withCurve = (
  base: CanvasAdjustmentsContract,
  channel: CurveChannel,
  points: [number, number][]
): CanvasAdjustmentsContract => ({
  ...base,
  curves: {
    b: base.curves?.b ?? IDENTITY_CURVE,
    g: base.curves?.g ?? IDENTITY_CURVE,
    r: base.curves?.r ?? IDENTITY_CURVE,
    [channel]: points,
  },
});

const formatSigned = (value: number): string => `${value > 0 ? '+' : ''}${Math.round(value * 100)}`;

interface AdjustmentsPopoverProps {
  engine: CanvasStructuralEngine | null;
  layer: CanvasRasterLayerContractV2;
}

/**
 * Non-destructive raster-adjustment editor (plan §1.3): brightness / contrast /
 * saturation sliders plus a minimal per-channel curves editor and a reset. All
 * edits patch the layer's `adjustments` through the canvas undo stack
 * (`applyStructural` → `updateCanvasLayerConfig`); sliders use the same
 * draft/commit pattern as opacity (one history entry per drag). The curve math
 * lives in the pure `render/adjustments` module — this component only manages
 * control points and previews the LUT.
 *
 * Rendered inline inside the per-layer properties popover (round 3): a nested
 * popover would be an "outside interaction" for its parent and close it, so this
 * renders as a plain section rather than its own popover.
 */
export const AdjustmentsPopover = ({ engine, layer }: AdjustmentsPopoverProps) => {
  const adjustments = layer.adjustments ?? DEFAULT_ADJUSTMENTS;
  return <AdjustmentsControls adjustments={adjustments} engine={engine} layer={layer} />;
};

interface AdjustmentsControlsProps {
  adjustments: CanvasAdjustmentsContract;
  engine: CanvasStructuralEngine | null;
  layer: CanvasRasterLayerContractV2;
}

type ScalarKey = 'brightness' | 'contrast' | 'saturation';

const AdjustmentsControls = ({ adjustments, engine, layer }: AdjustmentsControlsProps) => {
  const { t } = useTranslation();
  const dispatch = useCanvasProjectMutationDispatch();

  const patchLive = useCallback(
    (next: CanvasAdjustmentsContract) => {
      applyStructuralPreview(engine, dispatch, {
        config: { adjustments: next, layerType: 'raster' },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [dispatch, engine, layer.id]
  );

  const commit = useCallback(
    (label: string, next: CanvasAdjustmentsContract, before: CanvasAdjustmentsContract) => {
      applyStructural(
        engine,
        dispatch,
        label,
        { config: { adjustments: next, layerType: 'raster' }, id: layer.id, type: 'updateCanvasLayerConfig' },
        { config: { adjustments: before, layerType: 'raster' }, id: layer.id, type: 'updateCanvasLayerConfig' }
      );
    },
    [dispatch, engine, layer.id]
  );

  const handleScalarLive = useCallback(
    (key: ScalarKey, next: number) => patchLive({ ...adjustments, [key]: next }),
    [adjustments, patchLive]
  );

  const handleScalarCommit = useCallback(
    (label: string, key: ScalarKey, next: number, before: CanvasAdjustmentsContract) => {
      commit(label, { ...before, [key]: next }, before);
    },
    [commit]
  );

  const handleReset = useCallback(() => {
    commit(t('widgets.layers.adjustments.reset'), { ...DEFAULT_ADJUSTMENTS }, adjustments);
  }, [adjustments, commit, t]);

  // Live (render-only) during a curve-point drag: preview without pushing history.
  const handleCurveLive = useCallback(
    (channel: CurveChannel, points: [number, number][]) => {
      patchLive(withCurve(adjustments, channel, points));
    },
    [adjustments, patchLive]
  );

  const handleCurveCancel = useCallback((before: CanvasAdjustmentsContract) => patchLive(before), [patchLive]);

  // Single history entry per gesture (drag end, click-add, dbl-click-remove). The
  // `before` snapshot is captured at gesture start by the editor (during a drag
  // `adjustments` has already advanced via the live previews), so it undoes the
  // WHOLE gesture rather than the last frame.
  const handleCurveCommit = useCallback(
    (current: CanvasAdjustmentsContract, before: CanvasAdjustmentsContract) => {
      commit(t('widgets.layers.adjustments.curves'), current, before);
    },
    [commit, t]
  );

  return (
    <Stack gap="3">
      <AdjustmentSlider
        adjustments={adjustments}
        adjustmentKey="brightness"
        label={t('widgets.layers.adjustments.brightness')}
        onCommit={handleScalarCommit}
        onLive={handleScalarLive}
      />
      <AdjustmentSlider
        adjustments={adjustments}
        adjustmentKey="contrast"
        label={t('widgets.layers.adjustments.contrast')}
        onCommit={handleScalarCommit}
        onLive={handleScalarLive}
      />
      <AdjustmentSlider
        adjustments={adjustments}
        adjustmentKey="saturation"
        label={t('widgets.layers.adjustments.saturation')}
        onCommit={handleScalarCommit}
        onLive={handleScalarLive}
      />
      <CurvesEditor
        adjustments={adjustments}
        onCancel={handleCurveCancel}
        onCommit={handleCurveCommit}
        onLive={handleCurveLive}
      />
      <Button size="xs" variant="ghost" onClick={handleReset}>
        {t('widgets.layers.adjustments.reset')}
      </Button>
    </Stack>
  );
};

interface AdjustmentSliderProps {
  label: string;
  adjustmentKey: ScalarKey;
  adjustments: CanvasAdjustmentsContract;
  onLive: (key: ScalarKey, next: number) => void;
  onCommit: (label: string, key: ScalarKey, next: number, before: CanvasAdjustmentsContract) => void;
}

/** A single -1..1 adjustment slider owning its own draft/before (one history entry per drag). */
const AdjustmentSlider = ({ adjustmentKey, adjustments, label, onCommit, onLive }: AdjustmentSliderProps) => {
  const beforeRef = useRef<CanvasAdjustmentsContract | null>(null);
  const value = adjustments[adjustmentKey] ?? 0;
  const sliderValue = useMemo(() => [value], [value]);
  const aria = useMemo(() => [label], [label]);

  const handleChange = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const next = v[0];
      if (next === undefined || !Number.isFinite(next)) {
        return;
      }
      if (beforeRef.current === null) {
        beforeRef.current = adjustments;
      }
      onLive(adjustmentKey, next);
    },
    [adjustmentKey, adjustments, onLive]
  );

  const handleChangeEnd = useCallback(
    ({ value: v }: SliderValueChangeDetails) => {
      const next = v[0];
      const before = beforeRef.current ?? adjustments;
      beforeRef.current = null;
      if (next === undefined || !Number.isFinite(next)) {
        return;
      }
      onCommit(label, adjustmentKey, next, before);
    },
    [adjustmentKey, adjustments, label, onCommit]
  );

  return (
    <Field label={label}>
      <Slider
        aria-label={aria}
        formatValue={formatSigned}
        max={1}
        min={-1}
        size="sm"
        step={0.01}
        value={sliderValue}
        withThumbTooltip
        onValueChange={handleChange}
        onValueChangeEnd={handleChangeEnd}
      />
    </Field>
  );
};

interface CurvesEditorProps {
  adjustments: CanvasAdjustmentsContract;
  /** Render-only preview during a point drag (no history entry). */
  onLive: (channel: CurveChannel, points: [number, number][]) => void;
  /** Restores the pre-drag snapshot when the browser cancels a gesture. */
  onCancel: (before: CanvasAdjustmentsContract) => void;
  /** Commits one history entry for a completed gesture, undoing to `before`. */
  onCommit: (current: CanvasAdjustmentsContract, before: CanvasAdjustmentsContract) => void;
}

/** A compact per-channel curves editor (SVG): drag points, click to add, double-click to remove. */
const CurvesEditor = ({ adjustments, onCancel, onCommit, onLive }: CurvesEditorProps) => {
  const { t } = useTranslation();
  const [channel, setChannel] = useState<CurveChannel>('r');
  const dragIndexRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  // A point drag streams render-only previews; `beforeRef` snapshots the
  // adjustments at drag start and `latestPointsRef` holds the last previewed
  // points, so pointer-up commits the whole drag as ONE history entry (mirrors
  // the scalar sliders' live/commit split — no per-frame undo-stack flooding).
  const beforeRef = useRef<CanvasAdjustmentsContract | null>(null);
  const latestPointsRef = useRef<[number, number][] | null>(null);
  const dragTargetRef = useRef<Element | null>(null);

  const points = useMemo<[number, number][]>(() => {
    const raw = adjustments.curves?.[channel];
    return raw && raw.length >= 2 ? [...raw].map(([x, y]) => [x, y] as [number, number]) : [...IDENTITY_CURVE];
  }, [adjustments.curves, channel]);

  const channelCollection = useMemo(
    () =>
      createListCollection({
        items: CURVE_CHANNELS.map((c) => ({ label: t(`widgets.layers.adjustments.channels.${c}`), value: c })),
      }),
    [t]
  );

  const svgPointFromEvent = (event: ReactPointerEvent<SVGElement>): { px: number; py: number } => {
    const svg = svgRef.current;
    if (!svg) {
      return { px: 0, py: 0 };
    }
    const rect = svg.getBoundingClientRect();
    return {
      px: ((event.clientX - rect.left) / rect.width) * CURVE_SIZE,
      py: ((event.clientY - rect.top) / rect.height) * CURVE_SIZE,
    };
  };

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

  const handlePointDown = (index: number) => (event: ReactPointerEvent<SVGCircleElement>) => {
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragIndexRef.current = index;
    dragTargetRef.current = event.currentTarget;
    // Snapshot the pre-drag state once, for a single whole-drag history entry.
    beforeRef.current = adjustments;
    latestPointsRef.current = null;
  };

  const handleMove = (event: ReactPointerEvent<SVGSVGElement>) => {
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
      // Endpoints keep their x anchored (0 / 255); only y moves.
      return isEndpoint ? ([p[0], ny] as [number, number]) : ([nx, ny] as [number, number]);
    });
    // Keep interior x within its neighbours to preserve monotonic ordering.
    if (!isEndpoint) {
      const lo = next[index - 1][0] + 1;
      const hi = next[index + 1][0] - 1;
      next[index] = [Math.max(lo, Math.min(hi, next[index][0])), next[index][1]];
    }
    latestPointsRef.current = next;
    onLive(channel, next);
  };

  const finishDrag = (event: ReactPointerEvent<SVGSVGElement>, cancelled: boolean) => {
    const wasDragging = dragIndexRef.current !== null;
    const dragTarget = dragTargetRef.current;
    if (dragTarget?.hasPointerCapture(event.pointerId)) {
      dragTarget.releasePointerCapture(event.pointerId);
    }
    dragIndexRef.current = null;
    dragTargetRef.current = null;
    // Commit the whole drag as one history entry (only if the point actually
    // moved — a click with no move streams no previews and needs no commit).
    const before = beforeRef.current;
    const finalPoints = latestPointsRef.current;
    beforeRef.current = null;
    latestPointsRef.current = null;
    if (wasDragging && before && finalPoints) {
      finishCurveDragResult({
        before,
        cancelled,
        current: withCurve(before, channel, finalPoints),
        onCommit: (current) => onCommit(current, before),
        onPreview: onCancel,
      });
    }
  };

  const handleUp = (event: ReactPointerEvent<SVGSVGElement>) => finishDrag(event, false);
  const handleCancel = (event: ReactPointerEvent<SVGSVGElement>) => finishDrag(event, true);

  const handleAdd = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (dragIndexRef.current !== null) {
      return;
    }
    const { px, py } = svgPointFromEvent(event);
    const [nx, ny] = curvePointFromSvg(px, py);
    if (nx <= 0 || nx >= 255) {
      return;
    }
    const next = [...points, [nx, ny] as [number, number]].sort((a, b) => a[0] - b[0]);
    onCommit(withCurve(adjustments, channel, next), adjustments);
  };

  const handleRemove = (index: number) => (event: ReactPointerEvent<SVGCircleElement>) => {
    event.stopPropagation();
    if (index === 0 || index === points.length - 1 || points.length <= 2) {
      return;
    }
    onCommit(
      withCurve(
        adjustments,
        channel,
        points.filter((_, i) => i !== index)
      ),
      adjustments
    );
  };

  const channelValue = useMemo(() => [channel], [channel]);

  return (
    <Stack gap="2">
      <HStack justify="space-between">
        <Text fontSize="xs" fontWeight="medium">
          {t('widgets.layers.adjustments.curves')}
        </Text>
        <Select
          aria-label={t('widgets.layers.adjustments.channel')}
          collection={channelCollection}
          positioning={SELECT_POSITIONING}
          size="xs"
          value={channelValue}
          valueText={t(`widgets.layers.adjustments.channels.${channel}`)}
          w="6rem"
          onValueChange={handleChannelChange}
        />
      </HStack>
      <svg
        height={CURVE_SIZE}
        onDoubleClick={handleAdd}
        onPointerCancel={handleCancel}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        ref={svgRef}
        style={{
          background: 'var(--chakra-colors-bg-inset)',
          borderRadius: 4,
          touchAction: 'none',
          width: '100%',
        }}
        viewBox={`0 0 ${CURVE_SIZE} ${CURVE_SIZE}`}
      >
        <rect
          fill="var(--chakra-colors-bg-inset)"
          height={CURVE_SIZE - CURVE_PADDING * 2}
          width={CURVE_SIZE - CURVE_PADDING * 2}
          x={CURVE_PADDING}
          y={CURVE_PADDING}
        />
        <g stroke="var(--chakra-colors-fg-grid)">
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
        </g>
        <rect
          fill="none"
          height={CURVE_SIZE - CURVE_PADDING * 2}
          stroke="var(--chakra-colors-border-emphasized)"
          vectorEffect="non-scaling-stroke"
          width={CURVE_SIZE - CURVE_PADDING * 2}
          x={CURVE_PADDING}
          y={CURVE_PADDING}
        />
        <line
          stroke="var(--chakra-colors-fg-muted)"
          strokeDasharray="4 4"
          vectorEffect="non-scaling-stroke"
          x1={CURVE_PADDING}
          x2={CURVE_SIZE - CURVE_PADDING}
          y1={CURVE_SIZE - CURVE_PADDING}
          y2={CURVE_PADDING}
        />
        <path
          d={lutPath}
          fill="none"
          stroke="var(--chakra-colors-accent-solid)"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          vectorEffect="non-scaling-stroke"
        />
        {points.map((p, i) => {
          const { cx, cy } = curvePointToSvg(p[0], p[1]);
          return (
            <circle
              cx={cx}
              cy={cy}
              fill="var(--chakra-colors-accent-solid)"
              key={i}
              onContextMenu={(e) => e.preventDefault()}
              onDoubleClick={handleRemove(i)}
              onPointerDown={handlePointDown(i)}
              r={5}
              stroke="var(--chakra-colors-bg-inset)"
              strokeWidth={2}
              style={{ cursor: 'pointer' }}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
      <Text color="fg.muted" fontSize="2xs">
        {t('widgets.layers.adjustments.curvesHint')}
      </Text>
    </Stack>
  );
};

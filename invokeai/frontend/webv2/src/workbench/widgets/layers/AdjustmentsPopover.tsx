import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasEngine } from '@workbench/canvas-engine/engine';
import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/types';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { DEFAULT_ADJUSTMENTS, buildCurveLut } from '@workbench/canvas-engine/render/adjustments';
import { Button, Field, Select, Slider } from '@workbench/components/ui';
import { useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useCallback, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { applyStructural } from './layerOps';

const SELECT_POSITIONING = { placement: 'bottom-end', sameWidth: false } as const;

type CurveChannel = 'r' | 'g' | 'b';
const CURVE_CHANNELS: readonly CurveChannel[] = ['r', 'g', 'b'];

/** The identity curve control points (diagonal). */
const IDENTITY_CURVE: [number, number][] = [
  [0, 0],
  [255, 255],
];

const formatSigned = (value: number): string => `${value > 0 ? '+' : ''}${Math.round(value * 100)}`;

interface AdjustmentsPopoverProps {
  engine: CanvasEngine | null;
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
  engine: CanvasEngine | null;
  layer: CanvasRasterLayerContractV2;
}

type ScalarKey = 'brightness' | 'contrast' | 'saturation';

const AdjustmentsControls = ({ adjustments, engine, layer }: AdjustmentsControlsProps) => {
  const { t } = useTranslation();
  const dispatch = useWorkbenchDispatch();

  const patchLive = useCallback(
    (next: CanvasAdjustmentsContract) => {
      dispatch({
        config: { adjustments: next, layerType: 'raster' },
        id: layer.id,
        type: 'updateCanvasLayerConfig',
      });
    },
    [dispatch, layer.id]
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

  const withCurve = useCallback(
    (
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
    }),
    []
  );

  // Live (render-only) during a curve-point drag: preview without pushing history.
  const handleCurveLive = useCallback(
    (channel: CurveChannel, points: [number, number][]) => {
      patchLive(withCurve(adjustments, channel, points));
    },
    [adjustments, patchLive, withCurve]
  );

  // Single history entry per gesture (drag end, click-add, dbl-click-remove). The
  // `before` snapshot is captured at gesture start by the editor (during a drag
  // `adjustments` has already advanced via the live previews), so it undoes the
  // WHOLE gesture rather than the last frame.
  const handleCurveCommit = useCallback(
    (channel: CurveChannel, points: [number, number][], before: CanvasAdjustmentsContract) => {
      commit(t('widgets.layers.adjustments.curves'), withCurve(before, channel, points), before);
    },
    [commit, t, withCurve]
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
      <CurvesEditor adjustments={adjustments} onCommit={handleCurveCommit} onLive={handleCurveLive} />
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

const CURVE_SIZE = 180;

interface CurvesEditorProps {
  adjustments: CanvasAdjustmentsContract;
  /** Render-only preview during a point drag (no history entry). */
  onLive: (channel: CurveChannel, points: [number, number][]) => void;
  /** Commits one history entry for a completed gesture, undoing to `before`. */
  onCommit: (channel: CurveChannel, points: [number, number][], before: CanvasAdjustmentsContract) => void;
}

/** A compact per-channel curves editor (SVG): drag points, click to add, double-click to remove. */
const CurvesEditor = ({ adjustments, onCommit, onLive }: CurvesEditorProps) => {
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

  const toSvg = (x: number, y: number): { cx: number; cy: number } => ({
    cx: (x / 255) * CURVE_SIZE,
    cy: CURVE_SIZE - (y / 255) * CURVE_SIZE,
  });

  const fromSvg = (px: number, py: number): [number, number] => {
    const x = Math.max(0, Math.min(255, Math.round((px / CURVE_SIZE) * 255)));
    const y = Math.max(0, Math.min(255, Math.round(((CURVE_SIZE - py) / CURVE_SIZE) * 255)));
    return [x, y];
  };

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
      const cx = (i / 255) * CURVE_SIZE;
      const cy = CURVE_SIZE - (lut[i] / 255) * CURVE_SIZE;
      d += `${i === 0 ? 'M' : 'L'}${cx.toFixed(1)},${cy.toFixed(1)} `;
    }
    return d.trim();
  }, [points]);

  const handleChannelChange = useCallback(
    ({ value }: { value: string[] }) => setChannel((value[0] as CurveChannel) ?? 'r'),
    []
  );

  const handlePointDown = (index: number) => (event: ReactPointerEvent<SVGCircleElement>) => {
    event.stopPropagation();
    (event.target as Element).setPointerCapture?.(event.pointerId);
    dragIndexRef.current = index;
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
    const [nx, ny] = fromSvg(px, py);
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

  const handleUp = (event: ReactPointerEvent<SVGSVGElement>) => {
    const wasDragging = dragIndexRef.current !== null;
    if (wasDragging) {
      (event.target as Element).releasePointerCapture?.(event.pointerId);
    }
    dragIndexRef.current = null;
    // Commit the whole drag as one history entry (only if the point actually
    // moved — a click with no move streams no previews and needs no commit).
    const before = beforeRef.current;
    const finalPoints = latestPointsRef.current;
    beforeRef.current = null;
    latestPointsRef.current = null;
    if (wasDragging && before && finalPoints) {
      onCommit(channel, finalPoints, before);
    }
  };

  const handleAdd = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (dragIndexRef.current !== null) {
      return;
    }
    const { px, py } = svgPointFromEvent(event);
    const [nx, ny] = fromSvg(px, py);
    if (nx <= 0 || nx >= 255) {
      return;
    }
    const next = [...points, [nx, ny] as [number, number]].sort((a, b) => a[0] - b[0]);
    onCommit(channel, next, adjustments);
  };

  const handleRemove = (index: number) => (event: ReactPointerEvent<SVGCircleElement>) => {
    event.stopPropagation();
    if (index === 0 || index === points.length - 1 || points.length <= 2) {
      return;
    }
    onCommit(
      channel,
      points.filter((_, i) => i !== index),
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
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        ref={svgRef}
        style={{ background: 'var(--chakra-colors-bg-inset)', borderRadius: 4, touchAction: 'none', width: '100%' }}
        viewBox={`0 0 ${CURVE_SIZE} ${CURVE_SIZE}`}
      >
        <line stroke="var(--chakra-colors-border-subtle)" x1={0} x2={CURVE_SIZE} y1={CURVE_SIZE} y2={0} />
        <path d={lutPath} fill="none" stroke="var(--chakra-colors-fg-emphasized)" strokeWidth={1.5} />
        {points.map((p, i) => {
          const { cx, cy } = toSvg(p[0], p[1]);
          return (
            <circle
              cx={cx}
              cy={cy}
              fill="var(--chakra-colors-fg-emphasized)"
              key={i}
              onContextMenu={(e) => e.preventDefault()}
              onDoubleClick={handleRemove(i)}
              onPointerDown={handlePointDown(i)}
              r={4}
              style={{ cursor: 'pointer' }}
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

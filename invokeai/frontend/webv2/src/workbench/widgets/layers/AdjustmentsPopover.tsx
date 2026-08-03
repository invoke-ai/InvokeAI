import type { SliderValueChangeDetails } from '@chakra-ui/react';
import type { CanvasAdjustmentsContract, CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/api';
import type { CanvasStructuralEngine } from '@workbench/widgets/layers/layerOps';
import type { PointerEvent as ReactPointerEvent } from 'react';

import { chakra, createListCollection, HStack, Stack, Text } from '@chakra-ui/react';
import { Button, Field, Select, Slider } from '@platform/ui';
import { DEFAULT_ADJUSTMENTS, buildCurveLut } from '@workbench/canvas-engine/api';
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

  const handleCurveLive = useCallback(
    (channel: CurveChannel, points: [number, number][]) => {
      patchLive(withCurve(adjustments, channel, points));
    },
    [adjustments, patchLive]
  );

  const handleCurveCancel = useCallback((before: CanvasAdjustmentsContract) => patchLive(before), [patchLive]);

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
  onLive: (channel: CurveChannel, points: [number, number][]) => void;
  onCancel: (before: CanvasAdjustmentsContract) => void;
  onCommit: (current: CanvasAdjustmentsContract, before: CanvasAdjustmentsContract) => void;
}

const CurvesEditor = ({ adjustments, onCancel, onCommit, onLive }: CurvesEditorProps) => {
  const { t } = useTranslation();
  const [channel, setChannel] = useState<CurveChannel>('r');
  const dragIndexRef = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
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
      beforeRef.current = adjustments;
      latestPointsRef.current = null;
    },
    [adjustments]
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
      onLive(channel, next);
    },
    [channel, onLive, points, svgPointFromEvent]
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
        finishCurveDragResult({
          before,
          cancelled,
          current: withCurve(before, channel, finalPoints),
          onCommit: (current) => onCommit(current, before),
          onPreview: onCancel,
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
      onCommit(withCurve(adjustments, channel, next), adjustments);
    },
    [adjustments, channel, onCommit, points, svgPointFromEvent]
  );

  const handleRemove = useCallback(
    (event: ReactPointerEvent<SVGCircleElement>) => {
      event.stopPropagation();
      const index = Number(event.currentTarget.dataset.index);
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
    },
    [adjustments, channel, onCommit, points]
  );

  const channelValue = useMemo(() => [channel], [channel]);

  return (
    <Stack css={CURVE_EDITOR_CSS} gap="2">
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
      <Text color="fg.muted" fontSize="2xs">
        {t('widgets.layers.adjustments.curvesHint')}
      </Text>
    </Stack>
  );
};

import { Box, Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterContext } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsCurvesUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntity } from 'features/controlLayers/store/selectors';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

const DEFAULT_POINTS: Array<[number, number]> = [
  [0, 0],
  [255, 255],
];

type Channel = 'master' | 'r' | 'g' | 'b';

const channelColor: Record<Channel, string> = {
  master: '#888',
  r: '#e53e3e',
  g: '#38a169',
  b: '#3182ce',
};

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

const sortPoints = (pts: Array<[number, number]>) =>
  [...pts]
    .sort((a, b) => a[0] - b[0])
    .map(([x, y]) => [clamp(Math.round(x), 0, 255), clamp(Math.round(y), 0, 255)] as [number, number]);

// Base canvas logical coordinate system (used for aspect ratio & initial sizing)
const CANVAS_WIDTH = 256;
const CANVAS_HEIGHT = 160;
const MARGIN_LEFT = 8;
const MARGIN_RIGHT = 8;
const MARGIN_TOP = 8;
const MARGIN_BOTTOM = 10;

const CANVAS_STYLE: React.CSSProperties = {
  width: '100%',
  // Maintain aspect ratio while allowing responsive width. Height is set automatically via aspect-ratio.
  aspectRatio: `${CANVAS_WIDTH} / ${CANVAS_HEIGHT}`,
  height: 'auto',
  touchAction: 'none',
  borderRadius: 4,
  background: '#111',
  display: 'block',
};

type CurveGraphProps = {
  title: string;
  channel: Channel;
  points: Array<[number, number]> | undefined;
  histogram: number[] | null;
  onChange: (pts: Array<[number, number]>) => void;
};

const CurveGraph = memo(function CurveGraph(props: CurveGraphProps) {
  const { title, channel, points, histogram, onChange } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [localPoints, setLocalPoints] = useState<Array<[number, number]>>(sortPoints(points ?? DEFAULT_POINTS));
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  useEffect(() => {
    setLocalPoints(sortPoints(points ?? DEFAULT_POINTS));
  }, [points]);

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }

    // Use device pixel ratio for crisp rendering on HiDPI displays.
    const dpr = window.devicePixelRatio || 1;
    const cssWidth = c.clientWidth || CANVAS_WIDTH; // CSS pixels
    const cssHeight = (cssWidth * CANVAS_HEIGHT) / CANVAS_WIDTH; // maintain aspect ratio

    // Ensure the backing store matches current display size * dpr (only if changed).
    const targetWidth = Math.round(cssWidth * dpr);
    const targetHeight = Math.round(cssHeight * dpr);
    if (c.width !== targetWidth || c.height !== targetHeight) {
      c.width = targetWidth;
      c.height = targetHeight;
    }
    // Guarantee the CSS height stays synced (width is 100%).
    if (c.style.height !== `${cssHeight}px`) {
      c.style.height = `${cssHeight}px`;
    }

    const ctx = c.getContext('2d');
    if (!ctx) {
      return;
    }

    // Reset transform then scale for dpr so we can draw in CSS pixel coordinates.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);

    // Dynamic inner geometry (CSS pixel space)
    const innerWidth = cssWidth - MARGIN_LEFT - MARGIN_RIGHT;
    const innerHeight = cssHeight - MARGIN_TOP - MARGIN_BOTTOM;

    const valueToCanvasX = (x: number) => MARGIN_LEFT + (clamp(x, 0, 255) / 255) * innerWidth;
    const valueToCanvasY = (y: number) => MARGIN_TOP + innerHeight - (clamp(y, 0, 255) / 255) * innerHeight;

    // Clear & background
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    // Grid
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = MARGIN_TOP + (i * innerHeight) / 4;
      ctx.beginPath();
      ctx.moveTo(MARGIN_LEFT + 0.5, y + 0.5);
      ctx.lineTo(MARGIN_LEFT + innerWidth - 0.5, y + 0.5);
      ctx.stroke();
    }
    for (let i = 0; i <= 4; i++) {
      const x = MARGIN_LEFT + (i * innerWidth) / 4;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, MARGIN_TOP + 0.5);
      ctx.lineTo(x + 0.5, MARGIN_TOP + innerHeight - 0.5);
      ctx.stroke();
    }

    // Histogram
    if (histogram) {
      const logHist = histogram.map((v) => Math.log10((v ?? 0) + 1));
      const max = Math.max(1e-6, ...logHist);
      ctx.fillStyle = '#5557';

      // If there's enough horizontal room, draw each of the 256 bins with exact (possibly fractional) width so they tessellate.
      // Otherwise, aggregate multiple bins into per-pixel columns to avoid aliasing.
      if (innerWidth >= 256) {
        for (let i = 0; i < 256; i++) {
          const v = logHist[i] ?? 0;
          const h = (v / max) * (innerHeight - 2);
          // Exact fractional coordinates for seamless coverage (no gaps as width grows)
          const x0 = MARGIN_LEFT + (i / 256) * innerWidth;
          const x1 = MARGIN_LEFT + ((i + 1) / 256) * innerWidth;
          const w = x1 - x0;
          if (w <= 0) {
            continue;
          } // safety
          const y = MARGIN_TOP + innerHeight - h;
          ctx.fillRect(x0, y, w, h);
        }
      } else {
        // Aggregate bins per CSS pixel column (similar to previous anti-moire approach)
        const columns = Math.max(1, Math.round(innerWidth));
        const binsPerCol = 256 / columns;
        for (let col = 0; col < columns; col++) {
          const startBin = Math.floor(col * binsPerCol);
          const endBin = Math.min(255, Math.floor((col + 1) * binsPerCol - 1));
          let acc = 0;
          let count = 0;
          for (let b = startBin; b <= endBin; b++) {
            acc += logHist[b] ?? 0;
            count++;
          }
          const v = count > 0 ? acc / count : 0;
          const h = (v / max) * (innerHeight - 2);
          const x = MARGIN_LEFT + col;
          const y = MARGIN_TOP + innerHeight - h;
          ctx.fillRect(x, y, 1, h);
        }
      }
    }

    // Curve
    const pts = sortPoints(localPoints);
    ctx.strokeStyle = channelColor[channel];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i]!;
      const cx = valueToCanvasX(x);
      const cy = valueToCanvasY(y);
      if (i === 0) {
        ctx.moveTo(cx, cy);
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();

    // Control points
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i]!;
      const cx = valueToCanvasX(x);
      const cy = valueToCanvasY(y);
      ctx.fillStyle = '#000';
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = channelColor[channel];
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [channel, histogram, localPoints]);

  useEffect(() => {
    draw();
  }, [draw]);

  const getNearestPointIndex = useCallback(
    (mx: number, my: number) => {
      const c = canvasRef.current;
      if (!c) {
        return -1;
      }
      const cssWidth = c.clientWidth || CANVAS_WIDTH;
      const cssHeight = c.clientHeight || CANVAS_HEIGHT;
      const innerWidth = cssWidth - MARGIN_LEFT - MARGIN_RIGHT;
      const innerHeight = cssHeight - MARGIN_TOP - MARGIN_BOTTOM;
      const canvasToValueX = (cx: number) => clamp(Math.round(((cx - MARGIN_LEFT) / innerWidth) * 255), 0, 255);
      const canvasToValueY = (cy: number) => clamp(Math.round(255 - ((cy - MARGIN_TOP) / innerHeight) * 255), 0, 255);
      const xVal = canvasToValueX(mx);
      const yVal = canvasToValueY(my);
      let best = -1;
      let bestDist = 9999;
      for (let i = 0; i < localPoints.length; i++) {
        const [px, py] = localPoints[i]!;
        const dx = px - xVal;
        const dy = py - yVal;
        const d = dx * dx + dy * dy;
        if (d < bestDist) {
          best = i;
          bestDist = d;
        }
      }
      if (best !== -1 && bestDist <= 20 * 20) {
        return best;
      }
      return -1;
    },
    [localPoints]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const c = canvasRef.current;
      if (!c) {
        return;
      }
      // Capture the pointer so we still get pointerup even if released outside the canvas.
      try {
        c.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left; // CSS pixel coordinates
      const my = e.clientY - rect.top;
      const cssWidth = c.clientWidth || CANVAS_WIDTH;
      const cssHeight = c.clientHeight || CANVAS_HEIGHT;
      const innerWidth = cssWidth - MARGIN_LEFT - MARGIN_RIGHT;
      const innerHeight = cssHeight - MARGIN_TOP - MARGIN_BOTTOM;
      const canvasToValueX = (cx: number) => clamp(Math.round(((cx - MARGIN_LEFT) / innerWidth) * 255), 0, 255);
      const canvasToValueY = (cy: number) => clamp(Math.round(255 - ((cy - MARGIN_TOP) / innerHeight) * 255), 0, 255);
      const idx = getNearestPointIndex(mx, my);
      if (idx !== -1 && idx !== 0 && idx !== localPoints.length - 1) {
        setDragIndex(idx);
        return;
      }
      const xVal = canvasToValueX(mx);
      const yVal = canvasToValueY(my);
      const next = sortPoints([...localPoints, [xVal, yVal]]);
      setLocalPoints(next);
      setDragIndex(next.findIndex(([x, y]) => x === xVal && y === yVal));
    },
    [getNearestPointIndex, localPoints]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      if (dragIndex === null) {
        return;
      }
      const c = canvasRef.current;
      if (!c) {
        return;
      }
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const cssWidth = c.clientWidth || CANVAS_WIDTH;
      const cssHeight = c.clientHeight || CANVAS_HEIGHT;
      const innerWidth = cssWidth - MARGIN_LEFT - MARGIN_RIGHT;
      const innerHeight = cssHeight - MARGIN_TOP - MARGIN_BOTTOM;
      const canvasToValueX = (cx: number) => clamp(Math.round(((cx - MARGIN_LEFT) / innerWidth) * 255), 0, 255);
      const canvasToValueY = (cy: number) => clamp(Math.round(255 - ((cy - MARGIN_TOP) / innerHeight) * 255), 0, 255);
      const mxVal = canvasToValueX(mx);
      const myVal = canvasToValueY(my);
      setLocalPoints((prev) => {
        // Endpoints are immutable; safety check.
        if (dragIndex === 0 || dragIndex === prev.length - 1) {
          return prev;
        }
        const leftX = prev[dragIndex - 1]![0];
        const rightX = prev[dragIndex + 1]![0];
        // Constrain to strictly between neighbors so ordering is preserved & no crossing.
        const minX = Math.min(254, leftX);
        const maxX = Math.max(1, rightX);
        const clampedX = clamp(mxVal, minX, maxX);
        // If neighbors are adjacent (minX > maxX after adjustments), effectively lock X.
        const finalX = minX > maxX ? leftX + 1 - 1 /* keep existing */ : clampedX;
        const next = [...prev];
        next[dragIndex] = [finalX, myVal];
        return next; // already ordered due to constraints
      });
    },
    [dragIndex]
  );

  const commit = useCallback(
    (pts: Array<[number, number]>) => {
      onChange(sortPoints(pts));
    },
    [onChange]
  );

  const handlePointerUp = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const c = canvasRef.current;
      if (c) {
        try {
          c.releasePointerCapture(e.pointerId);
        } catch {
          /* ignore */
        }
      }
      setDragIndex(null);
      commit(localPoints);
    },
    [commit, localPoints]
  );

  const handlePointerCancel = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const c = canvasRef.current;
      if (c) {
        try {
          c.releasePointerCapture(e.pointerId);
        } catch {
          /* ignore */
        }
      }
      setDragIndex(null);
      commit(localPoints);
    },
    [commit, localPoints]
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const c = canvasRef.current;
      if (!c) {
        return;
      }
      const rect = c.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const idx = getNearestPointIndex(mx, my);
      if (idx > 0 && idx < localPoints.length - 1) {
        const next = localPoints.filter((_, i) => i !== idx);
        setLocalPoints(next);
        commit(next);
      }
    },
    [commit, getNearestPointIndex, localPoints]
  );

  // Observe size changes to redraw (responsive behavior)
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }
    const ro = new ResizeObserver(() => {
      draw();
    });
    ro.observe(c);
    return () => ro.disconnect();
  }, [draw]);

  const resetPoints = useCallback(() => {
    setLocalPoints(sortPoints(DEFAULT_POINTS));
    commit(DEFAULT_POINTS);
  }, [commit]);

  return (
    <Flex flexDir="column" gap={2}>
      <Flex justifyContent="space-between">
        <Text fontSize="sm" color={channelColor[channel]} fontWeight="semibold">
          {title}
        </Text>
        <IconButton
          icon={<PiArrowCounterClockwiseBold />}
          aria-label="Reset"
          size="sm"
          variant="link"
          onClick={resetPoints}
        />
      </Flex>
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
        onDoubleClick={handleDoubleClick}
        style={CANVAS_STYLE}
      />
    </Flex>
  );
});

export const RasterLayerCurvesAdjustmentsEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const adapter = useEntityAdapterContext<'raster_layer'>('raster_layer');
  const { t } = useTranslation();
  const selectLayer = useMemo(
    () => createSelector(selectCanvasSlice, (canvas) => selectEntity(canvas, entityIdentifier)),
    [entityIdentifier]
  );
  const layer = useAppSelector(selectLayer);
  const selectIsDisabled = useMemo(() => {
    return createSelector(
      selectCanvasSlice,
      (canvas) => selectEntity(canvas, entityIdentifier)?.adjustments?.enabled !== true
    );
  }, [entityIdentifier]);
  const isDisabled = useAppSelector(selectIsDisabled);

  const [histMaster, setHistMaster] = useState<number[] | null>(null);
  const [histR, setHistR] = useState<number[] | null>(null);
  const [histG, setHistG] = useState<number[] | null>(null);
  const [histB, setHistB] = useState<number[] | null>(null);

  const pointsMaster = layer?.adjustments?.curves.master ?? DEFAULT_POINTS;
  const pointsR = layer?.adjustments?.curves.r ?? DEFAULT_POINTS;
  const pointsG = layer?.adjustments?.curves.g ?? DEFAULT_POINTS;
  const pointsB = layer?.adjustments?.curves.b ?? DEFAULT_POINTS;

  const recalcHistogram = useCallback(() => {
    try {
      const rect = adapter.transformer.getRelativeRect();
      if (rect.width === 0 || rect.height === 0) {
        setHistMaster(Array(256).fill(0));
        setHistR(Array(256).fill(0));
        setHistG(Array(256).fill(0));
        setHistB(Array(256).fill(0));
        return;
      }
      const imageData = adapter.renderer.getImageData({ rect });
      const data = imageData.data;
      const len = data.length / 4;
      const master = new Array<number>(256).fill(0);
      const r = new Array<number>(256).fill(0);
      const g = new Array<number>(256).fill(0);
      const b = new Array<number>(256).fill(0);
      // sample every 4th pixel to lighten work
      for (let i = 0; i < len; i += 4) {
        const idx = i * 4;
        const rv = data[idx] as number;
        const gv = data[idx + 1] as number;
        const bv = data[idx + 2] as number;
        const m = Math.round(0.2126 * rv + 0.7152 * gv + 0.0722 * bv);
        if (m >= 0 && m < 256) {
          master[m] = (master[m] ?? 0) + 1;
        }
        if (rv >= 0 && rv < 256) {
          r[rv] = (r[rv] ?? 0) + 1;
        }
        if (gv >= 0 && gv < 256) {
          g[gv] = (g[gv] ?? 0) + 1;
        }
        if (bv >= 0 && bv < 256) {
          b[bv] = (b[bv] ?? 0) + 1;
        }
      }
      setHistMaster(master);
      setHistR(r);
      setHistG(g);
      setHistB(b);
    } catch {
      // ignore
    }
  }, [adapter]);

  useEffect(() => {
    recalcHistogram();
  }, [layer?.objects, layer?.adjustments, recalcHistogram]);

  const onChangePoints = useCallback(
    (channel: Channel, pts: Array<[number, number]>) => {
      dispatch(rasterLayerAdjustmentsCurvesUpdated({ entityIdentifier, channel, points: pts }));
    },
    [dispatch, entityIdentifier]
  );

  // Memoize per-channel change handlers to avoid inline lambdas in JSX
  const onChangeMaster = useCallback((pts: Array<[number, number]>) => onChangePoints('master', pts), [onChangePoints]);
  const onChangeR = useCallback((pts: Array<[number, number]>) => onChangePoints('r', pts), [onChangePoints]);
  const onChangeG = useCallback((pts: Array<[number, number]>) => onChangePoints('g', pts), [onChangePoints]);
  const onChangeB = useCallback((pts: Array<[number, number]>) => onChangePoints('b', pts), [onChangePoints]);

  return (
    <Flex
      direction="column"
      gap={2}
      px={3}
      pb={3}
      opacity={isDisabled ? 0.3 : 1}
      pointerEvents={isDisabled ? 'none' : 'auto'}
    >
      <Box display="grid" gridTemplateColumns="repeat(2, minmax(0, 1fr))" gap={4}>
        <CurveGraph
          title={t('controlLayers.adjustments.master')}
          channel="master"
          points={pointsMaster}
          histogram={histMaster}
          onChange={onChangeMaster}
        />
        <CurveGraph title={t('common.red')} channel="r" points={pointsR} histogram={histR} onChange={onChangeR} />
        <CurveGraph title={t('common.green')} channel="g" points={pointsG} histogram={histG} onChange={onChangeG} />
        <CurveGraph title={t('common.blue')} channel="b" points={pointsB} histogram={histB} onChange={onChangeB} />
      </Box>
    </Flex>
  );
});

RasterLayerCurvesAdjustmentsEditor.displayName = 'RasterLayerCurvesEditor';

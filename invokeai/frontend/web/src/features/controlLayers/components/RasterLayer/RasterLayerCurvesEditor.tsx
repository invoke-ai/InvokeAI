import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterContext } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsCurvesUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';

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

  const width = 256;
  const height = 160;

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }
    c.width = width;
    c.height = height;
    const ctx = c.getContext('2d');
    if (!ctx) {
      return;
    }

    // background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, width, height);

    // grid
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = (i * height) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.stroke();
    }
    for (let i = 0; i <= 4; i++) {
      const x = (i * width) / 4;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, height);
      ctx.stroke();
    }

    // histogram
    if (histogram) {
      const max = Math.max(1, ...histogram);
      ctx.fillStyle = '#5557';
      for (let x = 0; x < 256; x++) {
        const v = histogram[x] ?? 0;
        const h = Math.round((v / max) * (height - 4));
        ctx.fillRect(x, height - h, 1, h);
      }
    }

    // curve
    const pts = sortPoints(localPoints);
    ctx.strokeStyle = channelColor[channel];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i]!;
      const cx = x;
      const cy = height - (y / 255) * height;
      if (i === 0) {
        ctx.moveTo(cx, cy);
      } else {
        ctx.lineTo(cx, cy);
      }
    }
    ctx.stroke();

    // control points
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i]!;
      const cx = x;
      const cy = height - (y / 255) * height;
      ctx.fillStyle = '#000';
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = channelColor[channel];
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // title
    ctx.fillStyle = '#bbb';
    ctx.font = '12px sans-serif';
    ctx.fillText(title, 6, 14);
  }, [channel, height, histogram, localPoints, title, width]);

  useEffect(() => {
    draw();
  }, [draw]);

  const getNearestPointIndex = useCallback(
    (mx: number, my: number) => {
      // map canvas y to [0..255]
      const yVal = clamp(Math.round(255 - (my / height) * 255), 0, 255);
      const xVal = clamp(Math.round(mx), 0, 255);
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
    [height, localPoints]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const idx = getNearestPointIndex(mx, my);
      if (idx !== -1 && idx !== 0 && idx !== localPoints.length - 1) {
        setDragIndex(idx);
        return;
      }
      // add new point
      const xVal = clamp(Math.round(mx), 0, 255);
      const yVal = clamp(Math.round(255 - (my / height) * 255), 0, 255);
      const next = sortPoints([...localPoints, [xVal, yVal]]);
      setLocalPoints(next);
      setDragIndex(next.findIndex(([x, y]) => x === xVal && y === yVal));
    },
    [getNearestPointIndex, height, localPoints]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      if (dragIndex === null) {
        return;
      }
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const mx = clamp(Math.round(e.clientX - rect.left), 0, 255);
      const myPx = clamp(Math.round(255 - ((e.clientY - rect.top) / height) * 255), 0, 255);
      setLocalPoints((prev) => {
        const next = [...prev];
        // clamp endpoints to ends and keep them immutable
        if (dragIndex === 0) {
          return prev;
        }
        if (dragIndex === prev.length - 1) {
          return prev;
        }
        next[dragIndex] = [mx, myPx];
        return sortPoints(next);
      });
    },
    [dragIndex, height]
  );

  const commit = useCallback(
    (pts: Array<[number, number]>) => {
      onChange(sortPoints(pts));
    },
    [onChange]
  );

  const handlePointerUp = useCallback(() => {
    setDragIndex(null);
    commit(localPoints);
  }, [commit, localPoints]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
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

  const canvasStyle = useMemo<React.CSSProperties>(
    () => ({ width: '100%', height: height, touchAction: 'none', borderRadius: 4, background: '#111' }),
    [height]
  );

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onDoubleClick={handleDoubleClick}
      style={canvasStyle}
    />
  );
});

export const RasterLayerCurvesEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const adapter = useEntityAdapterContext<'raster_layer'>('raster_layer');
  const layer = useAppSelector((s) => selectEntity(s.canvas.present, entityIdentifier)) as
    | CanvasRasterLayerState
    | undefined;

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layer?.objects, layer?.adjustments]);

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

  const gridStyles: React.CSSProperties = useMemo(
    () => ({ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 8 }),
    []
  );

  return (
    <Flex direction="column" gap={2}>
      <Text fontSize="sm" color="base.300">
        Curves
      </Text>
      <div style={gridStyles}>
        <CurveGraph
          title="Master"
          channel="master"
          points={pointsMaster}
          histogram={histMaster}
          onChange={onChangeMaster}
        />
        <CurveGraph title="Red" channel="r" points={pointsR} histogram={histR} onChange={onChangeR} />
        <CurveGraph title="Green" channel="g" points={pointsG} histogram={histG} onChange={onChangeG} />
        <CurveGraph title="Blue" channel="b" points={pointsB} histogram={histB} onChange={onChangeB} />
      </div>
    </Flex>
  );
});

RasterLayerCurvesEditor.displayName = 'RasterLayerCurvesEditor';

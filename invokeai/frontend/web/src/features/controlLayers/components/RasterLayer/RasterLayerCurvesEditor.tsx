import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityAdapterContext } from 'features/controlLayers/contexts/EntityAdapterContext';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rasterLayerAdjustmentsCurvesUpdated } from 'features/controlLayers/store/canvasSlice';
import { selectEntity } from 'features/controlLayers/store/selectors';
import type { CanvasRasterLayerState } from 'features/controlLayers/store/types';
import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

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

// Extracted canvas constants and helpers out of the component
const CANVAS_WIDTH = 256;
const CANVAS_HEIGHT = 160;
const MARGIN_LEFT = 8;
const MARGIN_RIGHT = 8;
const MARGIN_TOP = 8;
const MARGIN_BOTTOM = 10;
const INNER_WIDTH = CANVAS_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
const INNER_HEIGHT = CANVAS_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

const valueToCanvasX = (x: number) => MARGIN_LEFT + (clamp(x, 0, 255) / 255) * INNER_WIDTH;
const valueToCanvasY = (y: number) => MARGIN_TOP + INNER_HEIGHT - (clamp(y, 0, 255) / 255) * INNER_HEIGHT;
const canvasToValueX = (cx: number) => clamp(Math.round(((cx - MARGIN_LEFT) / INNER_WIDTH) * 255), 0, 255);
const canvasToValueY = (cy: number) => clamp(Math.round(255 - ((cy - MARGIN_TOP) / INNER_HEIGHT) * 255), 0, 255);

// Optional: stable canvas style from constants
const CANVAS_STYLE: React.CSSProperties = {
  width: '100%',
  height: CANVAS_HEIGHT,
  touchAction: 'none',
  borderRadius: 4,
  background: '#111',
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
    c.width = CANVAS_WIDTH;
    c.height = CANVAS_HEIGHT;
    const ctx = c.getContext('2d');
    if (!ctx) {
      return;
    }

    // background
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    // grid inside inner rect
    ctx.strokeStyle = '#2a2a2a';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = MARGIN_TOP + (i * INNER_HEIGHT) / 4;
      ctx.beginPath();
      ctx.moveTo(MARGIN_LEFT + 0.5, y + 0.5);
      ctx.lineTo(MARGIN_LEFT + INNER_WIDTH - 0.5, y + 0.5);
      ctx.stroke();
    }
    for (let i = 0; i <= 4; i++) {
      const x = MARGIN_LEFT + (i * INNER_WIDTH) / 4;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, MARGIN_TOP + 0.5);
      ctx.lineTo(x + 0.5, MARGIN_TOP + INNER_HEIGHT - 0.5);
      ctx.stroke();
    }

    // histogram
    if (histogram) {
      // logarithmic histogram for readability when values vary widely
      const logHist = histogram.map((v) => Math.log10((v ?? 0) + 1));
      const max = Math.max(1e-6, ...logHist);
      ctx.fillStyle = '#5557';
      const binW = Math.max(1, INNER_WIDTH / 256);
      for (let i = 0; i < 256; i++) {
        const v = logHist[i] ?? 0;
        const h = Math.round((v / max) * (INNER_HEIGHT - 2));
        const x = MARGIN_LEFT + Math.floor(i * binW);
        const y = MARGIN_TOP + INNER_HEIGHT - h;
        ctx.fillRect(x, y, Math.ceil(binW), h);
      }
    }

    // curve
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

    // control points
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
    (mxCanvas: number, myCanvas: number) => {
      // convert canvas px to value-space [0..255]
      const xVal = canvasToValueX(mxCanvas);
      const yVal = canvasToValueY(myCanvas);
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
      const rect = c.getBoundingClientRect();
      const scaleX = c.width / rect.width;
      const scaleY = c.height / rect.height;
      const mxCanvas = (e.clientX - rect.left) * scaleX;
      const myCanvas = (e.clientY - rect.top) * scaleY;
      const idx = getNearestPointIndex(mxCanvas, myCanvas);
      if (idx !== -1 && idx !== 0 && idx !== localPoints.length - 1) {
        setDragIndex(idx);
        return;
      }
      // add new point
      const xVal = canvasToValueX(mxCanvas);
      const yVal = canvasToValueY(myCanvas);
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
      const scaleX = c.width / rect.width;
      const scaleY = c.height / rect.height;
      const mxCanvas = (e.clientX - rect.left) * scaleX;
      const myCanvas = (e.clientY - rect.top) * scaleY;
      const mxVal = canvasToValueX(mxCanvas);
      const myVal = canvasToValueY(myCanvas);
      setLocalPoints((prev) => {
        const next = [...prev];
        // clamp endpoints to ends and keep them immutable
        if (dragIndex === 0) {
          return prev;
        }
        if (dragIndex === prev.length - 1) {
          return prev;
        }
        next[dragIndex] = [mxVal, myVal];
        return sortPoints(next);
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

  const handlePointerUp = useCallback(() => {
    setDragIndex(null);
    commit(localPoints);
  }, [commit, localPoints]);

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      e.stopPropagation();
      const c = canvasRef.current;
      if (!c) {
        return;
      }
      const rect = c.getBoundingClientRect();
      const scaleX = c.width / rect.width;
      const scaleY = c.height / rect.height;
      const mxCanvas = (e.clientX - rect.left) * scaleX;
      const myCanvas = (e.clientY - rect.top) * scaleY;
      const idx = getNearestPointIndex(mxCanvas, myCanvas);
      if (idx > 0 && idx < localPoints.length - 1) {
        const next = localPoints.filter((_, i) => i !== idx);
        setLocalPoints(next);
        commit(next);
      }
    },
    [commit, getNearestPointIndex, localPoints]
  );

  const canvasStyle = useMemo<React.CSSProperties>(() => CANVAS_STYLE, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <Text fontSize="xs" color={channelColor[channel]}>
        {title}
      </Text>
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={handleDoubleClick}
        style={canvasStyle}
      />
    </div>
  );
});

export const RasterLayerCurvesEditor = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext<'raster_layer'>();
  const adapter = useEntityAdapterContext<'raster_layer'>('raster_layer');
  const { t } = useTranslation();
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
      <div style={gridStyles}>
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
      </div>
    </Flex>
  );
});

RasterLayerCurvesEditor.displayName = 'RasterLayerCurvesEditor';

import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import type { ChannelName, ChannelPoints } from 'features/controlLayers/store/types';
import React, { memo, useCallback, useEffect, useRef, useState } from 'react';
import { PiArrowCounterClockwiseBold } from 'react-icons/pi';

const DEFAULT_POINTS: ChannelPoints = [
  [0, 0],
  [255, 255],
];

const channelColor: Record<ChannelName, string> = {
  master: '#888',
  r: '#e53e3e',
  g: '#38a169',
  b: '#3182ce',
};

const clamp = (v: number, min: number, max: number) => (v < min ? min : v > max ? max : v);

const sortPoints = (pts: ChannelPoints) =>
  [...pts]
    .sort((a, b) => {
      const xDiff = a[0] - b[0];
      if (xDiff) {
        return xDiff;
      }
      if (a[0] === 0 || a[0] === 255) {
        return a[1] - b[1];
      }
      return 0;
    })
    // Finally, clamp to valid range and round to integers
    .map(([x, y]) => [clamp(Math.round(x), 0, 255), clamp(Math.round(y), 0, 255)] satisfies [number, number]);

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
  channel: ChannelName;
  points: ChannelPoints | undefined;
  histogram: number[] | null;
  onChange: (pts: ChannelPoints) => void;
};

const drawHistogram = (
  c: HTMLCanvasElement,
  channel: ChannelName,
  histogram: number[] | null,
  points: ChannelPoints
) => {
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
  const pts = sortPoints(points);
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
};

const getNearestPointIndex = (c: HTMLCanvasElement, points: ChannelPoints, mx: number, my: number) => {
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
  for (let i = 0; i < points.length; i++) {
    const [px, py] = points[i]!;
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
};

const canvasXToValueX = (c: HTMLCanvasElement, cx: number): number => {
  const cssWidth = c.clientWidth || CANVAS_WIDTH;
  const innerWidth = cssWidth - MARGIN_LEFT - MARGIN_RIGHT;
  return clamp(Math.round(((cx - MARGIN_LEFT) / innerWidth) * 255), 0, 255);
};

const canvasYToValueY = (c: HTMLCanvasElement, cy: number) => {
  const cssHeight = c.clientHeight || CANVAS_HEIGHT;
  const innerHeight = cssHeight - MARGIN_TOP - MARGIN_BOTTOM;
  return clamp(Math.round(255 - ((cy - MARGIN_TOP) / innerHeight) * 255), 0, 255);
};

export const RasterLayerCurvesAdjustmentsGraph = memo((props: CurveGraphProps) => {
  const { title, channel, points, histogram, onChange } = props;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [localPoints, setLocalPoints] = useState<ChannelPoints>(sortPoints(points ?? DEFAULT_POINTS));
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  useEffect(() => {
    setLocalPoints(sortPoints(points ?? DEFAULT_POINTS));
  }, [points]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }
    drawHistogram(c, channel, histogram, localPoints);
  }, [channel, histogram, localPoints]);

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
      const idx = getNearestPointIndex(c, localPoints, mx, my);
      if (idx !== -1 && idx !== 0 && idx !== localPoints.length - 1) {
        setDragIndex(idx);
        return;
      }
      const xVal = canvasXToValueX(c, mx);
      const yVal = canvasYToValueY(c, my);
      const next = sortPoints([...localPoints, [xVal, yVal]]);
      setLocalPoints(next);
      setDragIndex(next.findIndex(([x, y]) => x === xVal && y === yVal));
    },
    [localPoints]
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
      const mxVal = canvasXToValueX(c, mx);
      const myVal = canvasYToValueY(c, my);
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
    (pts: ChannelPoints) => {
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
      const idx = getNearestPointIndex(c, localPoints, mx, my);
      if (idx > 0 && idx < localPoints.length - 1) {
        const next = localPoints.filter((_, i) => i !== idx);
        setLocalPoints(next);
        commit(next);
      }
    },
    [commit, localPoints]
  );

  // Observe size changes to redraw (responsive behavior)
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) {
      return;
    }
    const ro = new ResizeObserver(() => {
      drawHistogram(c, channel, histogram, localPoints);
    });
    ro.observe(c);
    return () => ro.disconnect();
  }, [channel, histogram, localPoints]);

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

RasterLayerCurvesAdjustmentsGraph.displayName = 'RasterLayerCurvesAdjustmentsGraph';

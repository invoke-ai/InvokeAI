/**
 * Rectilinear boundary tracing for alpha masks.
 *
 * Converts a mask's alpha channel into closed pixel-edge contours so the
 * marching ants can stroke the mask's true outline instead of its bounding
 * rect. Outer boundaries wind clockwise (screen coordinates) and holes wind
 * counter-clockwise, so the emitted path also nonzero-fills to the same shape
 * it outlines.
 */

interface MaskAlphaSource {
  /** RGBA pixel data (only the alpha channel is read). */
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

// Directions: 0 = right, 1 = down, 2 = left, 3 = up (screen coordinates).
const DIRECTION_X = [1, 0, -1, 0] as const;
const DIRECTION_Y = [0, 1, 0, -1] as const;

/**
 * Traces the outline of every alpha region as an SVG path in document space
 * (`origin` offsets the mask-local pixel grid). Pixels count as inside at
 * `alpha >= threshold`. Returns an empty string for an empty mask.
 */
export const traceMaskOutlinePath = (
  source: MaskAlphaSource,
  origin: { x: number; y: number },
  threshold = 128
): string => {
  const { data, height, width } = source;
  if (width <= 0 || height <= 0) {
    return '';
  }
  const inside = (x: number, y: number): boolean =>
    x >= 0 && y >= 0 && x < width && y < height && data[(y * width + x) * 4 + 3]! >= threshold;

  // Directed boundary edges bucketed by start vertex on the (width+1)×(height+1)
  // pixel-corner grid. Travelling each edge keeps the inside on the right, which
  // makes outer loops clockwise and holes counter-clockwise.
  const stride = width + 1;
  const outgoing = new Map<number, number[]>();
  const addEdge = (x: number, y: number, direction: number): void => {
    const key = y * stride + x;
    const bucket = outgoing.get(key);
    if (bucket) {
      bucket.push(direction);
    } else {
      outgoing.set(key, [direction]);
    }
  };
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!inside(x, y)) {
        continue;
      }
      if (!inside(x, y - 1)) {
        addEdge(x, y, 0);
      }
      if (!inside(x + 1, y)) {
        addEdge(x + 1, y, 1);
      }
      if (!inside(x, y + 1)) {
        addEdge(x + 1, y + 1, 2);
      }
      if (!inside(x - 1, y)) {
        addEdge(x, y + 1, 3);
      }
    }
  }
  if (outgoing.size === 0) {
    return '';
  }

  // At a pinch vertex (two regions touching diagonally) two edges leave the
  // same corner; preferring the tightest right turn keeps each loop around its
  // own region instead of crossing over to the neighbour.
  const takeEdge = (vertex: number, incoming: number): number | null => {
    const bucket = outgoing.get(vertex);
    if (!bucket || bucket.length === 0) {
      return null;
    }
    for (const direction of [(incoming + 1) % 4, incoming, (incoming + 3) % 4]) {
      const index = bucket.indexOf(direction);
      if (index !== -1) {
        bucket.splice(index, 1);
        if (bucket.length === 0) {
          outgoing.delete(vertex);
        }
        return direction;
      }
    }
    return null;
  };

  const segments: string[] = [];
  for (const [startVertex, bucket] of outgoing) {
    while (bucket.length > 0) {
      let direction = bucket[0]!;
      bucket.splice(0, 1);
      if (bucket.length === 0) {
        outgoing.delete(startVertex);
      }
      let vertex = startVertex;
      const startX = vertex % stride;
      const startY = (vertex - startX) / stride;
      const points: number[] = [startX, startY];
      let lastDirection = direction;
      for (;;) {
        vertex += DIRECTION_Y[direction]! * stride + DIRECTION_X[direction]!;
        const x = vertex % stride;
        const y = (vertex - x) / stride;
        if (direction === lastDirection && points.length >= 4) {
          points[points.length - 2] = x;
          points[points.length - 1] = y;
        } else {
          points.push(x, y);
        }
        lastDirection = direction;
        if (vertex === startVertex) {
          break;
        }
        const next = takeEdge(vertex, direction);
        if (next === null) {
          // Unreachable for well-formed edge sets; bail rather than loop forever.
          break;
        }
        direction = next;
      }
      // The walk re-appends the start vertex before closing; `Z` already closes.
      if (points.length >= 4 && points[points.length - 2] === startX && points[points.length - 1] === startY) {
        points.length -= 2;
      }
      let d = `M ${origin.x + points[0]!} ${origin.y + points[1]!}`;
      for (let index = 2; index < points.length; index += 2) {
        d += ` L ${origin.x + points[index]!} ${origin.y + points[index + 1]!}`;
      }
      segments.push(`${d} Z`);
    }
  }
  return segments.join(' ');
};

import type { ToolContext } from '@workbench/canvas-engine/tools/tool';
import type { PointerInput } from '@workbench/canvas-engine/types';

import { describe, expect, it } from 'vitest';

import { isOnFirstVertex, movePolyline, polylinePreview, pressPolyline, startPolyline } from './polylineSession';

const pointer = (x: number, y: number, timeStamp = 0): PointerInput => ({
  buttons: 1,
  documentPoint: { x, y },
  modifiers: { alt: false, ctrl: false, meta: false, shift: false },
  pointerType: 'mouse',
  pressure: 0.5,
  screenPoint: { x, y },
  timeStamp,
});

// Identity viewport: document and screen coordinates coincide.
const ctx = { viewport: { documentToScreen: (p: { x: number; y: number }) => p } } as unknown as ToolContext;

describe('polylineSession', () => {
  it('places vertices on separate presses and closes on a press at the first vertex', () => {
    const session = startPolyline(pointer(0, 0, 0));
    expect(pressPolyline(ctx, session, pointer(40, 0, 1000))).toBe('place');
    expect(pressPolyline(ctx, session, pointer(40, 40, 2000))).toBe('place');
    expect(pressPolyline(ctx, session, pointer(3, 2, 3000))).toBe('close');
    expect(session.points).toHaveLength(3);
  });

  it('does not treat the first vertex as a close target before the polygon is fillable', () => {
    const session = startPolyline(pointer(0, 0, 0));
    pressPolyline(ctx, session, pointer(40, 0, 1000));
    expect(isOnFirstVertex(ctx, session, { x: 1, y: 1 })).toBe(false);
    expect(pressPolyline(ctx, session, pointer(1, 1, 2000))).toBe('place');
  });

  it('closes on a quick double-click but places a slow second click at the same spot', () => {
    const quick = startPolyline(pointer(0, 0, 0));
    pressPolyline(ctx, quick, pointer(40, 0, 1000));
    expect(pressPolyline(ctx, quick, pointer(40, 1, 1200))).toBe('close');

    const slow = startPolyline(pointer(0, 0, 0));
    pressPolyline(ctx, slow, pointer(40, 0, 1000));
    expect(pressPolyline(ctx, slow, pointer(40, 1, 5000))).toBe('place');
  });

  it('arms the close cue only while hovering the first vertex and reports each flip', () => {
    const session = startPolyline(pointer(0, 0, 0));
    pressPolyline(ctx, session, pointer(40, 0, 1000));
    pressPolyline(ctx, session, pointer(40, 40, 2000));

    expect(movePolyline(ctx, session, pointer(30, 30))).toBe(false);
    expect(movePolyline(ctx, session, pointer(3, 2))).toBe(true);
    expect(movePolyline(ctx, session, pointer(4, 2))).toBe(false);
    expect(polylinePreview(session)).toEqual({
      closeArmed: true,
      closeRadiusPx: 8,
      cursor: { x: 4, y: 2 },
      kind: 'polygon',
      points: [
        { x: 0, y: 0 },
        { x: 40, y: 0 },
        { x: 40, y: 40 },
      ],
    });
    expect(movePolyline(ctx, session, pointer(20, 20))).toBe(true);
  });
});

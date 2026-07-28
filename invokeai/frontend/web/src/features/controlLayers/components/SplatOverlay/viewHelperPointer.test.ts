import { describe, expect, it } from 'vitest';

import type { ElementBox } from './viewHelperPointer';
import { remapPointerForViewHelper, VIEW_HELPER_DIM } from './viewHelperPointer';

/** Oracle: the exact NDC mapping three's ViewHelper.handleClick applies to a pointer event. */
const viewHelperNdc = (box: ElementBox, clientX: number, clientY: number, dim = VIEW_HELPER_DIM) => {
  const { rect, offsetWidth, offsetHeight } = box;
  const offsetX = rect.left + (offsetWidth - dim);
  const offsetY = rect.top + (offsetHeight - dim);
  return {
    x: ((clientX - offsetX) / (rect.right - offsetX)) * 2 - 1,
    y: -((clientY - offsetY) / (rect.bottom - offsetY)) * 2 + 1,
  };
};

const makeBox = (left: number, top: number, offsetWidth: number, offsetHeight: number, scale: number): ElementBox => ({
  rect: {
    left,
    top,
    right: left + offsetWidth * scale,
    bottom: top + offsetHeight * scale,
    width: offsetWidth * scale,
    height: offsetHeight * scale,
  },
  offsetWidth,
  offsetHeight,
});

/** Visual (client) position of a point given in the element's layout space. */
const visualPos = (box: ElementBox, layoutX: number, layoutY: number, scale: number) => ({
  clientX: box.rect.left + layoutX * scale,
  clientY: box.rect.top + layoutY * scale,
});

describe('remapPointerForViewHelper', () => {
  it('is the identity when the element is unscaled', () => {
    const box = makeBox(10, 20, 400, 300, 1);
    const remapped = remapPointerForViewHelper(box, 350, 250);
    expect(remapped.clientX).toBeCloseTo(350);
    expect(remapped.clientY).toBeCloseTo(250);
  });

  it.each([2, 0.5, 1.37])('maps a click on the gizmo center to NDC (0, 0) at scale %f', (scale) => {
    const box = makeBox(10, 20, 400, 300, scale);
    // Gizmo center in layout space: middle of the bottom-right dim×dim box.
    const { clientX, clientY } = visualPos(box, 400 - VIEW_HELPER_DIM / 2, 300 - VIEW_HELPER_DIM / 2, scale);
    const remapped = remapPointerForViewHelper(box, clientX, clientY);
    const ndc = viewHelperNdc(box, remapped.clientX, remapped.clientY);
    expect(ndc.x).toBeCloseTo(0);
    expect(ndc.y).toBeCloseTo(0);
  });

  it('maps the gizmo corners to NDC extremes at scale 2', () => {
    const box = makeBox(0, 0, 640, 480, 2);
    const topLeft = visualPos(box, 640 - VIEW_HELPER_DIM, 480 - VIEW_HELPER_DIM, 2);
    const bottomRight = visualPos(box, 640, 480, 2);
    const remappedTl = remapPointerForViewHelper(box, topLeft.clientX, topLeft.clientY);
    const remappedBr = remapPointerForViewHelper(box, bottomRight.clientX, bottomRight.clientY);
    const ndcTl = viewHelperNdc(box, remappedTl.clientX, remappedTl.clientY);
    const ndcBr = viewHelperNdc(box, remappedBr.clientX, remappedBr.clientY);
    expect(ndcTl.x).toBeCloseTo(-1);
    expect(ndcTl.y).toBeCloseTo(1);
    expect(ndcBr.x).toBeCloseTo(1);
    expect(ndcBr.y).toBeCloseTo(-1);
  });

  it('documents the bug: without remapping, a scaled gizmo-center click drifts far from its true NDC', () => {
    const box = makeBox(10, 20, 400, 300, 2);
    const { clientX, clientY } = visualPos(box, 400 - VIEW_HELPER_DIM / 2, 300 - VIEW_HELPER_DIM / 2, 2);
    // The true position is the gizmo center, NDC (0, 0); unremapped it lands well off it, missing the axes.
    const ndc = viewHelperNdc(box, clientX, clientY);
    expect(Math.hypot(ndc.x, ndc.y)).toBeGreaterThan(0.3);
  });
});

import { propagateNearestVisibleLabels } from '@workbench/canvas-engine/render/spatial/alphaFill';
import { describe, expect, it } from 'vitest';

import parity from './__fixtures__/spatial_adjustment_parity.json';

const bytes = (text: string): Uint8Array => Uint8Array.from(atob(text), (char) => char.charCodeAt(0));

describe('propagateNearestVisibleLabels', () => {
  it.each(parity.labels)('picks the visible pixel OpenCV labels, ties included, for $name', (entry) => {
    const { height, width } = entry;
    const hidden = bytes(entry.hidden);
    const labels = new Int32Array(width * height);
    const visibleRank = new Int32Array(width * height);
    let rank = 0;
    hidden.forEach((isHidden, index) => {
      labels[index] = isHidden ? -1 : index;
      visibleRank[index] = isHidden ? -1 : rank++;
    });

    propagateNearestVisibleLabels(labels, width, height, new Uint32Array(width * height));

    const expected = new DataView(bytes(entry.labels).buffer);
    expect(Array.from(labels, (label) => visibleRank[label])).toEqual(
      Array.from({ length: width * height }, (_, index) => expected.getInt32(index * 4, true))
    );
  });
});

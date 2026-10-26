import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  clampPercent,
  nudgeAt,
  percentFromKey,
  percentFromPointer,
  touchGesture,
  valueText,
} from './canvas-compare.mjs';

describe('clampPercent', () => {
  it('keeps positions inside the image and rounds to a tenth', () => {
    assert.equal(clampPercent(-12), 0);
    assert.equal(clampPercent(140), 100);
    assert.equal(clampPercent(33.333333), 33.3);
  });

  it('treats a non-number as the left edge', () => {
    assert.equal(clampPercent(Number.NaN), 0);
    assert.equal(clampPercent(Number.POSITIVE_INFINITY), 0);
  });
});

describe('percentFromPointer', () => {
  const rect = { left: 100, width: 400 };

  it('maps the pointer to the share of the image left of it', () => {
    assert.equal(percentFromPointer(100, rect), 0);
    assert.equal(percentFromPointer(250, rect), 37.5);
    assert.equal(percentFromPointer(500, rect), 100);
  });

  it('clamps a pointer dragged past either edge', () => {
    assert.equal(percentFromPointer(40, rect), 0);
    assert.equal(percentFromPointer(900, rect), 100);
  });

  it('falls back to the middle for an image with no width yet', () => {
    assert.equal(percentFromPointer(120, { left: 0, width: 0 }), 50);
  });
});

describe('percentFromKey', () => {
  it('moves 5% per arrow key, in both orientations', () => {
    assert.equal(percentFromKey('ArrowRight', 50), 55);
    assert.equal(percentFromKey('ArrowUp', 50), 55);
    assert.equal(percentFromKey('ArrowLeft', 50), 45);
    assert.equal(percentFromKey('ArrowDown', 50), 45);
  });

  it('stops at the ends', () => {
    assert.equal(percentFromKey('ArrowRight', 98), 100);
    assert.equal(percentFromKey('ArrowLeft', 3), 0);
    assert.equal(percentFromKey('PageDown', 10), 0);
  });

  it('jumps to the ends with Home and End', () => {
    assert.equal(percentFromKey('Home', 64), 0);
    assert.equal(percentFromKey('End', 64), 100);
  });

  it('ignores keys that do not move a slider', () => {
    assert.equal(percentFromKey('Enter', 50), null);
    assert.equal(percentFromKey('a', 50), null);
  });
});

describe('nudgeAt', () => {
  it('swings out to the target at the midpoint and returns to the start', () => {
    assert.equal(nudgeAt(0, 50, 64), 50);
    assert.equal(nudgeAt(0.5, 50, 64), 64);
    assert.equal(nudgeAt(1, 50, 64), 50);
  });

  it('eases: a quarter of the way through it has covered half the swing', () => {
    assert.equal(nudgeAt(0.25, 50, 64), 57);
  });
});

describe('valueText', () => {
  it('names the share of each image in view', () => {
    const template = '{{before}}% before, {{after}}% after';
    assert.equal(valueText(50, template), '50% before, 50% after');
    assert.equal(valueText(62.4, template), '62% before, 38% after');
    assert.equal(valueText(0, template), '0% before, 100% after');
    assert.equal(valueText(30, 'vorher {{before}} %, nachher {{after}} %'), 'vorher 30 %, nachher 70 %');
  });
});

describe('touchGesture', () => {
  it('waits while the press stays within 6px', () => {
    assert.equal(touchGesture(0, 0), 'pending');
    assert.equal(touchGesture(5, -5), 'pending');
  });

  it('scrubs once the press moves further sideways than up or down', () => {
    assert.equal(touchGesture(6, 2), 'scrub');
    assert.equal(touchGesture(-20, 12), 'scrub');
  });

  it('leaves the slider to a vertical swipe, including a diagonal one', () => {
    assert.equal(touchGesture(1, 6), 'scroll');
    assert.equal(touchGesture(-14, 30), 'scroll');
    assert.equal(touchGesture(8, -8), 'scroll');
  });
});

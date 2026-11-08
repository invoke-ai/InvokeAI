import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { stepSpring, velocityFactor, wrapOffset } from './marquee.mjs';

describe('wrapOffset', () => {
  it('keeps offsets within one loop unit to the left of the start', () => {
    assert.equal(wrapOffset(-250, 1000), -250);
    assert.equal(wrapOffset(-1250, 1000), -250);
    assert.equal(wrapOffset(-3250, 1000), -250);
  });

  it('wraps offsets past the start when the strip runs backwards', () => {
    assert.equal(wrapOffset(250, 1000), -750);
    assert.equal(wrapOffset(1250, 1000), -750);
  });

  it('maps both ends of the unit to the start', () => {
    assert.equal(wrapOffset(0, 1000), 0);
    assert.equal(wrapOffset(-1000, 1000), 0);
    assert.ok(Object.is(wrapOffset(-1000, 1000), 0), 'not -0, which would write "-0px"');
  });

  it('stays at the start while the unit has no width', () => {
    assert.equal(wrapOffset(-250, 0), 0);
  });
});

describe('velocityFactor', () => {
  it('scales with scroll velocity and keeps its direction', () => {
    assert.equal(velocityFactor(0), 0);
    assert.equal(velocityFactor(500), 2.5);
    assert.equal(velocityFactor(-200), -1);
  });

  it('is clamped to the maximum either way', () => {
    assert.equal(velocityFactor(9000), 5);
    assert.equal(velocityFactor(-9000), -5);
    assert.equal(velocityFactor(9000, { max: 3 }), 3);
  });
});

describe('stepSpring', () => {
  it('settles on its target', () => {
    const spring = { value: 0, velocity: 0 };
    for (let frame = 0; frame < 120; frame += 1) stepSpring(spring, 800, 1 / 60);
    assert.ok(Math.abs(spring.value - 800) < 1, `value ${spring.value}`);
    assert.ok(Math.abs(spring.velocity) < 5, `velocity ${spring.velocity}`);
  });

  it('eases back to rest without overshooting far past it', () => {
    const spring = { value: 1000, velocity: 0 };
    let lowest = Infinity;
    for (let frame = 0; frame < 120; frame += 1) lowest = Math.min(lowest, stepSpring(spring, 0, 1 / 60).value);
    assert.ok(Math.abs(spring.value) < 1, `value ${spring.value}`);
    assert.ok(lowest > -100, `overshoot ${lowest}`);
  });

  it('stays stable across a long frame', () => {
    const spring = { value: 0, velocity: 0 };
    stepSpring(spring, 1000, 0.5);
    assert.ok(Number.isFinite(spring.value) && Math.abs(spring.value - 1000) < 50, `value ${spring.value}`);
  });
});

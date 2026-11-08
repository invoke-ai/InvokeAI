// The showcase marquee's motion (`ShowcaseMarquee.astro`): a steady drift that page scrolling speeds up and turns
// around, after reactbits' Scroll Velocity.

/**
 * An offset in the loop, from just above `-unit` to 0. The track holds two copies of the loop unit, so any offset in
 * that range shows a seamless strip; wrapping keeps it there whichever way the strip moves.
 * @param {number} offset in pixels
 * @param {number} unit the loop unit's width in pixels
 */
export const wrapOffset = (offset, unit) => (unit > 0 ? (((offset % unit) - unit) % unit) + 0 : 0);

/**
 * Extra speed from scroll velocity, as a multiple of the drift: `full` pixels a second of scrolling adds `max` times
 * the drift. Its sign is the scroll direction; it is clamped to `±max`.
 * @param {number} velocity in pixels a second, positive when scrolling down
 * @param {{ full?: number, max?: number }} [options]
 */
export const velocityFactor = (velocity, { full = 1000, max = 5 } = {}) =>
  Math.max(-max, Math.min(max, (velocity / full) * max));

/**
 * Advances a damped spring (unit mass) towards `target` by `dt` seconds, in steps short enough to stay stable.
 * @param {{ value: number, velocity: number }} spring updated in place
 * @param {number} target
 * @param {number} dt in seconds
 * @param {{ stiffness?: number, damping?: number }} [options]
 */
export const stepSpring = (spring, target, dt, { stiffness = 400, damping = 50 } = {}) => {
  for (let remaining = dt; remaining > 0; remaining -= 1 / 120) {
    const step = Math.min(remaining, 1 / 120);
    spring.velocity += (stiffness * (target - spring.value) - damping * spring.velocity) * step;
    spring.value += spring.velocity * step;
  }
  return spring;
};

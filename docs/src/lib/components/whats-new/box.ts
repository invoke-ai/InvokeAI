/** Places a `wn-abs` element on a motif stage (`Stage.astro`): position and size in the stage's units. */
export const box = (x: number, y: number, width: number, height: number) =>
  `--x: ${x}; --y: ${y}; --w: ${width}; --h: ${height}`;

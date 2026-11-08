/* eslint-disable no-console -- this module exists to intercept and re-emit console output */
import { afterAll, beforeAll } from 'vitest';

/** Count React act warnings per component and report once per file; preserve every other console message unchanged. */
const ACT_WARNING = 'was not wrapped in act(';
const COMPONENT = /An update to (\S+)/;
const PLACEHOLDER = /%[sdifoOc]/g;

/** React logs this warning as a format string with the component in a trailing argument. */
const interpolate = (args: unknown[]): string => {
  const [template, ...rest] = args;

  if (typeof template !== 'string') {
    return args.map((arg) => String(arg)).join(' ');
  }

  let next = 0;
  const filled = template.replace(PLACEHOLDER, () => (next < rest.length ? String(rest[next++]) : '%s'));

  return [filled, ...rest.slice(next)].map((arg) => String(arg)).join(' ');
};

const counts = new Map<string, number>();
let passThrough: typeof console.error | null = null;

beforeAll(() => {
  passThrough = console.error;
  console.error = (...args: unknown[]) => {
    const text = interpolate(args);

    if (text.includes(ACT_WARNING)) {
      const component = COMPONENT.exec(text)?.[1] ?? 'unknown component';
      counts.set(component, (counts.get(component) ?? 0) + 1);
      return;
    }

    passThrough?.(...args);
  };
});

afterAll(() => {
  const report = passThrough;

  if (report) {
    console.error = report;
  }
  if (counts.size === 0) {
    return;
  }

  const summary = [...counts.entries()]
    .sort(([, left], [, right]) => right - left)
    .map(([component, count]) => `${component} x${String(count)}`)
    .join(', ');

  counts.clear();
  report?.(`act() warnings collapsed: ${summary}`);
});

import { afterEach, expect, it } from 'vitest';

import { auditAccessibility } from './auditAccessibility.testing';

/** Use a deliberately slow fade to reproduce a failing mid-animation contrast audit and a passing settled audit. */

const FADE_MS = 800;

let host: HTMLDivElement | null = null;
let styles: HTMLStyleElement | null = null;

/** Light text on a dark panel: ~15:1 at rest, far inside the 4.5:1 rule. */
const render = (): HTMLDivElement => {
  styles = document.head.appendChild(document.createElement('style'));
  styles.textContent = `@keyframes audit-probe-fade { from { opacity: 0.15 } to { opacity: 1 } }
    .audit-probe { animation: audit-probe-fade ${String(FADE_MS)}ms both;
      background: #10131a; color: #f2f5fa; font-size: 16px; padding: 16px; width: 320px }`;

  host = document.body.appendChild(document.createElement('div'));
  host.className = 'audit-probe';
  host.textContent = 'Replace all uses of this missing font';

  return host;
};

afterEach(() => {
  host?.remove();
  styles?.remove();
  host = null;
  styles = null;
});

it('reports contrast failures on a palette that is fine, when the surface is still fading in', async () => {
  const panel = render();
  // Deliberately NOT settled: this is the bug, kept as an executable description of it.
  const { default: axe } = await import('axe-core');
  const midFlight = (await axe.run(panel)).violations;

  expect(panel.getAnimations()[0]?.playState).toBe('running');
  expect(midFlight.map((violation) => violation.id)).toContain('color-contrast');
});

it('reports nothing once the fade has finished, which is what the helper waits for', async () => {
  const panel = render();

  expect(await auditAccessibility(panel)).toEqual([]);
  expect(panel.getAnimations().every((animation) => animation.playState === 'finished')).toBe(true);
});

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { PinIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it } from 'vitest';

import { ProgressBadge, type ProgressBadgeProps } from './ProgressBadge';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderBadge = async (props: ProgressBadgeProps): Promise<HTMLElement> => {
  host ??= document.createElement('div');
  host.style.cssText = 'position:relative;width:80px;height:80px;';
  document.body.append(host);
  root ??= createRoot(host);
  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <ProgressBadge {...props} />
      </ChakraProvider>
    );
  });
  return host.firstElementChild as HTMLElement;
};

const indicatorSize = (badge: HTMLElement) => {
  const { height, width } = badge.firstElementChild!.getBoundingClientRect();
  return { height, width };
};

it('keeps one footprint across unknown progress, known progress and settled states', async () => {
  const unknown = await renderBadge({ label: 'Preparing', percentage: null, state: 'running' });
  const spinner = unknown.querySelector('[role="progressbar"]')!;
  expect(spinner.classList.contains('chakra-spinner')).toBe(true);
  expect(spinner.getAttribute('aria-label')).toBe('Preparing');
  const spinnerSize = indicatorSize(unknown);

  const known = await renderBadge({ label: '42%', percentage: 42, state: 'running' });
  const ring = known.querySelector('[role="progressbar"]')!;
  expect(ring.classList.contains('chakra-spinner')).toBe(false);
  expect(ring.getAttribute('aria-valuenow')).toBe('42');
  expect(indicatorSize(known)).toEqual(spinnerSize);

  const queued = await renderBadge({ label: 'Queued', percentage: null, state: 'queued' });
  expect(queued.querySelector('svg[aria-label="Queued"]')).not.toBeNull();
  expect(indicatorSize(queued)).toEqual(spinnerSize);
});

it('shows the override icon in place of any state indicator, and a check once completed', async () => {
  const pinned = await renderBadge({ icon: PinIcon, percentage: 42, state: 'running' });
  expect(pinned.querySelector('.lucide-pin')).not.toBeNull();
  expect(pinned.querySelector('[role="progressbar"]')).toBeNull();

  const completed = await renderBadge({ label: 'Finishing', percentage: null, state: 'completed' });
  expect(completed.querySelector('svg.lucide-check[aria-label="Finishing"]')).not.toBeNull();
});

it('hides an unlabeled badge from assistive technology', async () => {
  const badge = await renderBadge({ percentage: null, state: 'running' });
  expect(badge.getAttribute('aria-hidden')).toBe('true');
  expect(badge.querySelector('.chakra-spinner')).not.toBeNull();
});

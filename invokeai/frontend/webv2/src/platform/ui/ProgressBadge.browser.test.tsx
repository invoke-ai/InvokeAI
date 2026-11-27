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

it('shows the override icon in place of any state indicator, and a check once completed', async () => {
  const pinned = await renderBadge({ icon: PinIcon, percentage: 42, state: 'running' });
  expect(pinned.querySelector('.lucide-pin')).not.toBeNull();
  expect(pinned.querySelector('[role="progressbar"]')).toBeNull();

  const completed = await renderBadge({ label: 'Finishing', percentage: null, state: 'completed' });
  expect(completed.querySelector('svg.lucide-check[aria-label="Finishing"]')).not.toBeNull();
});

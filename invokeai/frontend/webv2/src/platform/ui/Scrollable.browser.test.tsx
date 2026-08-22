import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { Scrollable } from './Scrollable';

/**
 * zag pins `min-width: fit-content` inline on the scroll-area content box. A
 * horizontal strip needs that; a vertical area must not inherit it, because it
 * renders no horizontal scrollbar — anything pushed sideways there is simply
 * unreachable. These two tests are the whole contract.
 */

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** No spaces and no hyphens: nothing the layout can break on, so it contributes its whole width to min-content. */
const UNBREAKABLE = 'AVeryLongUnbreakableWorkflowNameThatHasNowhereToWrap';

const renderScrollable = async (children: ReactNode, orientation: 'horizontal' | 'vertical'): Promise<HTMLElement> => {
  host = document.createElement('div');
  host.style.cssText = 'width:200px;height:80px;';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <Scrollable label="subject" orientation={orientation}>
          {children}
        </Scrollable>
      </ChakraProvider>
    );
  });

  return host.querySelector<HTMLElement>('[role="region"][aria-label="subject"]')!;
};

describe('Scrollable', () => {
  it('does not let a vertical area scroll sideways, whatever its content asks for', async () => {
    const viewport = await renderScrollable(<div>{UNBREAKABLE}</div>, 'vertical');

    expect(viewport.clientWidth).toBeGreaterThan(0);
    expect(viewport.scrollWidth).toBeLessThanOrEqual(viewport.clientWidth + 1);
  });

  it('still lets a horizontal strip outgrow its viewport', async () => {
    const viewport = await renderScrollable(
      <div style={{ display: 'flex', whiteSpace: 'nowrap' }}>
        {UNBREAKABLE}
        {UNBREAKABLE}
      </div>,
      'horizontal'
    );

    expect(viewport.scrollWidth).toBeGreaterThan(viewport.clientWidth);
  });
});

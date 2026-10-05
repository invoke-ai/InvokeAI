/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

import { PanelBodySlot } from './WidgetRenderer';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

afterEach(async () => {
  await interact(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const PANEL_WIDTH = 300;
const LONG_UNBREAKABLE = 'x'.repeat(600);

describe('PanelBodySlot', () => {
  it('never lets intrinsically wide widget content stretch past the panel width', async () => {
    host = document.createElement('div');
    host.style.cssText = `display:flex;flex-direction:column;height:400px;width:${PANEL_WIDTH}px;`;
    document.body.append(host);
    root = createRoot(host);

    await interact(() => {
      root?.render(
        <ChakraProvider value={system}>
          <PanelBodySlot>
            <div data-testid="widget-root">
              {/* A row like QueueItemRow: clips its own content when constrained. */}
              <div data-testid="row" style={{ overflow: 'hidden' }}>
                <span style={{ whiteSpace: 'nowrap' }}>{LONG_UNBREAKABLE}</span>
              </div>
              {/* An unconstrained wide flex row like a button group. */}
              <div data-testid="strip" style={{ display: 'flex', flexWrap: 'wrap' }}>
                {Array.from({ length: 8 }, (_, index) => (
                  <button key={index} style={{ whiteSpace: 'nowrap' }}>
                    Wide recall verb {index}
                  </button>
                ))}
              </div>
            </div>
          </PanelBodySlot>
        </ChakraProvider>
      );
    });

    const widgetRoot = document.querySelector<HTMLElement>('[data-testid="widget-root"]')!;
    const row = document.querySelector<HTMLElement>('[data-testid="row"]')!;
    const strip = document.querySelector<HTMLElement>('[data-testid="strip"]')!;

    expect(widgetRoot.getBoundingClientRect().width).toBeLessThanOrEqual(PANEL_WIDTH);
    expect(row.getBoundingClientRect().width).toBeLessThanOrEqual(PANEL_WIDTH);
    expect(strip.getBoundingClientRect().width).toBeLessThanOrEqual(PANEL_WIDTH);
    // The wrap actually engaged: eight buttons cannot fit one 300px line.
    expect(strip.getBoundingClientRect().height).toBeGreaterThan(row.getBoundingClientRect().height * 1.5);
  });
});

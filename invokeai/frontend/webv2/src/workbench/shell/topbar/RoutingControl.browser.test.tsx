import { ChakraProvider } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { system } from '@theme/system';
import { graphWidgetSources } from '@workbench/graphWidgets';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { InvocationState } from './useInvocationState';

vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({
    generation: {
      setDestination: vi.fn(),
      setSource: vi.fn(),
      toggleRoutingLock: vi.fn(),
    },
  }),
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'topbar.routing.change': 'Change routing',
        'topbar.routing.destination': 'Destination',
        'topbar.routing.lockRouting': 'Lock routing',
        'topbar.routing.source': 'Source',
        'topbar.routing.unlockRouting': 'Unlock routing',
      })[key] ?? key,
  }),
}));

import { RoutingControl } from './RoutingControl';

const NOOP_INVOKE = () => Promise.resolve();
const BASE_STATE = {
  batchCount: 1,
  blockingReasons: [],
  invocation: {
    destination: 'gallery',
    destinationLocked: false,
    sourceId: 'generate',
    sourceLocked: false,
  },
  invoke: NOOP_INVOKE,
  isValid: true,
  placedTypeIds: new Set(['generate']),
  promptExpansion: {},
  sourceValues: {},
  sources: graphWidgetSources,
  visibleTypeIds: new Set(['generate']),
} as unknown as InvocationState;
const LOCKED_STATE = {
  ...BASE_STATE,
  invocation: { ...BASE_STATE.invocation, destinationLocked: true, sourceLocked: true },
} satisfies InvocationState;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderControl = async (state: InvocationState) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(async () => {
    root?.render(
      <ChakraProvider value={system}>
        <Group attached>
          <Button data-routing-height-reference="" size="xs">
            Invoke
          </Button>
          <RoutingControl state={state} />
        </Group>
      </ChakraProvider>
    );
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('RoutingControl', () => {
  it('uses narrow diagonal route geometry without an arrow and adds only a top-right dot when locked', async () => {
    await renderControl(BASE_STATE);

    const button = document.querySelector<HTMLButtonElement>('[data-routing-control]');
    const sourceIcon = button?.querySelector<SVGElement>('[data-routing-source-icon]');
    const destinationIcon = button?.querySelector<SVGElement>('[data-routing-destination-icon]');
    const heightReference = document.querySelector<HTMLButtonElement>('[data-routing-height-reference]');
    expect(button).not.toBeNull();
    expect(heightReference).not.toBeNull();
    expect(sourceIcon).not.toBeNull();
    expect(destinationIcon).not.toBeNull();

    const buttonBounds = button!.getBoundingClientRect();
    const sourceBounds = sourceIcon!.getBoundingClientRect();
    const destinationBounds = destinationIcon!.getBoundingClientRect();
    expect(buttonBounds.width).toBe(36);
    expect(buttonBounds.height).toBe(32);
    expect(buttonBounds.height).toBe(heightReference!.getBoundingClientRect().height);
    expect(destinationBounds.left - sourceBounds.left).toBeGreaterThanOrEqual(8);
    expect(destinationBounds.top - sourceBounds.top).toBeGreaterThanOrEqual(8);
    expect(button?.querySelectorAll('svg')).toHaveLength(2);
    expect(button?.querySelector('[data-routing-lock-indicator]')).toBeNull();

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Group attached>
            <Button data-routing-height-reference="" size="xs">
              Invoke
            </Button>
            <RoutingControl state={LOCKED_STATE} />
          </Group>
        </ChakraProvider>
      );
    });

    const lockIndicator = button?.querySelector<HTMLElement>('[data-routing-lock-indicator]');
    expect(lockIndicator).not.toBeNull();
    const dotBounds = lockIndicator!.getBoundingClientRect();
    expect(dotBounds.left).toBeGreaterThan(sourceBounds.right);
    expect(dotBounds.top).toBeLessThan(destinationBounds.top);
  });
});

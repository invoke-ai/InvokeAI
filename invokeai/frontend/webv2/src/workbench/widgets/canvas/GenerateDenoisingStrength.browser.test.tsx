import type { CanvasDenoisingStrength, GenerationUiAdapter } from '@features/generation/react';
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { system } from '@theme/system';
import { readCanvasDenoisingStrength } from '@workbench/widgets/canvas/invoke/canvasStrength';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { createWorkbenchStore, type WorkbenchInternalStore } from '@workbench/workbenchStore';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ store: null as unknown as WorkbenchInternalStore }));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
// The real aggregate store behind the hooks the strength control reads.
vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  return {
    useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected): Selected =>
      useSyncExternalStore(harness.store.subscribe, () => selector(harness.store.getSnapshot().activeProject)),
    useWorkbenchCommands: () => harness.store.commands,
  };
});

import { GenerateDenoisingStrength } from './GenerateDenoisingStrength';

const generationUi = {
  sectionPreferences: { sectionsOpen: {}, setSectionOpen: () => undefined },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Stands in for the Render section that lays itself out around the strength slots. */
const sectionRendered = vi.fn();
const Section = ({ slots }: { slots: CanvasDenoisingStrength }) => {
  sectionRendered();
  return (
    <div>
      <div data-testid="header">{slots.badges}</div>
      {slots.field}
    </div>
  );
};

const slider = () => host!.querySelector<HTMLElement>('[data-scope="scrubber"] [role="slider"]')!;
const committedStrength = () =>
  readCanvasDenoisingStrength(getProjectWidgetValues(harness.store.getSnapshot().activeProject, 'canvas'));

beforeEach(async () => {
  accountLifecycle.activate('generate-denoising-strength-test');
  harness.store = createWorkbenchStore();
  sectionRendered.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GenerationUiProvider adapter={generationUi}>
          <GenerateDenoisingStrength>{(slots) => <Section slots={slots} />}</GenerateDenoisingStrength>
        </GenerationUiProvider>
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  accountLifecycle.invalidate();
});

describe('GenerateDenoisingStrength', () => {
  it('updates its badge and field while scrubbing without re-rendering the section around them', async () => {
    const rendersAtMount = sectionRendered.mock.calls.length;
    const header = host!.querySelector('[data-testid="header"]')!;
    expect(header.textContent).toBe('75%');

    for (let step = 0; step < 2; step += 1) {
      await act(() => {
        slider().dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
      });
    }

    expect(header.textContent).toBe('77%');
    expect(slider().getAttribute('aria-valuenow')).toBe('0.77');
    // The draft commits after its debounce, and that commit does not re-render the section either.
    await vi.waitFor(() => expect(committedStrength()).toBeCloseTo(0.77), { timeout: 1_000 });
    expect(sectionRendered).toHaveBeenCalledTimes(rendersAtMount);
  });
});

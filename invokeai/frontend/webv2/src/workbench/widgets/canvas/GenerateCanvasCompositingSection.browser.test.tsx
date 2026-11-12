import type { GenerationUiAdapter } from '@features/generation/react';

import { ChakraProvider } from '@chakra-ui/react';
import { GenerationUiProvider } from '@features/generation/react';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { DEFAULT_CANVAS_COMPOSITING } from '@workbench/widgets/canvas/invoke/canvasCompositing';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

/** The canvas widget's values, as the workbench holds them. */
const canvas = vi.hoisted(() => ({
  values: null as unknown as ReturnType<typeof createExternalStoreCore<Record<string, unknown>>>,
}));

vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@workbench/WorkbenchContext', async () => {
  const { useSyncExternalStore } = await import('react');
  return {
    useActiveProjectId: () => 'project-1',
    useWidgetValuesSelector: <Selected,>(_typeId: string, selector: (values: Record<string, unknown>) => Selected) =>
      useSyncExternalStore(canvas.values.subscribe, () => selector(canvas.values.getSnapshot())),
    useWorkbenchCommands: () => ({
      widgets: {
        patchValues: (_typeId: string, patch: Record<string, unknown>) => canvas.values.patchSnapshot(patch),
      },
    }),
  };
});

import { GenerateCanvasCompositingSection } from './GenerateCanvasCompositingSection';

const generationUi = {
  sectionPreferences: { sectionsOpen: {}, setSectionsOpen: () => undefined },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (values: Record<string, unknown>) => {
  canvas.values = createExternalStoreCore<Record<string, unknown>>(values);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <GenerationUiProvider adapter={generationUi}>
          <GenerateCanvasCompositingSection />
        </GenerationUiProvider>
      </ChakraProvider>
    )
  );
};

const title = () => host!.querySelector<HTMLElement>('.generate-section > [data-part="trigger"]')!;
const openMenu = () =>
  act(() => {
    title().dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
  });
const resetItem = () => page.getByRole('menuitem', { name: 'widgets.generate.sectionMenu.reset' });

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateCanvasCompositingSection', () => {
  it('marks its title while a field is off its default, and resets the section from its menu', async () => {
    await render({ coherenceMode: 'Staged', infillMethod: 'color', infillColorValue: { a: 1, b: 255, g: 0, r: 0 } });

    expect(title().querySelector('[data-section-changed]')).not.toBeNull();

    await openMenu();
    await act(() => userEvent.click(resetItem()));

    expect(canvas.values.getSnapshot()).toMatchObject({
      coherenceEdgeSize: DEFAULT_CANVAS_COMPOSITING.coherenceEdgeSize,
      coherenceMinDenoise: DEFAULT_CANVAS_COMPOSITING.coherenceMinDenoise,
      coherenceMode: DEFAULT_CANVAS_COMPOSITING.coherenceMode,
      infillColorValue: DEFAULT_CANVAS_COMPOSITING.infillColorValue,
      infillMethod: DEFAULT_CANVAS_COMPOSITING.infillMethod,
      maskBlur: DEFAULT_CANVAS_COMPOSITING.maskBlur,
    });
    expect(title().querySelector('[data-section-changed]')).toBeNull();
  });

  it('ignores a stored infill color that the chosen infill does not use', async () => {
    await render({
      infillMethod: DEFAULT_CANVAS_COMPOSITING.infillMethod,
      infillColorValue: { a: 1, b: 255, g: 0, r: 0 },
    });

    expect(title().querySelector('[data-section-changed]')).toBeNull();
    await openMenu();
    await expect.element(resetItem()).toHaveAttribute('data-disabled');
  });
});

import { ChakraProvider } from '@chakra-ui/react';
import { FeatureHintsProvider, type FeatureHintsAdapter } from '@platform/ui/hints/hintsContext';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { createInstance } from 'i18next';
import { PlusIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { LayerRowCommands } from './layerRowCommands';

import { LayerStackHeader } from './LayerStackHeader';

vi.mock('./useLayerStackActions', () => ({
  useLayerStackActions: () => [
    { disabled: false, icon: PlusIcon, id: 'add', label: 'New layer', run: () => undefined },
  ],
}));

const HEADING = 'Raster Layers';
const GIST = 'The pixel layers that compose your image and feed generation.';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        hints: {
          layerStackRaster: { heading: HEADING, paragraphs: [GIST, 'Layers apply top to bottom.'] },
        },
        widgets: {
          layers: {
            groupActions: { collapse: 'Collapse', expand: 'Expand', new: 'New layer' },
            groups: { raster: 'Raster Layers' },
          },
        },
      },
    },
  },
});

const commands = {
  focus: () => undefined,
  keyDown: () => undefined,
  openStackMenu: () => undefined,
  toggleCollapse: () => undefined,
} as unknown as LayerRowCommands;

const hintsAdapter: FeatureHintsAdapter = { enabled: true, onDisable: null };

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const wait = (ms: number): Promise<void> =>
  act(
    () =>
      new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, ms);
      })
  );
const hintCard = (): Element | null => document.querySelector('[data-scope="hover-card"][data-part="content"]');

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const render = async () => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '320px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FeatureHintsProvider adapter={hintsAdapter}>
            <LayerStackHeader
              collapsed={false}
              commands={commands}
              document={createEmptyCanvasDocument()}
              editingLocked={false}
              engine={null}
              focused
              leafCount={0}
              posInSet={1}
              rowKey="header:raster"
              setSize={4}
              stack="raster"
            />
          </FeatureHintsProvider>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LayerStackHeader hint', () => {
  it('describes the tree item persistently and opens the informational popover on keyboard focus', async () => {
    await render();
    const header = host!.querySelector<HTMLElement>('[role="treeitem"]')!;
    const description = document.getElementById(header.getAttribute('aria-describedby')!);
    expect(header.contains(description)).toBe(true);
    expect(description!.textContent).toBe(GIST);

    await act(async () => {
      await userEvent.tab();
    });
    expect(document.activeElement).toBe(header);
    await wait(900);
    const card = hintCard();
    expect(card?.textContent).toContain(HEADING);
    expect(card?.textContent).toContain(GIST);
  });

  it('opens over the title but never while the pointer rests on a stack action', async () => {
    await render();
    const action = host!.querySelector<HTMLElement>('button[aria-label="New layer"]')!;
    const title = [...host!.querySelectorAll<HTMLElement>('[role="treeitem"] *')].find(
      (element) => element.textContent === 'Raster Layers (0)'
    )!;

    await act(() => userEvent.hover(action));
    await wait(900);
    expect(hintCard()).toBeNull();

    await act(() => userEvent.hover(title));
    await wait(900);
    expect(hintCard()?.textContent).toContain(HEADING);

    await act(() => userEvent.hover(action));
    await wait(900);
    expect(hintCard()).toBeNull();

    // Leaving the header straight from an action must not arm a late open.
    await act(() => userEvent.unhover(action));
    await wait(900);
    expect(hintCard()).toBeNull();
  });
});

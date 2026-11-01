/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider, Menu } from '@chakra-ui/react';
import { system } from '@theme/system';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const harness = vi.hoisted(() => ({ project: null as Project | null }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.project!),
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }) }));

import { CanvasCropAllLayersMenuItem } from './CanvasCropAllLayersMenuItem';

const i18n = createInstance();
let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Opens a menu holding only the item, over a project whose canvas holds `nodes`. */
const renderItem = async (nodes: CanvasNodeContract[], documentEditingLocked: boolean) => {
  const base = createInitialWorkbenchState().projects[0]!;
  harness.project = {
    ...base,
    canvas: { ...base.canvas, document: { ...base.canvas.document, stacks: stacksFrom(nodes) } },
  };
  const layers = { cropAllLayersToBbox: vi.fn(() => Promise.resolve({ status: 'busy' as const })) };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Menu.Root>
            <Menu.Trigger>Canvas</Menu.Trigger>
            <Menu.Positioner>
              <Menu.Content>
                <CanvasCropAllLayersMenuItem documentEditingLocked={documentEditingLocked} layers={layers} />
              </Menu.Content>
            </Menu.Positioner>
          </Menu.Root>
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  await act(() => userEvent.click(page.getByRole('button', { name: 'Canvas' })));
  const item = document.querySelector<HTMLElement>('[data-value="crop-all-layers-to-bbox"]')!;
  await act(() => userEvent.keyboard('{ArrowDown}'));
  await vi.waitFor(() => expect(item.hasAttribute('data-highlighted')).toBe(true));
  return { item, layers };
};

const reasonOf = (item: HTMLElement): string | null | undefined =>
  document.getElementById(item.getAttribute('aria-describedby') ?? '')?.textContent;

describe('Crop all layers menu item', () => {
  it.each([
    [
      'every layer is locked, by itself or by its group',
      [
        layerContract('locked', 'raster', { isLocked: true }),
        groupContract('G', [layerContract('inside', 'control')], { isLocked: true }),
      ],
      false,
      'Every layer is locked. Unlock a layer to crop it.',
    ],
    [
      'Filter or Select Object holds the document',
      [layerContract('free', 'raster')],
      true,
      'Unavailable while Filter or Select Object is running on the canvas.',
    ],
  ])('stays reachable and says why it cannot run when %s', async (_case, nodes, documentEditingLocked, reason) => {
    const { item, layers } = await renderItem(nodes, documentEditingLocked);

    expect(item).toHaveAttribute('aria-disabled', 'true');
    expect(reasonOf(item)).toBe(reason);
    await act(() => userEvent.keyboard('{Enter}'));
    expect(layers.cropAllLayersToBbox).not.toHaveBeenCalled();
  });

  it('crops when an unlocked layer exists and the canvas is free', async () => {
    const { item, layers } = await renderItem([layerContract('free', 'raster')], false);

    expect(item).not.toHaveAttribute('aria-disabled');
    await act(() => userEvent.keyboard('{Enter}'));
    expect(layers.cropAllLayersToBbox).toHaveBeenCalledOnce();
  });
});

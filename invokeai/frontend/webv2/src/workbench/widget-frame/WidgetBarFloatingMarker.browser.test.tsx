/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetTypeId,
} from '@workbench/widgetContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { system } from '@theme/system';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import { createWidgetRegionViewModel } from '@workbench/widgetRegionViewModel';
import i18next from 'i18next';
import { MapIcon } from 'lucide-react';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { WidgetBar, type WidgetBarGroup } from './WidgetBar';

/**
 * A floated widget keeps its rail slot as a marker: it points at the window instead of docking it, and docking
 * and removing are its context menu's job.
 */

const TestView = () => null;

const createWidget = (id: WidgetTypeId, label: string): RegisteredWidget => ({
  implementation: createWidgetImplementationResource(id, () => Promise.resolve({ view: TestView })),
  manifest: {
    apiVersion: 1,
    allowMultiple: false,
    allowedRegions: ['left', 'right'],
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: MapIcon,
    id,
    label,
    load: () => Promise.resolve({ view: TestView }),
    state: { createInitial: () => ({}), persistence: 'project', version: 1 },
    version: 1,
  } as NormalizedWidgetManifest,
  status: 'enabled',
});

const createInstance = (id: string): WidgetInstanceContract => ({
  createdAt: '2026-01-01T00:00:00.000Z',
  id,
  state: { id, label: id, values: {}, version: 1 },
  typeId: id,
});

const viewModel = createWidgetRegionViewModel({
  activeInstanceId: 'gallery',
  floatingWidgets: { 'image-map': { returnIndex: 1, returnRegion: 'right' } },
  instanceIds: ['gallery', 'queue'],
  region: 'right',
  widgetInstances: {
    gallery: createInstance('gallery'),
    'image-map': createInstance('image-map'),
    queue: createInstance('queue'),
  },
  widgets: [createWidget('gallery', 'Gallery'), createWidget('image-map', 'Image Map'), createWidget('queue', 'Queue')],
});

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          floating: {
            destinations: { right: 'right panel' },
            dockTo: 'Dock to {{destination}}',
            railMarker: '{{label}}, floating window',
            railMarkerHint: '{{label}} is floating — click to show the window',
          },
          rail: { inspectWidgets: 'Inspect widgets' },
          removeWidget: 'Remove {{label}}',
        },
      },
    },
  },
});

const handlers = { onDock: vi.fn(), onRemoveFloating: vi.fn(), onSelect: vi.fn(), onToggle: vi.fn() };
const GROUPS: WidgetBarGroup[] = [
  {
    activeId: 'gallery',
    dropState: { helperText: '', isActive: false, isAllowed: false },
    railItems: viewModel.placedItems,
    region: 'right',
  },
];

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderBar = async (projectId = 'project-a') => {
  await act(async () => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <DndContext>
            <WidgetBar
              edgeRegion="right"
              groups={GROUPS}
              menuItems={viewModel.placedItems}
              projectId={projectId}
              side="right"
              {...handlers}
            />
          </DndContext>
        </ChakraProvider>
      </I18nextProvider>
    );
    await Promise.resolve();
  });
};

const slot = (label: string) => host!.querySelector<HTMLButtonElement>(`button[aria-label="${label}"]`)!;
const marker = () => slot('Image Map, floating window');

const openMenu = async (button: HTMLElement) => {
  await act(async () => {
    button.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 20, clientY: 20 }));
    await Promise.resolve();
  });
};

const menuItem = (value: string) =>
  document.body.querySelector<HTMLElement>(`[role="menuitem"][data-value="${value}"]`);

beforeEach(() => {
  for (const handler of Object.values(handlers)) {
    handler.mockClear();
  }
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('WidgetBar floating marker', () => {
  it('keeps the floated widget in its slot as a badged marker that is not a sortable tab', async () => {
    await renderBar();

    const labels = [...host!.querySelectorAll('[data-rail-group="right"] button')].map((button) =>
      button.getAttribute('aria-label')
    );

    expect(labels).toEqual(['Gallery', 'Image Map, floating window', 'Queue']);
    expect(marker().querySelector('[data-floating-badge]')).not.toBeNull();
    // A plain button: not a sortable tab, and not a toggle.
    expect(marker().getAttribute('aria-roledescription')).toBeNull();
    expect(marker().hasAttribute('aria-pressed')).toBe(false);
    expect(slot('Gallery').getAttribute('aria-roledescription')).toBe('sortable');
    expect(slot('Gallery').getAttribute('aria-pressed')).toBe('true');
    expect(slot('Gallery').querySelector('[data-floating-badge]')).toBeNull();
  });

  it('trades its dashed outline for the focus ring under keyboard focus', async () => {
    await renderBar();

    expect(getComputedStyle(marker()).outlineStyle).toBe('dashed');

    slot('Gallery').focus();
    await userEvent.keyboard('{Tab}');

    expect(document.activeElement).toBe(marker());
    expect(marker().matches(':focus-visible')).toBe(true);
    expect(getComputedStyle(marker())).toMatchObject({ outlineStyle: 'solid', outlineWidth: '2px' });
  });

  it('reports activation as a selection, leaving docking to its menu', async () => {
    await renderBar();

    await act(async () => {
      marker().click();
      await Promise.resolve();
    });

    expect(handlers.onSelect).toHaveBeenCalledWith('right', 'image-map');
    expect(handlers.onDock).not.toHaveBeenCalled();
  });

  it('offers Dock to its return region and Remove from the context menu', async () => {
    await renderBar();
    await openMenu(marker());

    expect(menuItem('dock-widget')?.textContent).toBe('Dock to right panel');
    expect(menuItem('remove-widget')?.textContent).toBe('Remove Image Map');

    await act(async () => {
      menuItem('dock-widget')?.click();
      await Promise.resolve();
    });

    expect(handlers.onDock).toHaveBeenCalledWith('image-map');
    expect(handlers.onToggle).not.toHaveBeenCalled();
  });

  it('removes the window from the marker menu', async () => {
    await renderBar();
    await openMenu(marker());

    await act(async () => {
      menuItem('remove-widget')?.click();
      await Promise.resolve();
    });

    expect(handlers.onRemoveFloating).toHaveBeenCalledExactlyOnceWith('image-map');
    expect(handlers.onToggle).not.toHaveBeenCalled();
    expect(handlers.onDock).not.toHaveBeenCalled();
  });

  it('removes the window from the enable menu as a toggle, not as a marker removal', async () => {
    await renderBar();
    await act(async () => {
      slot('Inspect widgets').click();
      await Promise.resolve();
    });

    const row = [...document.body.querySelectorAll<HTMLElement>('[role="menuitemcheckbox"]')].find(
      (item) => item.textContent === 'Image Map'
    )!;

    await act(async () => {
      row.click();
      await Promise.resolve();
    });

    expect(handlers.onToggle).toHaveBeenCalledExactlyOnceWith(
      expect.objectContaining({ id: 'image-map', isFloating: true })
    );
    expect(handlers.onRemoveFloating).not.toHaveBeenCalled();
  });

  it('removes a docked tab from its menu as a toggle', async () => {
    await renderBar();
    await openMenu(slot('Queue'));

    await act(async () => {
      menuItem('remove-widget')?.click();
      await Promise.resolve();
    });

    expect(handlers.onToggle).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ id: 'queue' }));
    expect(handlers.onRemoveFloating).not.toHaveBeenCalled();
  });

  it("closes a marker's menu when the project changes, so it cannot dock or remove the new project's window", async () => {
    await renderBar('project-a');
    await openMenu(marker());
    expect(menuItem('dock-widget')).not.toBeNull();

    // Project B has a floating window with the same instance id, as default widgets do.
    await renderBar('project-b');
    expect(menuItem('dock-widget')).toBeNull();
    expect(menuItem('remove-widget')).toBeNull();

    // Coming back to project A does not bring the old menu back.
    await renderBar('project-a');
    expect(menuItem('dock-widget')).toBeNull();
  });

  it('offers no Dock item for a docked tab', async () => {
    await renderBar();
    await openMenu(slot('Queue'));

    expect(menuItem('remove-widget')).not.toBeNull();
    expect(menuItem('dock-widget')).toBeNull();
  });
});

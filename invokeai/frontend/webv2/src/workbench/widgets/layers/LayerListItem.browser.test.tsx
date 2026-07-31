/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';

import { ChakraProvider } from '@chakra-ui/react';
import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import { arrayMove, SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { LAYER_KEYBOARD_SENSOR_OPTIONS } from './layerDndConfig';
import { LayerListItem, type LayerListItemEngine } from './LayerListItem';
import { createEmptyPaintLayer } from './layerOps';

vi.mock('./ControlLayerWarningIcon', () => ({
  ControlLayerWarningIcon: () => null,
}));

vi.mock('@workbench/widgets/canvas/engineStoreHooks', () => ({
  useLayerThumbnailStatus: () => 'error',
  useLayerThumbnailVersion: () => 0,
}));

vi.mock('./LayerPropertiesPopover', () => ({
  LayerPropertiesPopover: () => <button aria-label="Layer properties" type="button" />,
}));

vi.mock('./LayerContextMenu', () => ({
  CanvasLayerContextMenu: () => null,
  LayerContextMenu: () => <button aria-label="Layer menu" type="button" />,
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          layers: {
            actions: {
              reorder: 'Reorder layer',
              select: 'Select {{name}}',
              toggleLock: 'Toggle lock',
              toggleVisibility: 'Toggle visibility',
            },
            types: { paint: 'Paint' },
          },
        },
      },
    },
  },
});

const INITIAL_LAYERS = [createEmptyPaintLayer('First layer', 'first'), createEmptyPaintLayer('Second layer', 'second')];
const requestLayerThumbnail = vi.fn();

const Harness = () => {
  const [layers, setLayers] = useState(INITIAL_LAYERS);
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(null);
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, LAYER_KEYBOARD_SENSOR_OPTIONS)
  );
  const layerIds = useMemo(() => layers.map((layer) => layer.id), [layers]);
  const dispatch = useCallback((mutation: CanvasProjectMutation) => {
    if (mutation.type === 'setCanvasSelectedLayer') {
      setSelectedLayerId(mutation.id);
    } else if (mutation.type === 'updateCanvasLayer' && mutation.patch.isEnabled !== undefined) {
      setLayers((current) =>
        current.map((layer) =>
          layer.id === mutation.id ? { ...layer, isEnabled: mutation.patch.isEnabled ?? layer.isEnabled } : layer
        )
      );
    }
  }, []);
  const engine = useMemo(
    () =>
      ({
        layers: {
          commitStructural: (_label: string, mutation: CanvasProjectMutation) => dispatch(mutation),
        },
        previews: {
          drawLayerThumbnail: () => false,
          requestLayerThumbnail,
        },
        projectId: 'test-project',
      }) as unknown as LayerListItemEngine,
    [dispatch]
  );
  const handleDragEnd = useCallback((event: DragEndEvent) => {
    if (!event.over || event.active.id === event.over.id) {
      return;
    }
    setLayers((current) => {
      const from = current.findIndex((layer) => layer.id === event.active.id);
      const to = current.findIndex((layer) => layer.id === event.over?.id);
      return from === -1 || to === -1 ? current : arrayMove(current, from, to);
    });
  }, []);

  return (
    <DndContext collisionDetection={closestCenter} sensors={sensors} onDragEnd={handleDragEnd}>
      <SortableContext items={layerIds} strategy={verticalListSortingStrategy}>
        {layers.map((layer, index) => (
          <LayerListItem
            key={layer.id}
            dispatch={dispatch}
            editingLocked={false}
            engine={engine}
            index={index}
            isSelected={selectedLayerId === layer.id}
            layer={layer}
            layers={layers}
          />
        ))}
      </SortableContext>
      <output data-testid="selected-layer">{selectedLayerId ?? 'none'}</output>
      <output data-testid="layer-order">{layers.map((layer) => layer.id).join(',')}</output>
    </DndContext>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderHarness = async () => {
  host = document.createElement('div');
  host.style.width = '480px';
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};

const selectionButton = (name: string): HTMLButtonElement =>
  host!.querySelector<HTMLButtonElement>(`button[aria-label="Select ${name}"]`)!;

const selectedLayer = (): string => host!.querySelector<HTMLOutputElement>('[data-testid="selected-layer"]')!.value;
const layerOrder = (): string => host!.querySelector<HTMLOutputElement>('[data-testid="layer-order"]')!.value;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

describe('LayerListItem accessibility', () => {
  it('keeps the sortable control free of interactive descendants', async () => {
    await renderHarness();

    const sortableControl = host!.querySelector<HTMLElement>('[aria-roledescription="sortable"]');
    expect(sortableControl).not.toBeNull();
    expect(
      sortableControl!.querySelector('button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])')
    ).toBeNull();
  });

  it('gives the compact visibility control a 24px touch target', async () => {
    await renderHarness();

    const visibility = host!.querySelector<HTMLButtonElement>('button[aria-label="Toggle visibility"]');
    expect(visibility).not.toBeNull();
    const rect = visibility!.getBoundingClientRect();
    expect(rect.width).toBeGreaterThanOrEqual(24);
    expect(rect.height).toBeGreaterThanOrEqual(24);
  });

  it('selects a layer with a pointer without requiring the drag handle', async () => {
    await renderHarness();

    await act(() => userEvent.click(selectionButton('First layer')));

    expect(selectedLayer()).toBe('first');
  });

  it.each([
    ['Enter', '{Enter}'],
    ['Space', ' '],
  ])('selects a layer with %s on the selection control', async (_label, key) => {
    await renderHarness();
    selectionButton('Second layer').focus();

    await act(() => userEvent.keyboard(key));

    expect(selectedLayer()).toBe('second');
    expect(layerOrder()).toBe('first,second');
  });

  it('keeps visibility toggling isolated from row selection', async () => {
    await renderHarness();
    const visibility = host!.querySelectorAll<HTMLButtonElement>('button[aria-label="Toggle visibility"]')[0]!;

    await act(() => userEvent.click(visibility));

    expect(visibility).toHaveAttribute('aria-pressed', 'false');
    expect(selectedLayer()).toBe('none');
  });

  it('keeps thumbnail retry focus, click, and pointer movement isolated from row selection and sorting', async () => {
    await renderHarness();
    const retry = host!.querySelector<HTMLButtonElement>('button[aria-label="Retry thumbnail for First layer"]')!;
    const second = selectionButton('Second layer');
    retry.focus();
    expect(document.activeElement).toBe(retry);

    await act(() => userEvent.click(retry));
    expect(requestLayerThumbnail).toHaveBeenCalledWith('first');
    expect(selectedLayer()).toBe('none');

    const retryRect = retry.getBoundingClientRect();
    const secondRect = second.getBoundingClientRect();
    const startX = retryRect.left + retryRect.width / 2;
    const startY = retryRect.top + retryRect.height / 2;
    const endX = secondRect.left + secondRect.width / 2;
    const endY = secondRect.top + secondRect.height / 2;

    await act(() => pointer('pointerdown', retry, startX, startY));
    await act(() => pointer('pointermove', retry.ownerDocument, startX + 8, startY));
    await act(() => pointer('pointermove', retry.ownerDocument, endX, endY));
    await act(() => pointer('pointerup', retry.ownerDocument, endX, endY));

    expect(layerOrder()).toBe('first,second');
    expect(selectedLayer()).toBe('none');
  });

  it('preserves whole-row pointer sorting outside the dedicated handle', async () => {
    await renderHarness();
    const first = selectionButton('First layer');
    const second = selectionButton('Second layer');
    const firstRect = first.getBoundingClientRect();
    const secondRect = second.getBoundingClientRect();
    const startX = firstRect.left + firstRect.width / 2;
    const startY = firstRect.top + firstRect.height / 2;
    const endX = secondRect.left + secondRect.width / 2;
    const endY = secondRect.top + secondRect.height / 2;

    await act(() => pointer('pointerdown', first, startX, startY));
    await act(() => pointer('pointermove', first.ownerDocument, startX + 8, startY));
    await act(() => pointer('pointermove', first.ownerDocument, endX, endY));
    await act(() => pointer('pointerup', first.ownerDocument, endX, endY));

    expect(layerOrder()).toBe('second,first');
  });

  // With no grip there is no keyboard drag gesture to claim Enter, so the
  // Enter/Arrow/Enter sequence that used to reorder now moves nothing. Enter
  // still selects (covered above); keyboard reordering is the context menu's
  // Move actions.
  it('no longer starts a keyboard drag from a focused row', async () => {
    await renderHarness();
    selectionButton('First layer').focus();

    await act(() => userEvent.keyboard('{Enter}'));
    await act(() => userEvent.keyboard('{ArrowDown}'));
    await act(() => userEvent.keyboard('{Enter}'));

    expect(layerOrder()).toBe('first,second');
  });
});

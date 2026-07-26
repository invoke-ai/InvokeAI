import { ChakraProvider } from '@chakra-ui/react';
import { DndContext, KeyboardSensor, PointerSensor, useSensor, useSensors, type DragEndEvent } from '@dnd-kit/core';
import {
  arrayMove,
  horizontalListSortingStrategy,
  sortableKeyboardCoordinates,
  SortableContext,
} from '@dnd-kit/sortable';
import { Tabs } from '@platform/ui';
import { system } from '@theme/system';
import { useWidgetSortable } from '@workbench/widget-frame/useWidgetSortable';
import { getWidgetInstanceDragId } from '@workbench/widgetDnd';
import { act, useCallback, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';

const TAB_IDS = ['first', 'second'] as const;

const getPanelId = (id: string): string => `test-center-panel-${id}`;

const SortableTab = ({ id }: { id: (typeof TAB_IDS)[number] }) => {
  const { isDragging, semanticDragHandleProps, setNodeRef, style } = useWidgetSortable({
    instanceId: id,
    region: 'center',
    typeId: 'preview',
  });

  return (
    <Tabs.Trigger
      ref={setNodeRef}
      aria-controls={getPanelId(id)}
      data-dragging={isDragging ? '' : undefined}
      style={style}
      value={id}
      {...semanticDragHandleProps}
    >
      {id}
    </Tabs.Trigger>
  );
};

const Harness = () => {
  const [value, setValue] = useState<(typeof TAB_IDS)[number]>('first');
  const [order, setOrder] = useState<(typeof TAB_IDS)[number][]>([...TAB_IDS]);
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );
  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const from = order.findIndex((id) => getWidgetInstanceDragId('center', id) === event.active.id);
      const to = order.findIndex((id) => getWidgetInstanceDragId('center', id) === event.over?.id);

      if (from !== -1 && to !== -1 && from !== to) {
        setOrder(arrayMove(order, from, to));
      }
    },
    [order]
  );
  const sortableIds = useMemo(() => order.map((id) => getWidgetInstanceDragId('center', id)), [order]);
  const handleValueChange = useCallback((details: { value: string }) => setValue(details.value as typeof value), []);

  return (
    <DndContext sensors={sensors} onDragEnd={handleDragEnd}>
      <SortableContext items={sortableIds} strategy={horizontalListSortingStrategy}>
        <Tabs.Root value={value} onValueChange={handleValueChange}>
          <Tabs.List aria-label="Center views">
            {order.map((id) => (
              <SortableTab key={id} id={id} />
            ))}
          </Tabs.List>
          {TAB_IDS.map((id) => (
            <Tabs.Content key={id} id={getPanelId(id)} value={id}>
              {id} panel
            </Tabs.Content>
          ))}
        </Tabs.Root>
      </SortableContext>
    </DndContext>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const dispatchKey = async (target: HTMLElement, key: string, code: string) => {
  await act(async () => {
    target.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, code, key }));
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

describe('sortable center tabs', () => {
  it('preserves tab navigation, activation, and controlled-panel relationships', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(() => {
      root?.render(
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      );
    });

    const tabs = Array.from(host.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
    expect(tabs).toHaveLength(2);

    for (const tab of tabs) {
      expect(tab.hasAttribute('aria-pressed')).toBe(false);
      const panelId = tab.getAttribute('aria-controls');
      expect(panelId).toBeTruthy();
      expect(document.getElementById(panelId!)).toHaveAttribute('role', 'tabpanel');
    }

    expect(tabs[0]).toHaveAttribute('tabindex', '0');
    await act(() => tabs[0]!.focus());
    await dispatchKey(tabs[0]!, 'ArrowRight', 'ArrowRight');
    expect(document.activeElement).toBe(tabs[1]);
    expect(tabs[1]).toHaveAttribute('aria-selected', 'true');

    await dispatchKey(tabs[1]!, 'Enter', 'Enter');
    expect(tabs[1]).toHaveAttribute('aria-selected', 'true');

    await dispatchKey(tabs[1]!, 'ArrowLeft', 'ArrowLeft');
    expect(document.activeElement).toBe(tabs[0]);
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');

    // Space → Arrow → Space is dnd-kit's keyboard reorder gesture. It must
    // preserve the tab role/state while moving the item.
    await dispatchKey(tabs[0]!, ' ', 'Space');
    expect(tabs[0]).toHaveAttribute('data-dragging');
    await dispatchKey(tabs[0]!, 'ArrowRight', 'ArrowRight');
    await dispatchKey(tabs[0]!, ' ', 'Space');
    const reorderedTabs = Array.from(host.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
    expect(reorderedTabs.map((tab) => tab.textContent)).toEqual(['second', 'first']);
    expect(reorderedTabs[1]).toHaveAttribute('role', 'tab');
    expect(reorderedTabs[1]!.hasAttribute('aria-pressed')).toBe(false);
  });
});

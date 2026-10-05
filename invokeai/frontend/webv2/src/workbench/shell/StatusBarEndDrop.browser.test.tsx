/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { DragEndEvent, DragOverEvent } from '@dnd-kit/core';

import { Box, ChakraProvider } from '@chakra-ui/react';
import { DndContext, useDroppable, useSensor, useSensors } from '@dnd-kit/core';
import { restrictToWindowEdges } from '@dnd-kit/modifiers';
import { horizontalListSortingStrategy } from '@dnd-kit/sortable';
import { system } from '@theme/system';
import { WidgetStrip, useWidgetSortable } from '@workbench/widget-frame';
import { getWidgetRegionEndDropData, getWidgetRegionEndDropId, widgetCollisionDetection } from '@workbench/widgetDnd';
import { act, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { PrimaryMouseSensor } from './holdToDragSensor';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Active, as during a real drag — the drop overlay renders over the strip.
const DROP_STATE = { helperText: 'Drop here', isActive: true, isAllowed: true };
const MODIFIERS = [restrictToWindowEdges];
const SORTABLE_IDS = ['alpha', 'beta'];

const Chip = ({ id }: { id: string }) => {
  const { dragHandleProps, setNodeRef, style } = useWidgetSortable({ instanceId: id, region: 'bottom', typeId: id });

  return (
    <Box ref={setNodeRef} data-chip={id} h="6" px="2" style={style} {...dragHandleProps}>
      {id}
    </Box>
  );
};

const EndZone = () => {
  const { setNodeRef } = useDroppable({
    data: getWidgetRegionEndDropData('bottom'),
    id: getWidgetRegionEndDropId('bottom'),
  });

  return <Box ref={setNodeRef} alignSelf="stretch" data-endzone flex="1" />;
};

const Harness = ({
  onDragCancel,
  onDragEnd,
  onDragOver,
  onDragStart,
}: {
  onDragCancel: () => void;
  onDragEnd: (event: DragEndEvent) => void;
  onDragOver: (id: string) => void;
  onDragStart: () => void;
}) => {
  // The shell's mouse sensor (holdToDragSensor.ts), same activation.
  const sensors = useSensors(useSensor(PrimaryMouseSensor, { activationConstraint: { distance: 6 } }));
  const handleDragOver = useCallback(
    (event: DragOverEvent) => onDragOver(String(event.over?.id ?? 'none')),
    [onDragOver]
  );

  return (
    <ChakraProvider value={system}>
      <DndContext
        collisionDetection={widgetCollisionDetection}
        modifiers={MODIFIERS}
        sensors={sensors}
        onDragEnd={onDragEnd}
        onDragCancel={onDragCancel}
        onDragOver={handleDragOver}
        onDragStart={onDragStart}
      >
        <WidgetStrip
          align="center"
          dropState={DROP_STATE}
          h="6"
          px="2"
          region="bottom"
          sortableInstanceIds={SORTABLE_IDS}
          strategy={horizontalListSortingStrategy}
          w="600px"
        >
          <Chip id="alpha" />
          <Chip id="beta" />
          <EndZone />
        </WidgetStrip>
      </DndContext>
    </ChakraProvider>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('status bar end-zone drop (real dnd pipeline)', () => {
  it('reports the end zone as the drop target for a pointer over the spacer', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    const events: DragEndEvent[] = [];
    let started = 0;
    let cancelled = 0;
    const overs: string[] = [];

    await act(async () => {
      const handleCancel = () => {
        cancelled += 1;
      };
      const handleEnd = (event: DragEndEvent) => {
        events.push(event);
      };
      const handleOver = (id: string) => {
        overs.push(id);
      };
      const handleStart = () => {
        started += 1;
      };

      root?.render(
        <Harness onDragCancel={handleCancel} onDragEnd={handleEnd} onDragOver={handleOver} onDragStart={handleStart} />
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const chip = host.querySelector<HTMLElement>('[data-chip="alpha"]')!;
    const zone = host.querySelector<HTMLElement>('[data-endzone]')!;

    // Deliberately OUTSIDE act(): wrapping the gesture starves dnd-kit's
    // effect-driven onDragOver/onDragEnd and reproduces the exact silent
    // failure this test exists to catch.
    await userEvent.dragAndDrop(chip, zone);

    const startedAt = Date.now();

    while (events.length === 0 && Date.now() - startedAt < 2000) {
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 25);
      });
    }

    expect(started).toBe(1);
    expect(cancelled).toBe(0);
    expect(overs).toContain(getWidgetRegionEndDropId('bottom'));
    expect(events).toHaveLength(1);
    expect(String(events[0]!.over?.id ?? 'none')).toBe(getWidgetRegionEndDropId('bottom'));
  });
});

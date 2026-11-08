/**
 * Drag between element centers the way a hand moves a mouse, for dnd-kit mouse sensors. Droppables that enable only
 * once a drag is active are measured on later moves, so a single-jump drag (`userEvent.dragAndDrop`) ends over
 * nothing. Call outside act: dnd-kit settles the drop in effects that act would hold back.
 */

// The pointer arrives twice so the target, measured after the first arrival, sees a move over it.
const PROGRESS = [0.25, 0.5, 0.75, 1, 1];
const STEP_DELAY_MS = 30;

const center = (element: Element) => {
  const rect = element.getBoundingClientRect();

  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
};

const wait = () =>
  new Promise<void>((resolve) => {
    globalThis.setTimeout(resolve, STEP_DELAY_MS);
  });

export const dragWithMouse = async (source: Element, target: Element): Promise<void> => {
  const from = center(source);
  const to = center(target);
  const mouse = (type: string, x: number, y: number, element: EventTarget = source.ownerDocument) =>
    element.dispatchEvent(new MouseEvent(type, { bubbles: true, button: 0, clientX: x, clientY: y }));

  mouse('mousedown', from.x, from.y, source);
  for (const progress of PROGRESS) {
    mouse('mousemove', from.x + (to.x - from.x) * progress, from.y + (to.y - from.y) * progress);
    await wait();
  }
  mouse('mouseup', to.x, to.y);
  await wait();
};

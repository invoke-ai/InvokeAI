/** Gap between the anchor and the card. */
export const HINT_GUTTER_PX = 8;
/** Clearance the card keeps from the viewport edge. */
export const HINT_VIEWPORT_PADDING_PX = 8;
/** The corridor's apex sits this far behind the exit point, so a little jitter on leaving does not close the card. */
const CORRIDOR_SLACK_PX = 6;

/** Marks the field a hint explains; the card sits beside it when no scrolling panel can hold it. */
export const FEATURE_HINT_FIELD_PROPS = { 'data-feature-hint-field': '' } as const;

const FIELD_SELECTOR = '[data-feature-hint-field]';

type Point = { x: number; y: number };
type AnchorRect = { height: number; width: number; x: number; y: number };

/** The nearest vertically scrolling ancestor: the body of the panel, window, or dialog the hint lives in. */
export const findScrollingPanel = (trigger: HTMLElement): HTMLElement | null => {
  for (let element = trigger.parentElement; element && element !== document.body; element = element.parentElement) {
    const { overflowY } = getComputedStyle(element);

    if (overflowY === 'auto' || overflowY === 'scroll') {
      return element;
    }
  }

  return null;
};

/**
 * Spans the panel across and the hovered label down, so the card points at the label from beside the panel instead
 * of covering the controls it explains. When neither side of the panel has room for the card (a maximized window),
 * the marked field stands in, then the label itself, so the card always stays on screen.
 */
export const getHintAnchorRect = (
  trigger: HTMLElement,
  panel: HTMLElement | null,
  card: HTMLElement | null
): AnchorRect => {
  const label = trigger.getBoundingClientRect();
  // Zero until the card has laid out; the first measurement then repositions it.
  const cardWidth = card?.getBoundingClientRect().width ?? 0;
  const viewportWidth = document.documentElement.clientWidth;
  const hasRoomBeside = (rect: DOMRect): boolean =>
    Math.max(rect.left, viewportWidth - rect.right) - HINT_GUTTER_PX - HINT_VIEWPORT_PADDING_PX >= cardWidth;
  const across =
    [panel, trigger.closest(FIELD_SELECTOR)]
      .map((element) => element?.getBoundingClientRect())
      .find((rect): rect is DOMRect => rect !== undefined && hasRoomBeside(rect)) ?? label;

  return { height: label.height, width: across.width, x: across.left, y: label.top };
};

const cross = (origin: Point, a: Point, b: Point): number =>
  (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);

/**
 * The safe triangle from where the pointer left the label to the card's near edge: moves inside it are on their way
 * to the card. Null when the card is not beside the label, where the ordinary close delay applies.
 */
export const createGraceCorridor = (exit: Point, label: DOMRect, card: DOMRect): ((point: Point) => boolean) | null => {
  const side = card.left >= label.right ? 1 : card.right <= label.left ? -1 : 0;

  if (side === 0) {
    return null;
  }

  const edgeX = side === 1 ? card.left : card.right;
  const apex = { x: exit.x - side * CORRIDOR_SLACK_PX, y: exit.y };
  const top = { x: edgeX, y: card.top };
  const bottom = { x: edgeX, y: card.bottom };

  return (point) => {
    const a = cross(apex, top, point);
    const b = cross(top, bottom, point);
    const c = cross(bottom, apex, point);

    return (a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0);
  };
};

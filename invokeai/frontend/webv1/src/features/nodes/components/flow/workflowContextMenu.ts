import { NO_DRAG_CLASS, NO_PAN_CLASS, NO_WHEEL_CLASS } from 'features/nodes/types/constants';

export type WorkflowContextMenuState =
  | {
      kind: 'pane';
      clientX: number;
      clientY: number;
      pageX: number;
      pageY: number;
    }
  | {
      kind: 'connector';
      connectorId: string;
      pageX: number;
      pageY: number;
    }
  | null;

const INTERACTIVE_CONTEXT_MENU_SELECTOR = [
  'input',
  'textarea',
  'select',
  'option',
  'button',
  'a[href]',
  '[contenteditable="true"]',
  '[contenteditable=""]',
  '[role="textbox"]',
  `.${NO_DRAG_CLASS}`,
  `.${NO_PAN_CLASS}`,
  `.${NO_WHEEL_CLASS}`,
  '.react-flow__node',
].join(',');

const isElementLike = (target: EventTarget | null): target is Element =>
  typeof target === 'object' &&
  target !== null &&
  'closest' in target &&
  typeof (target as { closest?: unknown }).closest === 'function';

export const getWorkflowContextMenuState = (
  event: Pick<globalThis.MouseEvent, 'shiftKey' | 'target' | 'clientX' | 'clientY' | 'pageX' | 'pageY'>,
  flowWrapper: Pick<HTMLDivElement, 'contains'> | null
): WorkflowContextMenuState => {
  if (event.shiftKey || !isElementLike(event.target) || !flowWrapper?.contains(event.target)) {
    return null;
  }

  const connectorId = event.target.closest<HTMLElement>('[data-connector-node-id]')?.dataset.connectorNodeId;
  if (connectorId) {
    return {
      kind: 'connector',
      connectorId,
      pageX: event.pageX,
      pageY: event.pageY,
    };
  }

  if (event.target.closest(INTERACTIVE_CONTEXT_MENU_SELECTOR)) {
    return null;
  }

  const paneTarget = event.target.closest('.react-flow__pane');
  if (paneTarget && flowWrapper.contains(paneTarget)) {
    return {
      kind: 'pane',
      clientX: event.clientX,
      clientY: event.clientY,
      pageX: event.pageX,
      pageY: event.pageY,
    };
  }

  return null;
};

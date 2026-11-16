import type { SyntheticEvent } from 'react';

/**
 * Whether an event started inside the handling element's own DOM. React bubbles events along the component tree, so
 * one from portalled content a descendant opened (a popover, menu, or tooltip) also reaches ancestors whose DOM does
 * not hold it. A handler acting for its own surface, such as a region's right-click menu, ignores those.
 */
export const isEventFromOwnDom = (event: SyntheticEvent<Element>): boolean =>
  event.target instanceof Node && event.currentTarget.contains(event.target);

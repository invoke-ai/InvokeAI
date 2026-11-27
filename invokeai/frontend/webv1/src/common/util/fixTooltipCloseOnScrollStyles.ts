import type { CSSProperties } from 'react';

/**
 * Chakra's Tooltip's method of finding the nearest scroll parent has a problem - it assumes the first parent with
 * `overflow: hidden` is the scroll parent. In this case, the Collapse component has that style, but isn't scrollable
 * itself. The result is that the tooltip does not close on scroll, because the scrolling happens higher up in the DOM.
 *
 * As a hacky workaround, we can set the overflow to `visible`, which allows the scroll parent search to continue up to
 * the actual scroll parent (in this case, the OverlayScrollbarsComponent in BoardsListWrapper).
 *
 * See: https://github.com/chakra-ui/chakra-ui/issues/7871#issuecomment-2453780958
 */
export const fixTooltipCloseOnScrollStyles: CSSProperties = {
  overflow: 'visible',
};

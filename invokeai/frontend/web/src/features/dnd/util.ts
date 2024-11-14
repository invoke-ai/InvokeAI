import type { GetOffsetFn } from '@atlaskit/pragmatic-drag-and-drop/dist/types/public-utils/element/custom-native-drag-preview/types';
import type { Input } from '@atlaskit/pragmatic-drag-and-drop/types';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { noop } from 'lodash-es';
import type { CSSProperties } from 'react';

/**
 * The size of the image drag preview in theme units.
 */
export const DND_IMAGE_DRAG_PREVIEW_SIZE = 32 satisfies SystemStyleObject['w'];

/**
 * A drag preview offset function that works like the provided `preserveOffsetOnSource`, except when either the X or Y
 * offset is outside the container, in which case it centers the preview in the container.
 */
export function preserveOffsetOnSourceFallbackCentered({
  element,
  input,
}: {
  element: HTMLElement;
  input: Input;
}): GetOffsetFn {
  return ({ container }) => {
    const sourceRect = element.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    let offsetX = input.clientX - sourceRect.x;
    let offsetY = input.clientY - sourceRect.y;

    if (offsetY > containerRect.height || offsetX > containerRect.width) {
      offsetX = containerRect.width / 2;
      offsetY = containerRect.height / 2;
    }

    return { x: offsetX, y: offsetY };
  };
}

// Based on https://github.com/atlassian/pragmatic-drag-and-drop/blob/main/packages/flourish/src/trigger-post-move-flash.tsx
// That package has a lot of extra deps so we just copied the function here
export function triggerPostMoveFlash(element: HTMLElement, backgroundColor: CSSProperties['backgroundColor']) {
  element.animate([{ backgroundColor }, {}], {
    duration: 700,
    easing: 'cubic-bezier(0.25, 0.1, 0.25, 1.0)',
    iterations: 1,
  });
}

/**
 * Firefox has a bug where input or textarea elements with draggable parents do not allow selection of their text.
 *
 * This helper function implements a workaround by setting the draggable attribute to false when the mouse is over a
 * input or textarea child of the draggable. It reverts the attribute on mouse out.
 *
 * The fix is only applied for Firefox, and should be used in every `pragmatic-drag-and-drop` `draggable`.
 *
 * See:
 * - https://github.com/atlassian/pragmatic-drag-and-drop/issues/111
 * - https://bugzilla.mozilla.org/show_bug.cgi?id=1853069
 *
 * @example
 * ```tsx
 * useEffect(() => {
 *   const element = ref.current;
 *   if (!element) {
 *     return;
 *   }
 *   return combine(
 *     firefoxDndFix(element),
 *     // The rest of the draggable setup is the same
 *     draggable({
 *       element,
 *       // ...
 *     }),
 *   );
 *```
 * @param element The draggable element
 * @returns A cleanup function that removes the event listeners
 */
export const firefoxDndFix = (element: HTMLElement): (() => void) => {
  if (!navigator.userAgent.includes('Firefox')) {
    return noop;
  }

  const abortController = new AbortController();

  element.addEventListener(
    'mouseover',
    (event) => {
      if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) {
        element.setAttribute('draggable', 'false');
      }
    },
    { signal: abortController.signal }
  );

  element.addEventListener(
    'mouseout',
    (event) => {
      if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) {
        element.setAttribute('draggable', 'true');
      }
    },
    { signal: abortController.signal }
  );

  return () => {
    element.setAttribute('draggable', 'true');
    abortController.abort();
  };
};

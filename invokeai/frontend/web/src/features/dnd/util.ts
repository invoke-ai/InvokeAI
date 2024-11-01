import type { GetOffsetFn } from '@atlaskit/pragmatic-drag-and-drop/dist/types/public-utils/element/custom-native-drag-preview/types';
import type { Input } from '@atlaskit/pragmatic-drag-and-drop/types';
import type { SystemStyleObject } from '@invoke-ai/ui-library';
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

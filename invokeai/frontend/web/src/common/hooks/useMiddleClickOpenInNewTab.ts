import { useAppSelector } from 'app/store/storeHooks';
import { openImageInNewTab } from 'common/util/openImageInNewTab';
import { selectSystemShouldUseMiddleClickToOpenInNewTab } from 'features/system/store/systemSlice';
import type { RefObject } from 'react';
import { useEffect } from 'react';

type Options = {
  requireDirectTarget?: boolean;
};

const shouldHandleMiddleClick = <T extends HTMLElement>(
  event: MouseEvent,
  element: T,
  requireDirectTarget: boolean
) => {
  if (event.button !== 1) {
    return false;
  }

  if (requireDirectTarget && event.target !== element) {
    return false;
  }

  return true;
};

export const useMiddleClickOpenInNewTab = <T extends HTMLElement = HTMLElement>(
  ref: RefObject<T>,
  imageUrl: string,
  { requireDirectTarget = false }: Options = {}
) => {
  const shouldUseMiddleClickToOpenInNewTab = useAppSelector(selectSystemShouldUseMiddleClickToOpenInNewTab);

  useEffect(() => {
    const element = ref.current;

    if (!element || !shouldUseMiddleClickToOpenInNewTab) {
      return;
    }

    // If auxclick is unsupported, leave the browser's default middle-click behavior intact.
    if (!('onauxclick' in element)) {
      return;
    }

    const onMouseDown = (event: MouseEvent) => {
      if (!shouldHandleMiddleClick(event, element, requireDirectTarget)) {
        return;
      }

      event.preventDefault();
    };

    const onAuxClick = (event: MouseEvent) => {
      if (!shouldHandleMiddleClick(event, element, requireDirectTarget)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      openImageInNewTab(imageUrl);
    };

    element.addEventListener('mousedown', onMouseDown);
    element.addEventListener('auxclick', onAuxClick);

    return () => {
      element.removeEventListener('mousedown', onMouseDown);
      element.removeEventListener('auxclick', onAuxClick);
    };
  }, [imageUrl, ref, requireDirectTarget, shouldUseMiddleClickToOpenInNewTab]);
};

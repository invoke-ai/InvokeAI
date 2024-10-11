import type { ChakraProps } from '@invoke-ai/ui-library';
import { Menu, MenuButton, MenuList, Portal, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import MultipleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/MultipleSelectionMenuItems';
import SingleSelectionMenuItems from 'features/gallery/components/ImageContextMenu/SingleSelectionMenuItems';
import { selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { map } from 'nanostores';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef } from 'react';
import type { ImageDTO } from 'services/api/types';

const LONGPRESS_DELAY_MS = 500;
const LONGPRESS_MOVE_THRESHOLD_PX = 10;

type ImageContextMenuState = {
  isOpen: boolean;
  imageDTO: ImageDTO | null;
  position: { x: number; y: number };
};

const $imageContextMenuState = map<ImageContextMenuState>({
  isOpen: false,
  imageDTO: null,
  position: { x: -1, y: -1 },
});
const onClose = () => {
  $imageContextMenuState.setKey('isOpen', false);
};
const elToImageMap = new Map<HTMLDivElement, ImageDTO>();
const getImageDTOFromMap = (target: Node): ImageDTO | undefined => {
  const entry = Array.from(elToImageMap.entries()).find((entry) => entry[0].contains(target));
  return entry?.[1];
};
export const useImageContextMenu = (imageDTO: ImageDTO | undefined, targetRef: RefObject<HTMLDivElement>) => {
  useEffect(() => {
    if (!targetRef.current || !imageDTO) {
      return;
    }
    const el = targetRef.current;
    elToImageMap.set(el, imageDTO);
    return () => {
      elToImageMap.delete(el);
    };
  }, [imageDTO, targetRef]);
};

export const ImageContextMenu = memo(() => {
  useAssertSingleton('ImageContextMenu');
  const state = useStore($imageContextMenuState);
  useGlobalMenuClose(onClose);

  return (
    <Portal>
      <Menu isOpen={state.isOpen} gutter={0} placement="auto-end" onClose={onClose}>
        <MenuButton
          aria-hidden={true}
          w={1}
          h={1}
          position="absolute"
          left={state.position.x}
          top={state.position.y}
          cursor="default"
          bg="transparent"
          _hover={_hover}
          pointerEvents="none"
        />
        <MenuContent />
      </Menu>
      <ImageContextMenuEventLogical />
    </Portal>
  );
});

ImageContextMenu.displayName = 'ImageContextMenu';

const _hover: ChakraProps['_hover'] = { bg: 'transparent' };

const ImageContextMenuEventLogical = memo(() => {
  const lastPositionRef = useRef<{ x: number; y: number }>({ x: -1, y: -1 });
  const longPressTimeoutRef = useRef(0);
  const animationTimeoutRef = useRef(0);

  const onContextMenu = useCallback((e: MouseEvent | PointerEvent) => {
    if (e.shiftKey) {
      onClose();
      return;
    }

    const imageDTO = getImageDTOFromMap(e.target as Node);

    if (!imageDTO) {
      onClose();
      return;
    }
    // clear pending delayed open
    window.clearTimeout(animationTimeoutRef.current);
    e.preventDefault();

    if (lastPositionRef.current.x !== e.pageX || lastPositionRef.current.y !== e.pageY) {
      // if the mouse moved, we need to close, wait for animation and reopen the menu at the new position
      if ($imageContextMenuState.get().isOpen) {
        onClose();
      }
      animationTimeoutRef.current = window.setTimeout(() => {
        $imageContextMenuState.set({
          isOpen: true,
          position: { x: e.pageX, y: e.pageY },
          imageDTO,
        });
      }, 100);
    } else {
      // else we can just open the menu at the current position
      $imageContextMenuState.set({
        isOpen: true,
        position: { x: e.pageX, y: e.pageY },
        imageDTO,
      });
    }

    lastPositionRef.current = { x: e.pageX, y: e.pageY };
  }, []);

  // Use a long press to open the context menu on touch devices
  const onPointerDown = useCallback(
    (e: PointerEvent) => {
      if (e.pointerType === 'mouse') {
        // Bail out if it's a mouse event - this is for touch/pen only
        return;
      }

      longPressTimeoutRef.current = window.setTimeout(() => {
        onContextMenu(e);
      }, LONGPRESS_DELAY_MS);

      lastPositionRef.current = { x: e.pageX, y: e.pageY };
    },
    [onContextMenu]
  );

  const onPointerMove = useCallback((e: PointerEvent) => {
    if (e.pointerType === 'mouse') {
      // Bail out if it's a mouse event - this is for touch/pen only
      return;
    }
    if (longPressTimeoutRef.current === null) {
      return;
    }

    const lastPosition = lastPositionRef.current;

    const distanceFromLastPosition = Math.hypot(e.pageX - lastPosition.x, e.pageY - lastPosition.y);

    if (distanceFromLastPosition > LONGPRESS_MOVE_THRESHOLD_PX) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  const onPointerUp = useCallback((e: PointerEvent) => {
    if (e.pointerType === 'mouse') {
      // Bail out if it's a mouse event - this is for touch/pen only
      return;
    }
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  const onPointerCancel = useCallback((e: PointerEvent) => {
    if (e.pointerType === 'mouse') {
      // Bail out if it's a mouse event - this is for touch/pen only
      return;
    }
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    // Context menu events
    window.addEventListener('contextmenu', onContextMenu, { signal: controller.signal });

    // Long press events
    window.addEventListener('pointerdown', onPointerDown, { signal: controller.signal });
    window.addEventListener('pointerup', onPointerUp, { signal: controller.signal });
    window.addEventListener('pointercancel', onPointerCancel, { signal: controller.signal });
    window.addEventListener('pointermove', onPointerMove, { signal: controller.signal });

    return () => {
      controller.abort();
    };
  }, [onContextMenu, onPointerCancel, onPointerDown, onPointerMove, onPointerUp]);

  useEffect(
    () => () => {
      window.clearTimeout(animationTimeoutRef.current);
      window.clearTimeout(longPressTimeoutRef.current);
    },
    []
  );

  return null;
});

ImageContextMenuEventLogical.displayName = 'ImageContextMenuEventLogical';

const MenuContent = memo(() => {
  const selectionCount = useAppSelector(selectSelectionCount);
  const state = useStore($imageContextMenuState);

  if (!state.imageDTO) {
    return null;
  }

  if (selectionCount > 1) {
    return (
      <MenuList visibility="visible">
        <MultipleSelectionMenuItems />
      </MenuList>
    );
  }

  return (
    <MenuList visibility="visible">
      <SingleSelectionMenuItems imageDTO={state.imageDTO} />
    </MenuList>
  );
});

MenuContent.displayName = 'MenuContent';

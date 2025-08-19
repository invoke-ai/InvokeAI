import type { ChakraProps } from '@invoke-ai/ui-library';
import { Menu, MenuButton, MenuList, Portal, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import MultipleSelectionVideoMenuItems from 'features/gallery/components/ContextMenu/MultipleSelectionVideoMenuItems';
import SingleSelectionVideoMenuItems from 'features/gallery/components/ContextMenu/SingleSelectionVideoMenuItems';
import { selectSelectionCount } from 'features/gallery/store/gallerySelectors';
import { map } from 'nanostores';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef } from 'react';
import type { VideoDTO } from 'services/api/types';

/**
 * The delay in milliseconds before the context menu opens on long press.
 */
const LONGPRESS_DELAY_MS = 500;
/**
 * The threshold in pixels that the pointer must move before the long press is cancelled.
 */
const LONGPRESS_MOVE_THRESHOLD_PX = 10;

/**
 * The singleton state of the context menu.
 */
const $videoContextMenuState = map<{
  isOpen: boolean;
  videoDTO: VideoDTO | null;
  position: { x: number; y: number };
}>({
  isOpen: false,
  videoDTO: null,
  position: { x: -1, y: -1 },
});

/**
 * Convenience function to close the context menu.
 */
const onClose = () => {
  $videoContextMenuState.setKey('isOpen', false);
};

/**
 * Map of elements to image DTOs. This is used to determine which image DTO to show the context menu for, depending on
 * the target of the context menu or long press event.
 */
const elToVideoMap = new Map<HTMLElement, VideoDTO>();

/**
 * Given a target node, find the first registered parent element that contains the target node and return the imageDTO
 * associated with it.
 */
const getVideoDTOFromMap = (target: Node): VideoDTO | undefined => {
  const entry = Array.from(elToVideoMap.entries()).find((entry) => entry[0].contains(target));
  return entry?.[1];
};

/**
 * Register a context menu for an image DTO on a target element.
 * @param imageDTO The image DTO to register the context menu for.
 * @param targetRef The ref of the target element that should trigger the context menu.
 */
export const useVideoContextMenu = (videoDTO: VideoDTO, ref: RefObject<HTMLElement> | (HTMLElement | null)) => {
  useEffect(() => {
    if (ref === null) {
      return;
    }
    const el = ref instanceof HTMLElement ? ref : ref.current;
    if (!el) {
      return;
    }
    elToVideoMap.set(el, videoDTO);
    return () => {
      elToVideoMap.delete(el);
    };
  }, [videoDTO, ref]);
};

/**
 * Singleton component that renders the context menu for images.
 */
export const VideoContextMenu = memo(() => {
  useAssertSingleton('VideoContextMenu');
  const state = useStore($videoContextMenuState);
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
      <VideoContextMenuEventLogical />
    </Portal>
  );
});

VideoContextMenu.displayName = 'VideoContextMenu';

const _hover: ChakraProps['_hover'] = { bg: 'transparent' };

/**
 * A logical component that listens for context menu events and opens the context menu. It's separate from
 * ImageContextMenu component to avoid re-rendering the whole context menu on every context menu event.
 */
const VideoContextMenuEventLogical = memo(() => {
  const lastPositionRef = useRef<{ x: number; y: number }>({ x: -1, y: -1 });
  const longPressTimeoutRef = useRef(0);
  const animationTimeoutRef = useRef(0);

  const onContextMenu = useCallback((e: MouseEvent | PointerEvent) => {
    if (e.shiftKey) {
      // This is a shift + right click event, which should open the native context menu
      onClose();
      return;
    }

    const videoDTO = getVideoDTOFromMap(e.target as Node);

    if (!videoDTO) {
      // Can't find the image DTO, close the context menu
      onClose();
      return;
    }

    // clear pending delayed open
    window.clearTimeout(animationTimeoutRef.current);
    e.preventDefault();

    if (lastPositionRef.current.x !== e.pageX || lastPositionRef.current.y !== e.pageY) {
      // if the mouse moved, we need to close, wait for animation and reopen the menu at the new position
      if ($videoContextMenuState.get().isOpen) {
        onClose();
      }
      animationTimeoutRef.current = window.setTimeout(() => {
        // Open the menu after the animation with the new state
        $videoContextMenuState.set({
          isOpen: true,
          position: { x: e.pageX, y: e.pageY },
          videoDTO,
        });
      }, 100);
    } else {
      // else we can just open the menu at the current position w/ new state
      $videoContextMenuState.set({
        isOpen: true,
        position: { x: e.pageX, y: e.pageY },
        videoDTO,
      });
    }

    // Always sync the last position
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

    // If the pointer has moved more than the threshold, cancel the long press
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
      // Clean up any timeouts when we unmount
      window.clearTimeout(animationTimeoutRef.current);
      window.clearTimeout(longPressTimeoutRef.current);
    },
    []
  );

  return null;
});

VideoContextMenuEventLogical.displayName = 'VideoContextMenuEventLogical';

// The content of the context menu, which changes based on the selection count. Split out and memoized to avoid
// re-rendering the whole context menu too often.
const MenuContent = memo(() => {
  const selectionCount = useAppSelector(selectSelectionCount);
  const state = useStore($videoContextMenuState);

  if (!state.videoDTO) {
    return null;
  }

  if (selectionCount > 1) {
    return (
      <MenuList visibility="visible">
        <MultipleSelectionVideoMenuItems />
      </MenuList>
    );
  }

  return (
    <MenuList visibility="visible">
      <SingleSelectionVideoMenuItems videoDTO={state.videoDTO} />
    </MenuList>
  );
});

MenuContent.displayName = 'MenuContent';

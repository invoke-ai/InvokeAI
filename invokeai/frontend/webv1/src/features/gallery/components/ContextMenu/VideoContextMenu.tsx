import type { ChakraProps } from '@invoke-ai/ui-library';
import { Menu, MenuButton, MenuDivider, MenuList, Portal, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { ContextMenuItemChangeBoardVideo } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemChangeBoardVideo';
import { ContextMenuItemDeleteVideo } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDeleteVideo';
import { ContextMenuItemDownloadVideo } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDownloadVideo';
import { ContextMenuItemOpenInNewTabVideo } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInNewTabVideo';
import MultipleSelectionMenuItemsVideos from 'features/gallery/components/ContextMenu/MultipleSelectionMenuItemsVideos';
import { VideoDTOContextProvider } from 'features/gallery/contexts/VideoDTOContext';
import { selectSelection } from 'features/gallery/store/gallerySelectors';
import { map } from 'nanostores';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef } from 'react';
import type { VideoDTO } from 'services/api/types';

// Mirror of ImageContextMenu, but pared down to the three actions the video item supports today:
// delete, change-board, download. Long-press on touch devices opens the menu the same way.

const LONGPRESS_DELAY_MS = 500;
const LONGPRESS_MOVE_THRESHOLD_PX = 10;

const $videoContextMenuState = map<{
  isOpen: boolean;
  videoDTO: VideoDTO | null;
  position: { x: number; y: number };
}>({
  isOpen: false,
  videoDTO: null,
  position: { x: -1, y: -1 },
});

const onClose = () => {
  $videoContextMenuState.setKey('isOpen', false);
};

const elToVideoMap = new Map<HTMLElement, VideoDTO>();

const getVideoDTOFromMap = (target: Node): VideoDTO | undefined => {
  const entry = Array.from(elToVideoMap.entries()).find((entry) => entry[0].contains(target));
  return entry?.[1];
};

/**
 * Register a context menu for a video DTO on a target element. Mirrors useImageContextMenu.
 */
export const useVideoContextMenu = (videoDTO: VideoDTO, ref: RefObject<HTMLElement | null> | (HTMLElement | null)) => {
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

const _hover: ChakraProps['_hover'] = { bg: 'transparent' };

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

const MenuContent = memo(() => {
  const state = useStore($videoContextMenuState);
  const selection = useAppSelector(selectSelection);
  if (!state.videoDTO) {
    return null;
  }
  // Only show the multi-selection menu when the clicked video is part of the selection —
  // right-clicking an item outside the selection acts on that item alone. Without this,
  // right-clicking a video while 2+ images are selected showed a menu whose every action
  // was disabled (the multi menu filters the selection down to videos and finds none).
  if (selection.length > 1 && selection.includes(state.videoDTO.video_name)) {
    return (
      <MenuList visibility="visible">
        <MultipleSelectionMenuItemsVideos />
      </MenuList>
    );
  }
  return (
    <MenuList visibility="visible">
      <VideoDTOContextProvider value={state.videoDTO}>
        <ContextMenuItemOpenInNewTabVideo />
        <ContextMenuItemDownloadVideo />
        <ContextMenuItemChangeBoardVideo />
        <MenuDivider />
        <ContextMenuItemDeleteVideo />
      </VideoDTOContextProvider>
    </MenuList>
  );
});

MenuContent.displayName = 'VideoContextMenuContent';

/**
 * Logical component that listens for context-menu events and dispatches to the singleton's state.
 * Split out from the visible menu to keep re-renders cheap.
 */
const VideoContextMenuEventLogical = memo(() => {
  const lastPositionRef = useRef<{ x: number; y: number }>({ x: -1, y: -1 });
  const longPressTimeoutRef = useRef(0);
  const animationTimeoutRef = useRef(0);

  const onContextMenu = useCallback((e: MouseEvent | PointerEvent) => {
    if (e.shiftKey) {
      // shift+right-click opens the native context menu
      onClose();
      return;
    }

    const videoDTO = getVideoDTOFromMap(e.target as Node);
    if (!videoDTO) {
      // Not over a registered video item — let ImageContextMenu handle it (or close).
      onClose();
      return;
    }

    window.clearTimeout(animationTimeoutRef.current);
    e.preventDefault();

    if (lastPositionRef.current.x !== e.pageX || lastPositionRef.current.y !== e.pageY) {
      if ($videoContextMenuState.get().isOpen) {
        onClose();
      }
      animationTimeoutRef.current = window.setTimeout(() => {
        $videoContextMenuState.set({
          isOpen: true,
          position: { x: e.pageX, y: e.pageY },
          videoDTO,
        });
      }, 100);
    } else {
      $videoContextMenuState.set({
        isOpen: true,
        position: { x: e.pageX, y: e.pageY },
        videoDTO,
      });
    }

    lastPositionRef.current = { x: e.pageX, y: e.pageY };
  }, []);

  const onPointerDown = useCallback(
    (e: PointerEvent) => {
      if (e.pointerType === 'mouse') {
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
      return;
    }
    if (longPressTimeoutRef.current === null) {
      return;
    }
    const distance = Math.hypot(e.pageX - lastPositionRef.current.x, e.pageY - lastPositionRef.current.y);
    if (distance > LONGPRESS_MOVE_THRESHOLD_PX) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  const onPointerUp = useCallback((e: PointerEvent) => {
    if (e.pointerType === 'mouse') {
      return;
    }
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  const onPointerCancel = useCallback((e: PointerEvent) => {
    if (e.pointerType === 'mouse') {
      return;
    }
    if (longPressTimeoutRef.current) {
      clearTimeout(longPressTimeoutRef.current);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    window.addEventListener('contextmenu', onContextMenu, { signal: controller.signal });
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

VideoContextMenuEventLogical.displayName = 'VideoContextMenuEventLogical';

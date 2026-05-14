import type { ChakraProps } from '@invoke-ai/ui-library';
import { Menu, MenuButton, MenuList, Portal, useGlobalMenuClose } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { ContextMenuItemDeleteCanvasProject } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDeleteCanvasProject';
import { ContextMenuItemDownloadCanvasProject } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDownloadCanvasProject';
import { ContextMenuItemLoadCanvasProject } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemLoadCanvasProject';
import { CanvasProjectDTOContextProvider } from 'features/gallery/contexts/CanvasProjectDTOContext';
import { map } from 'nanostores';
import type { RefObject } from 'react';
import { memo, useCallback, useEffect, useRef } from 'react';
import type { CanvasProjectDTO } from 'services/api/types';

// Mirror of VideoContextMenu, with the two canvas-project actions: delete, download. A
// "Change Board" entry follows once ChangeBoardModal learns about canvas projects.

const LONGPRESS_DELAY_MS = 500;
const LONGPRESS_MOVE_THRESHOLD_PX = 10;

const $canvasProjectContextMenuState = map<{
  isOpen: boolean;
  projectDTO: CanvasProjectDTO | null;
  position: { x: number; y: number };
}>({
  isOpen: false,
  projectDTO: null,
  position: { x: -1, y: -1 },
});

const onClose = () => {
  $canvasProjectContextMenuState.setKey('isOpen', false);
};

const elToProjectMap = new Map<HTMLElement, CanvasProjectDTO>();

const getProjectDTOFromMap = (target: Node): CanvasProjectDTO | undefined => {
  const entry = Array.from(elToProjectMap.entries()).find((entry) => entry[0].contains(target));
  return entry?.[1];
};

/**
 * Register a context menu for a canvas project DTO on a target element. Mirrors useVideoContextMenu.
 */
export const useCanvasProjectContextMenu = (
  projectDTO: CanvasProjectDTO,
  ref: RefObject<HTMLElement> | (HTMLElement | null)
) => {
  useEffect(() => {
    if (ref === null) {
      return;
    }
    const el = ref instanceof HTMLElement ? ref : ref.current;
    if (!el) {
      return;
    }
    elToProjectMap.set(el, projectDTO);
    return () => {
      elToProjectMap.delete(el);
    };
  }, [projectDTO, ref]);
};

const _hover: ChakraProps['_hover'] = { bg: 'transparent' };

export const CanvasProjectContextMenu = memo(() => {
  useAssertSingleton('CanvasProjectContextMenu');
  const state = useStore($canvasProjectContextMenuState);
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
      <CanvasProjectContextMenuEventLogical />
    </Portal>
  );
});

CanvasProjectContextMenu.displayName = 'CanvasProjectContextMenu';

const MenuContent = memo(() => {
  const state = useStore($canvasProjectContextMenuState);
  if (!state.projectDTO) {
    return null;
  }
  return (
    <MenuList visibility="visible">
      <CanvasProjectDTOContextProvider value={state.projectDTO}>
        <IconMenuItemGroup>
          <ContextMenuItemDownloadCanvasProject />
          <ContextMenuItemDeleteCanvasProject />
        </IconMenuItemGroup>
        <ContextMenuItemLoadCanvasProject />
      </CanvasProjectDTOContextProvider>
    </MenuList>
  );
});

MenuContent.displayName = 'CanvasProjectContextMenuContent';

/**
 * Listens for context-menu events and dispatches to the singleton's state. Split out from the
 * visible menu so re-renders on `state.isOpen` toggling stay cheap.
 */
const CanvasProjectContextMenuEventLogical = memo(() => {
  const lastPositionRef = useRef<{ x: number; y: number }>({ x: -1, y: -1 });
  const longPressTimeoutRef = useRef(0);
  const animationTimeoutRef = useRef(0);

  const onContextMenu = useCallback((e: MouseEvent | PointerEvent) => {
    if (e.shiftKey) {
      onClose();
      return;
    }

    const projectDTO = getProjectDTOFromMap(e.target as Node);
    if (!projectDTO) {
      onClose();
      return;
    }

    window.clearTimeout(animationTimeoutRef.current);
    e.preventDefault();

    if (lastPositionRef.current.x !== e.pageX || lastPositionRef.current.y !== e.pageY) {
      if ($canvasProjectContextMenuState.get().isOpen) {
        onClose();
      }
      animationTimeoutRef.current = window.setTimeout(() => {
        $canvasProjectContextMenuState.set({
          isOpen: true,
          position: { x: e.pageX, y: e.pageY },
          projectDTO,
        });
      }, 100);
    } else {
      $canvasProjectContextMenuState.set({
        isOpen: true,
        position: { x: e.pageX, y: e.pageY },
        projectDTO,
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

CanvasProjectContextMenuEventLogical.displayName = 'CanvasProjectContextMenuEventLogical';

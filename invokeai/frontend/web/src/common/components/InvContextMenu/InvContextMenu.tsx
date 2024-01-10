/**
 * Adapted from https://github.com/lukasbach/chakra-ui-contextmenu
 */
import type {
  ChakraProps,
  MenuButtonProps,
  MenuProps,
  PortalProps,
} from '@chakra-ui/react';
import { Portal, useDisclosure, useEventListener } from '@chakra-ui/react';
import { InvMenu, InvMenuButton } from 'common/components/InvMenu/wrapper';
import { useGlobalMenuClose } from 'common/hooks/useGlobalMenuClose';
import { typedMemo } from 'common/util/typedMemo';
import { useCallback, useEffect, useRef, useState } from 'react';

export interface InvContextMenuProps<T extends HTMLElement = HTMLDivElement> {
  renderMenu: () => JSX.Element | null;
  children: (ref: React.MutableRefObject<T | null>) => JSX.Element | null;
  menuProps?: Omit<MenuProps, 'children'> & { children?: React.ReactNode };
  portalProps?: Omit<PortalProps, 'children'> & { children?: React.ReactNode };
  menuButtonProps?: MenuButtonProps;
}

export const InvContextMenu = typedMemo(
  <T extends HTMLElement = HTMLElement>(props: InvContextMenuProps<T>) => {
    const { isOpen, onOpen, onClose } = useDisclosure();
    const [position, setPosition] = useState([-1, -1]);
    const targetRef = useRef<T>(null);
    const lastPositionRef = useRef([-1, -1]);
    const timeoutRef = useRef(0);

    useGlobalMenuClose(onClose);

    const onContextMenu = useCallback(
      (e: MouseEvent) => {
        if (e.shiftKey) {
          onClose();
          return;
        }
        if (
          targetRef.current?.contains(e.target as HTMLElement) ||
          e.target === targetRef.current
        ) {
          // clear pending delayed open
          window.clearTimeout(timeoutRef.current);
          e.preventDefault();
          if (
            lastPositionRef.current[0] !== e.pageX ||
            lastPositionRef.current[1] !== e.pageY
          ) {
            // if the mouse moved, we need to close, wait for animation and reopen the menu at the new position
            onClose();
            timeoutRef.current = window.setTimeout(() => {
              onOpen();
              setPosition([e.pageX, e.pageY]);
            }, 100);
          } else {
            // else we can just open the menu at the current position
            onOpen();
            setPosition([e.pageX, e.pageY]);
          }
        }
        lastPositionRef.current = [e.pageX, e.pageY];
      },
      [onClose, onOpen]
    );

    useEffect(
      () => () => {
        window.clearTimeout(timeoutRef.current);
      },
      []
    );

    useEventListener('contextmenu', onContextMenu);

    return (
      <>
        {props.children(targetRef)}
        <Portal {...props.portalProps}>
          <InvMenu
            isLazy
            isOpen={isOpen}
            gutter={0}
            placement="auto-end"
            onClose={onClose}
            {...props.menuProps}
          >
            <InvMenuButton
              aria-hidden={true}
              w={1}
              h={1}
              position="absolute"
              left={position[0]}
              top={position[1]}
              cursor="default"
              bg="transparent"
              size="sm"
              _hover={_hover}
              pointerEvents="none"
              {...props.menuButtonProps}
            />
            {props.renderMenu()}
          </InvMenu>
        </Portal>
      </>
    );
  }
);

const _hover: ChakraProps['_hover'] = { bg: 'transparent' };

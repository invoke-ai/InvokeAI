/**
 * This is a copy-paste of https://github.com/lukasbach/chakra-ui-contextmenu with a small change.
 *
 * The reactflow background element somehow prevents the chakra `useOutsideClick()` hook from working.
 * With a menu open, clicking on the reactflow background element doesn't close the menu.
 *
 * Reactflow does provide an `onPaneClick` to handle clicks on the background element, but it is not
 * straightforward to programatically close the menu.
 *
 * As a (hopefully temporary) workaround, we will use a dirty hack:
 * - create `globalContextMenuCloseTrigger: number` in `ui` slice
 * - increment it in `onPaneClick` (and wherever else we want to close the menu)
 * - `useEffect()` to close the menu when `globalContextMenuCloseTrigger` changes
 */
import type { MenuButtonProps, MenuProps, PortalProps } from '@chakra-ui/react';
import { Portal, useEventListener } from '@chakra-ui/react';
import { InvMenu, InvMenuButton } from 'common/components/InvMenu/wrapper';
import { useGlobalMenuCloseTrigger } from 'common/hooks/useGlobalMenuCloseTrigger';
import { memo, useCallback, useEffect, useRef, useState } from 'react';

export interface InvContextMenuProps<T extends HTMLElement = HTMLDivElement> {
  renderMenu: () => JSX.Element | null;
  children: (ref: React.MutableRefObject<T | null>) => JSX.Element | null;
  menuProps?: Omit<MenuProps, 'children'> & { children?: React.ReactNode };
  portalProps?: Omit<PortalProps, 'children'> & { children?: React.ReactNode };
  menuButtonProps?: MenuButtonProps;
}

export const InvContextMenu = memo(
  <T extends HTMLElement = HTMLElement>(props: InvContextMenuProps<T>) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isRendered, setIsRendered] = useState(false);
    const [isDeferredOpen, setIsDeferredOpen] = useState(false);
    const [position, setPosition] = useState<[number, number]>([0, 0]);
    const targetRef = useRef<T>(null);

    useEffect(() => {
      if (isOpen) {
        setTimeout(() => {
          setIsRendered(true);
          setTimeout(() => {
            setIsDeferredOpen(true);
          });
        });
      } else {
        setIsDeferredOpen(false);
        const timeout = setTimeout(() => {
          setIsRendered(isOpen);
        }, 1000);
        return () => clearTimeout(timeout);
      }
    }, [isOpen]);

    const onClose = useCallback(() => {
      setIsOpen(false);
      setIsDeferredOpen(false);
      setIsRendered(false);
    }, []);

    // This is the change from the original chakra-ui-contextmenu
    // Close all menus when the globalContextMenuCloseTrigger changes
    useGlobalMenuCloseTrigger(onClose);

    useEventListener('contextmenu', (e) => {
      if (
        targetRef.current?.contains(e.target as HTMLElement) ||
        e.target === targetRef.current
      ) {
        e.preventDefault();
        setIsOpen(true);
        setPosition([e.pageX, e.pageY]);
      } else {
        setIsOpen(false);
      }
    });

    const onCloseHandler = useCallback(() => {
      props.menuProps?.onClose?.();
      setIsOpen(false);
    }, [props.menuProps]);

    return (
      <>
        {props.children(targetRef)}
        {isRendered && (
          <Portal {...props.portalProps}>
            <InvMenu
              isLazy
              isOpen={isDeferredOpen}
              gutter={0}
              onClose={onCloseHandler}
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
                _hover={{ bg: 'transparent' }}
                {...props.menuButtonProps}
              />
              {props.renderMenu()}
            </InvMenu>
          </Portal>
        )}
      </>
    );
  }
);

InvContextMenu.displayName = 'InvContextMenu';

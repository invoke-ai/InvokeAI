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
 * - increment it in `onPaneClick`
 * - `useEffect()` to close the menu when `globalContextMenuCloseTrigger` changes
 */

import {
  Menu,
  MenuButton,
  MenuButtonProps,
  MenuProps,
  Portal,
  PortalProps,
  useEventListener,
} from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import * as React from 'react';
import {
  MutableRefObject,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';

export interface IAIContextMenuProps<T extends HTMLElement> {
  renderMenu: () => JSX.Element | null;
  children: (ref: MutableRefObject<T | null>) => JSX.Element | null;
  menuProps?: Omit<MenuProps, 'children'> & { children?: React.ReactNode };
  portalProps?: Omit<PortalProps, 'children'> & { children?: React.ReactNode };
  menuButtonProps?: MenuButtonProps;
}

export function IAIContextMenu<T extends HTMLElement = HTMLElement>(
  props: IAIContextMenuProps<T>
) {
  const [isOpen, setIsOpen] = useState(false);
  const [isRendered, setIsRendered] = useState(false);
  const [isDeferredOpen, setIsDeferredOpen] = useState(false);
  const [position, setPosition] = useState<[number, number]>([0, 0]);
  const targetRef = useRef<T>(null);

  const globalContextMenuCloseTrigger = useAppSelector(
    (state) => state.ui.globalContextMenuCloseTrigger
  );

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

  useEffect(() => {
    setIsOpen(false);
    setIsDeferredOpen(false);
    setIsRendered(false);
  }, [globalContextMenuCloseTrigger]);

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
          <Menu
            isOpen={isDeferredOpen}
            gutter={0}
            {...props.menuProps}
            onClose={onCloseHandler}
          >
            <MenuButton
              aria-hidden={true}
              w={1}
              h={1}
              style={{
                position: 'absolute',
                left: position[0],
                top: position[1],
                cursor: 'default',
              }}
              {...props.menuButtonProps}
            />
            {props.renderMenu()}
          </Menu>
        </Portal>
      )}
    </>
  );
}

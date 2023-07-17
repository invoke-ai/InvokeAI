import { MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { ImageDTO } from 'services/api/types';
import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';
import { MotionProps } from 'framer-motion';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const motionProps: MotionProps = {
  variants: {
    enter: {
      visibility: 'visible',
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.07,
        ease: [0.4, 0, 0.2, 1],
      },
    },
    exit: {
      transitionEnd: {
        visibility: 'hidden',
      },
      opacity: 0,
      scale: 0.8,
      transition: {
        duration: 0.07,
        easings: 'easeOut',
      },
    },
  },
};

const ImageContextMenu = ({ imageDTO, children }: Props) => {
  const selector = useMemo(
    () =>
      createSelector(
        [stateSelector],
        ({ gallery }) => {
          const selectionCount = gallery.selection.length;

          return { selectionCount };
        },
        defaultSelectorOptions
      ),
    []
  );

  const { selectionCount } = useAppSelector(selector);

  const handleContextMenu = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={() =>
        imageDTO ? (
          <MenuList
            sx={{ visibility: 'visible !important' }}
            motionProps={motionProps}
            onContextMenu={handleContextMenu}
          >
            {selectionCount === 1 ? (
              <SingleSelectionMenuItems imageDTO={imageDTO} />
            ) : (
              <MultipleSelectionMenuItems />
            )}
          </MenuList>
        ) : null
      }
    >
      {children}
    </ContextMenu>
  );
};

export default memo(ImageContextMenu);

import { MenuList } from '@chakra-ui/react';
import {
  IAIContextMenu,
  IAIContextMenuProps,
} from 'common/components/IAIContextMenu';
import { MouseEvent, memo, useCallback } from 'react';
import { ImageDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { useAppSelector } from 'app/store/storeHooks';
import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: IAIContextMenuProps<HTMLDivElement>['children'];
};

const selector = createSelector(
  [stateSelector],
  ({ gallery }) => {
    const selectionCount = gallery.selection.length;

    return { selectionCount };
  },
  defaultSelectorOptions
);

const ImageContextMenu = ({ imageDTO, children }: Props) => {
  const { selectionCount } = useAppSelector(selector);

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  return (
    <IAIContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={() => {
        if (!imageDTO) {
          return null;
        }

        if (selectionCount > 1) {
          return (
            <MenuList
              sx={{ visibility: 'visible !important' }}
              motionProps={menuListMotionProps}
              onContextMenu={skipEvent}
            >
              <MultipleSelectionMenuItems />
            </MenuList>
          );
        }

        return (
          <MenuList
            sx={{ visibility: 'visible !important' }}
            motionProps={menuListMotionProps}
            onContextMenu={skipEvent}
          >
            <SingleSelectionMenuItems imageDTO={imageDTO} />
          </MenuList>
        );
      }}
    >
      {children}
    </IAIContextMenu>
  );
};

export default memo(ImageContextMenu);

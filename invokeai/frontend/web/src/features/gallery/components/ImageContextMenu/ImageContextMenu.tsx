import { MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { MouseEvent, memo, useCallback, useMemo } from 'react';
import { ImageDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: ContextMenuProps<HTMLDivElement>['children'];
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
            motionProps={menuListMotionProps}
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

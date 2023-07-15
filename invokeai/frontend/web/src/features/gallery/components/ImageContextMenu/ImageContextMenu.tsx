import { MenuList } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { memo, useMemo } from 'react';
import { ImageDTO } from 'services/api/types';
import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO;
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

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      renderMenu={() => (
        <MenuList sx={{ visibility: 'visible !important' }}>
          {selectionCount === 1 ? (
            <SingleSelectionMenuItems imageDTO={imageDTO} />
          ) : (
            <MultipleSelectionMenuItems />
          )}
        </MenuList>
      )}
    >
      {children}
    </ContextMenu>
  );
};

export default memo(ImageContextMenu);

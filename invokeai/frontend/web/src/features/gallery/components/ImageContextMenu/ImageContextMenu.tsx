import { MenuList } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import {
  IAIContextMenu,
  IAIContextMenuProps,
} from 'common/components/IAIContextMenu';
import { MouseEvent, memo, useCallback } from 'react';
import { ImageDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: IAIContextMenuProps<HTMLDivElement>['children'];
};

const selector = createMemoizedSelector([stateSelector], ({ gallery }) => {
  const selectionCount = gallery.selection.length;

  return { selectionCount };
});

const ImageContextMenu = ({ imageDTO, children }: Props) => {
  const { selectionCount } = useAppSelector(selector);

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const renderMenuFunc = useCallback(() => {
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
  }, [imageDTO, selectionCount, skipEvent]);

  return (
    <IAIContextMenu<HTMLDivElement>
      menuProps={{ size: 'sm', isLazy: true }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={renderMenuFunc}
    >
      {children}
    </IAIContextMenu>
  );
};

export default memo(ImageContextMenu);

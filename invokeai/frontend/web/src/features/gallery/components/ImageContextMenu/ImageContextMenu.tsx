import { MenuList } from '@chakra-ui/react';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import { MouseEvent, memo, useCallback } from 'react';
import { ImageDTO } from 'services/api/types';
import { menuListMotionProps } from 'theme/components/menu';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const ImageContextMenu = ({ imageDTO, children }: Props) => {
  // const selector = useMemo(
  //   () =>
  //     createSelector(
  //       [stateSelector],
  //       ({ gallery }) => {
  //         const selectionCount = gallery.selection.length;

  //         return { selectionCount };
  //       },
  //       defaultSelectorOptions
  //     ),
  //   []
  // );

  // const { selectionCount } = useAppSelector(selector);

  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
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
            onContextMenu={skipEvent}
          >
            <SingleSelectionMenuItems imageDTO={imageDTO} />
          </MenuList>
        ) : null
      }
    >
      {children}
    </ContextMenu>
  );
};

export default memo(ImageContextMenu);

import { MenuItem, MenuList } from '@chakra-ui/react';
import { ContextMenu, ContextMenuProps } from 'chakra-ui-contextmenu';
import {
  InputFieldTemplate,
  InputFieldValue,
} from 'features/nodes/types/types';
import { MouseEvent, useCallback } from 'react';
import { menuListMotionProps } from 'theme/components/menu';

type Props = {
  nodeId: string;
  field: InputFieldValue;
  fieldTemplate: InputFieldTemplate;
  children: ContextMenuProps<HTMLDivElement>['children'];
};

const FieldContextMenu = (props: Props) => {
  const skipEvent = useCallback((e: MouseEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  return (
    <ContextMenu<HTMLDivElement>
      menuProps={{
        size: 'sm',
        isLazy: true,
      }}
      menuButtonProps={{
        bg: 'transparent',
        _hover: { bg: 'transparent' },
      }}
      renderMenu={() => (
        <MenuList
          sx={{ visibility: 'visible !important' }}
          motionProps={menuListMotionProps}
          onContextMenu={skipEvent}
        >
          <MenuItem>Test</MenuItem>
        </MenuList>
      )}
    >
      {props.children}
    </ContextMenu>
  );
};

export default FieldContextMenu;

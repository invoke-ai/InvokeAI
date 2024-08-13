import type { FlexProps } from '@invoke-ai/ui-library';
import { ContextMenu, Flex, MenuList } from '@invoke-ai/ui-library';
import { CanvasEntityActionMenuItems } from 'features/controlLayers/components/common/CanvasEntityActionMenuItems';
import { memo, useCallback } from 'react';

export const CanvasEntityHeader = memo(({ children, ...rest }: FlexProps) => {
  const renderMenu = useCallback(() => {
    return (
      <MenuList>
        <CanvasEntityActionMenuItems />
      </MenuList>
    );
  }, []);

  return (
    <ContextMenu renderMenu={renderMenu}>
      {(ref) => (
        <Flex ref={ref} gap={2} alignItems="center" p={2} role="button" {...rest}>
          {children}
        </Flex>
      )}
    </ContextMenu>
  );
});

CanvasEntityHeader.displayName = 'CanvasEntityHeader';

import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useEntityIsSelected } from 'features/controlLayers/hooks/useEntityIsSelected';
import { useEntitySelectionColor } from 'features/controlLayers/hooks/useEntitySelectionColor';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import type { PropsWithChildren } from 'react';
import { memo, useCallback } from 'react';

export const CanvasEntityContainer = memo((props: PropsWithChildren) => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isSelected = useEntityIsSelected(entityIdentifier);
  const selectionColor = useEntitySelectionColor(entityIdentifier);
  const onClick = useCallback(() => {
    if (isSelected) {
      return;
    }
    dispatch(entitySelected({ entityIdentifier }));
  }, [dispatch, entityIdentifier, isSelected]);

  return (
    <Flex
      position="relative"
      flexDir="column"
      w="full"
      bg={isSelected ? 'base.800' : 'base.850'}
      onClick={onClick}
      borderInlineStartWidth={5}
      borderColor={isSelected ? selectionColor : 'base.800'}
      borderRadius="base"
    >
      {props.children}
    </Flex>
  );
});

CanvasEntityContainer.displayName = 'CanvasEntityContainer';

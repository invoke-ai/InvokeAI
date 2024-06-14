import { Flex, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CAHeaderItems } from 'features/controlLayers/components/ControlAdapter/CAHeaderItems';
import { CASettings } from 'features/controlLayers/components/ControlAdapter/CASettings';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { entitySelected } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const CAEntity = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onClick = useCallback(() => {
    dispatch(entitySelected({ id, type: 'control_adapter' }));
  }, [dispatch, id]);

  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <CAHeaderItems id={id} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          <CASettings id={id} />
        </Flex>
      )}
    </LayerWrapper>
  );
});

CAEntity.displayName = 'CAEntity';

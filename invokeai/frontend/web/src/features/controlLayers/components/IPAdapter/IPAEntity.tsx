import { Flex, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IPAHeaderItems } from 'features/controlLayers/components/IPAdapter/IPAHeaderItems';
import { IPASettings } from 'features/controlLayers/components/IPAdapter/IPASettings';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { entitySelected } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';

type Props = {
  id: string;
};

export const IPAEntity = memo(({ id }: Props) => {
  const dispatch = useAppDispatch();
  const isSelected = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.id === id);
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  const onClick = useCallback(() => {
    dispatch(entitySelected({ id, type: 'ip_adapter' }));
  }, [dispatch, id]);

  return (
    <LayerWrapper onClick={onClick} borderColor={isSelected ? 'base.400' : 'base.800'}>
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <IPAHeaderItems id={id} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          <IPASettings id={id} />
        </Flex>
      )}
    </LayerWrapper>
  );
});

IPAEntity.displayName = 'IPAEntity';

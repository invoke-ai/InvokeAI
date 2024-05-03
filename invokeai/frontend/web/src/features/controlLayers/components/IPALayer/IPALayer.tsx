import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { IPALayerIPAdapterWrapper } from 'features/controlLayers/components/IPALayer/IPALayerIPAdapterWrapper';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { LayerWrapper } from 'features/controlLayers/components/LayerCommon/LayerWrapper';
import { memo } from 'react';

type Props = {
  layerId: string;
};

export const IPALayer = memo(({ layerId }: Props) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  return (
    <LayerWrapper borderColor="base.800">
      <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
        <LayerVisibilityToggle layerId={layerId} />
        <LayerTitle type="ip_adapter_layer" />
        <Spacer />
        <LayerDeleteButton layerId={layerId} />
      </Flex>
      {isOpen && (
        <Flex flexDir="column" gap={3} px={3} pb={3}>
          <IPALayerIPAdapterWrapper layerId={layerId} />
        </Flex>
      )}
    </LayerWrapper>
  );
});

IPALayer.displayName = 'IPALayer';

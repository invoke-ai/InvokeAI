import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { IPALayerConfig } from 'features/controlLayers/components/IPALayer/IPALayerConfig';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerCommon/LayerDeleteButton';
import { LayerTitle } from 'features/controlLayers/components/LayerCommon/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerCommon/LayerVisibilityToggle';
import { memo } from 'react';

type Props = {
  layerId: string;
};

export const IPALayer = memo(({ layerId }: Props) => {
  const { isOpen, onToggle } = useDisclosure({ defaultIsOpen: true });
  return (
    <Flex gap={2} bg="base.800" borderRadius="base" p="1px" px={2}>
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
          <LayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="ip_adapter_layer" />
          <Spacer />
          <LayerDeleteButton layerId={layerId} />
        </Flex>
        {isOpen && (
          <Flex flexDir="column" gap={3} px={3} pb={3}>
            <IPALayerConfig layerId={layerId} />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

IPALayer.displayName = 'IPALayer';

import { Box, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $stageAttrs } from 'features/controlLayers/store/controlLayersSlice';
import { round } from 'lodash-es';
import { memo } from 'react';

export const HeadsUpDisplay = memo(() => {
  const stageAttrs = useStore($stageAttrs);
  const layerCount = useAppSelector((s) => s.controlLayers.present.layers.length);
  const bbox = useAppSelector((s) => s.controlLayers.present.bbox);

  return (
    <Flex flexDir="column" bg="blackAlpha.400" borderBottomEndRadius="base" p={2} minW={64} gap={2}>
      <HUDItem label="Scale" value={round(stageAttrs.scale, 3)} />
      <HUDItem label="Stage Pos" value={`${round(stageAttrs.x, 3)}, ${round(stageAttrs.y, 3)}`} />
      <HUDItem label="Stage Size" value={`${round(stageAttrs.width, 3)}, ${round(stageAttrs.height, 3)}`} />
      <HUDItem label="Layer Count" value={layerCount} />
      <HUDItem label="BBox Size" value={`${bbox.width}×${bbox.height}`} />
      <HUDItem label="BBox Position" value={`${bbox.x}, ${bbox.y}`} />
      <HUDItem label="BBox Width % 8" value={round(bbox.width % 8, 3)} />
      <HUDItem label="BBox Height % 8" value={round(bbox.height % 8, 3)} />
      <HUDItem label="BBox X % 8" value={round(bbox.x % 8, 3)} />
      <HUDItem label="BBox Y % 8" value={round(bbox.y % 8, 3)} />
    </Flex>
  );
});

HeadsUpDisplay.displayName = 'HeadsUpDisplay';

const HUDItem = memo(({ label, value }: { label: string; value: string | number }) => {
  return (
    <Box display="inline-block" lineHeight={1}>
      <Text as="span">{label}: </Text>
      <Text as="span" fontWeight="semibold">
        {value}
      </Text>
    </Box>
  );
});

HUDItem.displayName = 'HUDItem';

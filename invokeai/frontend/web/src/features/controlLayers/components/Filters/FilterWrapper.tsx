import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useFilter } from 'features/controlLayers/components/Filters/Filter';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

const FilterWrapper = (props: PropsWithChildren) => {
  const isPreviewDisabled = useAppSelector((s) => s.canvasV2.selectedEntityIdentifier?.type !== 'layer');
  const filter = useFilter();
  return (
    <Flex flexDir="column" gap={3} w="full" h="full">
      {props.children}
    </Flex>
  );
};

export default memo(FilterWrapper);

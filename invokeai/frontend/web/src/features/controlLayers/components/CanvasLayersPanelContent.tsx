import { Divider, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { FocusRegionWrapper } from 'common/components/FocusRegionWrapper';
import { CanvasAddEntityButtons } from 'features/controlLayers/components/CanvasAddEntityButtons';
import { CanvasEntityList } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityList';
import { EntityListSelectedEntityActionBar } from 'features/controlLayers/components/CanvasEntityList/EntityListSelectedEntityActionBar';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectHasEntities } from 'features/controlLayers/store/selectors';
import { memo } from 'react';

import { ParamDenoisingStrength } from './ParamDenoisingStrength';

export const CanvasLayersPanel = memo(() => {
  const hasEntities = useAppSelector(selectHasEntities);

  return (
    <CanvasManagerProviderGate>
      <FocusRegionWrapper region="layers" as={Flex} flexDir="column" gap={2} w="full" h="full" p={2}>
        <EntityListSelectedEntityActionBar />
        <Divider py={0} />
        <ParamDenoisingStrength />
        <Divider py={0} />
        {!hasEntities && <CanvasAddEntityButtons />}
        {hasEntities && <CanvasEntityList />}
      </FocusRegionWrapper>
    </CanvasManagerProviderGate>
  );
});

CanvasLayersPanel.displayName = 'CanvasLayersPanel';

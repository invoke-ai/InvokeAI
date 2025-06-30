import { ButtonGroup } from '@invoke-ai/ui-library';
import { SimpleStagingAreaToolbarMenu } from 'features/controlLayers/components/StagingArea/SimpleStagingAreaToolbarMenu';
import { StagingAreaToolbarDiscardAllButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardAllButton';
import { StagingAreaToolbarDiscardSelectedButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardSelectedButton';
import { StagingAreaToolbarImageCountButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarImageCountButton';
import { StagingAreaToolbarInfoButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarInfoButton';
import { StagingAreaToolbarNextButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarNextButton';
import { StagingAreaToolbarPrevButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarPrevButton';
import { memo } from 'react';

export const SimpleStagingAreaToolbar = memo(() => {
  return (
    <>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarPrevButton />
        <StagingAreaToolbarImageCountButton />
        <StagingAreaToolbarNextButton />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarInfoButton />
        <StagingAreaToolbarDiscardSelectedButton />
        <SimpleStagingAreaToolbarMenu />
        <StagingAreaToolbarDiscardAllButton />
      </ButtonGroup>
    </>
  );
});

SimpleStagingAreaToolbar.displayName = 'SimpleStagingAreaToolbar';

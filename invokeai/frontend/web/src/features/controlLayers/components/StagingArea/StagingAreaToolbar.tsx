import { ButtonGroup } from '@invoke-ai/ui-library';
import { StagingAreaToolbarAcceptButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarAcceptButton';
import { StagingAreaToolbarDiscardAllButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardAllButton';
import { StagingAreaToolbarDiscardSelectedButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardSelectedButton';
import { StagingAreaToolbarImageCountButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarImageCountButton';
import { StagingAreaToolbarNextButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarNextButton';
import { StagingAreaToolbarPrevButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarPrevButton';
import { StagingAreaToolbarSaveAsMenu } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveAsMenu';
import { StagingAreaToolbarSaveSelectedToGalleryButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveSelectedToGalleryButton';
import { StagingAreaToolbarToggleShowResultsButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarToggleShowResultsButton';
import { memo } from 'react';

export const StagingAreaToolbar = memo(() => {
  return (
    <>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarPrevButton />
        <StagingAreaToolbarImageCountButton />
        <StagingAreaToolbarNextButton />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarAcceptButton />
        <StagingAreaToolbarToggleShowResultsButton />
        <StagingAreaToolbarSaveSelectedToGalleryButton />
        <StagingAreaToolbarSaveAsMenu />
        <StagingAreaToolbarDiscardSelectedButton />
        <StagingAreaToolbarDiscardAllButton />
      </ButtonGroup>
    </>
  );
});

StagingAreaToolbar.displayName = 'StagingAreaToolbar';

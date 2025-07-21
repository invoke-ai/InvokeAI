import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { useStagingAreaContext } from 'features/controlLayers/components/StagingArea/context';
import { StagingAreaToolbarAcceptButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarAcceptButton';
import { StagingAreaToolbarDiscardAllButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardAllButton';
import { StagingAreaToolbarDiscardSelectedButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardSelectedButton';
import { StagingAreaToolbarImageCountButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarImageCountButton';
import { StagingAreaToolbarMenu } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenu';
import { StagingAreaToolbarNextButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarNextButton';
import { StagingAreaToolbarPrevButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarPrevButton';
import { StagingAreaToolbarSaveSelectedToGalleryButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveSelectedToGalleryButton';
import { StagingAreaToolbarToggleShowResultsButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarToggleShowResultsButton';
import { memo } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

import { StagingAreaAutoSwitchButtons } from './StagingAreaAutoSwitchButtons';

export const StagingAreaToolbar = memo(() => {
  const ctx = useStagingAreaContext();

  useHotkeys('meta+left', ctx.selectFirst, { preventDefault: true });
  useHotkeys('meta+right', ctx.selectLast, { preventDefault: true });

  return (
    <Flex gap={2}>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarPrevButton />
        <StagingAreaToolbarImageCountButton />
        <StagingAreaToolbarNextButton />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarAcceptButton />
        <StagingAreaToolbarToggleShowResultsButton />
        <StagingAreaToolbarSaveSelectedToGalleryButton />
        <StagingAreaToolbarMenu />
        <StagingAreaToolbarDiscardSelectedButton />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaAutoSwitchButtons />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarDiscardAllButton />
      </ButtonGroup>
    </Flex>
  );
});

StagingAreaToolbar.displayName = 'StagingAreaToolbar';

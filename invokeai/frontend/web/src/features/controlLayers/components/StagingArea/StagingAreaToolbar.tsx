import { ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { StagingAreaToolbarAcceptButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarAcceptButton';
import { StagingAreaToolbarDiscardAllButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardAllButton';
import { StagingAreaToolbarDiscardSelectedButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardSelectedButton';
import { StagingAreaToolbarImageCountButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarImageCountButton';
import { StagingAreaToolbarMenu } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarMenu';
import { StagingAreaToolbarNextButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarNextButton';
import { StagingAreaToolbarPrevButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarPrevButton';
import { StagingAreaToolbarSaveSelectedToGalleryButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveSelectedToGalleryButton';
import { StagingAreaToolbarToggleShowResultsButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarToggleShowResultsButton';
import { useCanvasManager } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useEffect } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

import { StagingAreaAutoSwitchButtons } from './StagingAreaAutoSwitchButtons';

export const StagingAreaToolbar = memo(() => {
  const canvasManager = useCanvasManager();
  const shouldShowStagedImage = useStore(canvasManager.stagingArea.$shouldShowStagedImage);

  const ctx = useCanvasSessionContext();

  useEffect(() => {
    return ctx.$selectedItemId.listen((id) => {
      if (id !== null) {
        document.getElementById(getQueueItemElementId(id))?.scrollIntoView();
      }
    });
  }, [ctx.$selectedItemId]);

  useHotkeys('meta+left', ctx.selectFirst, { preventDefault: true });
  useHotkeys('meta+right', ctx.selectLast, { preventDefault: true });

  return (
    <Flex gap={2}>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarPrevButton isDisabled={!shouldShowStagedImage} />
        <StagingAreaToolbarImageCountButton />
        <StagingAreaToolbarNextButton isDisabled={!shouldShowStagedImage} />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarAcceptButton />
        <StagingAreaToolbarToggleShowResultsButton />
        <StagingAreaToolbarSaveSelectedToGalleryButton />
        <StagingAreaToolbarMenu />
        <StagingAreaToolbarDiscardSelectedButton isDisabled={!shouldShowStagedImage} />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaAutoSwitchButtons />
      </ButtonGroup>
      <ButtonGroup borderRadius="base" shadow="dark-lg">
        <StagingAreaToolbarDiscardAllButton isDisabled={!shouldShowStagedImage} />
      </ButtonGroup>
    </Flex>
  );
});

StagingAreaToolbar.displayName = 'StagingAreaToolbar';

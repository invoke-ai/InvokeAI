import { ButtonGroup } from '@invoke-ai/ui-library';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { StagingAreaToolbarAcceptButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarAcceptButton';
import { StagingAreaToolbarDiscardAllButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardAllButton';
import { StagingAreaToolbarDiscardSelectedButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarDiscardSelectedButton';
import { StagingAreaToolbarImageCountButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarImageCountButton';
import { StagingAreaToolbarNextButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarNextButton';
import { StagingAreaToolbarPrevButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarPrevButton';
import { StagingAreaToolbarSaveAsMenu } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveAsMenu';
import { StagingAreaToolbarSaveSelectedToGalleryButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarSaveSelectedToGalleryButton';
import { StagingAreaToolbarToggleShowResultsButton } from 'features/controlLayers/components/StagingArea/StagingAreaToolbarToggleShowResultsButton';
import { memo, useEffect } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';

export const StagingAreaToolbar = memo(() => {
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

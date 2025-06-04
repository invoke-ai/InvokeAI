/* eslint-disable i18next/no-literal-string */

import { Divider, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useCanvasSessionContext } from 'features/controlLayers/components/SimpleSession/context';
import { getQueueItemElementId } from 'features/controlLayers/components/SimpleSession/shared';
import { StagingAreaContent } from 'features/controlLayers/components/SimpleSession/StagingAreaContent';
import { StagingAreaHeader } from 'features/controlLayers/components/SimpleSession/StagingAreaHeader';
import { StagingAreaNoItems } from 'features/controlLayers/components/SimpleSession/StagingAreaNoItems';
import { useProgressEvents } from 'features/controlLayers/components/SimpleSession/use-progress-events';
import { useStagingAreaKeyboardNav } from 'features/controlLayers/components/SimpleSession/use-staging-keyboard-nav';
import { memo, useEffect } from 'react';

export const StagingArea = memo(() => {
  const ctx = useCanvasSessionContext();
  const hasItems = useStore(ctx.$hasItems);
  useProgressEvents();
  useStagingAreaKeyboardNav();

  useEffect(() => {
    return ctx.$selectedItemId.listen((id) => {
      if (id !== null) {
        document.getElementById(getQueueItemElementId(id))?.scrollIntoView();
      }
    });
  }, [ctx.$selectedItemId]);

  return (
    <Flex flexDir="column" gap={2} w="full" h="full" minW={0} minH={0}>
      <StagingAreaHeader />
      <Divider />
      {hasItems && <StagingAreaContent />}
      {!hasItems && <StagingAreaNoItems />}
    </Flex>
  );
});
StagingArea.displayName = 'StagingArea';

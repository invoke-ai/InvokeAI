/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { StageComponent } from 'features/controlLayers/components/StageComponent';
import { StagingAreaIsStagingGate } from 'features/controlLayers/components/StagingArea/StagingAreaIsStagingGate';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasToolbar } from 'features/controlLayers/components/Toolbar/CanvasToolbar';
import { Transform } from 'features/controlLayers/components/Transform/Transform';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useRef } from 'react';

export const CanvasEditor = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('canvas', ref);

  return (
    <Flex
      tabIndex={-1}
      ref={ref}
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
    >
      <CanvasToolbar />
      <StageComponent />
      <Flex position="absolute" bottom={8} gap={2} align="center" justify="center">
        <CanvasManagerProviderGate>
          <StagingAreaIsStagingGate>
            <StagingAreaToolbar />
          </StagingAreaIsStagingGate>
        </CanvasManagerProviderGate>
      </Flex>
      <Flex position="absolute" bottom={8}>
        <CanvasManagerProviderGate>
          <Filter />
          <Transform />
        </CanvasManagerProviderGate>
      </Flex>
      <CanvasDropArea />
    </Flex>
  );
});

CanvasEditor.displayName = 'CanvasEditor';

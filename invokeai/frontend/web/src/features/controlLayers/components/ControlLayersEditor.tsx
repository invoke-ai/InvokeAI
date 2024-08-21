/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useScopeOnFocus } from 'common/hooks/interactionScopes';
import { CanvasDropArea } from 'features/controlLayers/components/CanvasDropArea';
import { ControlLayersToolbar } from 'features/controlLayers/components/ControlLayersToolbar';
import { Filter } from 'features/controlLayers/components/Filters/Filter';
import { StageComponent } from 'features/controlLayers/components/StageComponent';
import { StagingAreaToolbar } from 'features/controlLayers/components/StagingArea/StagingAreaToolbar';
import { CanvasManagerProviderGate } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { memo, useRef } from 'react';

export const CanvasEditor = memo(() => {
  const ref = useRef<HTMLDivElement>(null);
  useScopeOnFocus('canvas', ref);

  return (
    <Flex
      tabIndex={-1}
      ref={ref}
      layerStyle="first"
      p={2}
      borderRadius="base"
      position="relative"
      flexDirection="column"
      height="full"
      width="full"
      gap={2}
      alignItems="center"
      justifyContent="center"
    >
      <ControlLayersToolbar />
      <StageComponent />
      <Flex position="absolute" bottom={2} gap={2} align="center" justify="center">
        <StagingAreaToolbar />
      </Flex>
      <Flex position="absolute" bottom={16}>
        <CanvasManagerProviderGate>
          <Filter />
        </CanvasManagerProviderGate>
      </Flex>
      <CanvasDropArea />
    </Flex>
  );
});

CanvasEditor.displayName = 'CanvasEditor';

import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { CanvasEntityAdapterControlLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterControlLayer';
import type { CanvasEntityAdapterRasterLayer } from 'features/controlLayers/konva/CanvasEntity/CanvasEntityAdapterRasterLayer';
import { memo, useCallback } from 'react';

interface SelectObjectInputTypeButtonsProps {
  adapter: CanvasEntityAdapterRasterLayer | CanvasEntityAdapterControlLayer;
}

export const SelectObjectInputTypeButtons = memo(({ adapter }: SelectObjectInputTypeButtonsProps) => {
  const inputType = useStore(adapter.segmentAnything.$inputType);

  const setInputToVisual = useCallback(() => {
    adapter.segmentAnything.setInputType('visual');
  }, [adapter.segmentAnything]);

  const setInputToPrompt = useCallback(() => {
    adapter.segmentAnything.setInputType('prompt');
  }, [adapter.segmentAnything]);

  return (
    <ButtonGroup size="sm" variant="outline">
      <Button colorScheme={inputType === 'visual' ? 'invokeBlue' : undefined} onClick={setInputToVisual}>
        Visual
      </Button>
      <Button colorScheme={inputType === 'prompt' ? 'invokeBlue' : undefined} onClick={setInputToPrompt}>
        Prompt
      </Button>
    </ButtonGroup>
  );
});

SelectObjectInputTypeButtons.displayName = 'SelectObjectInputTypeButtons';

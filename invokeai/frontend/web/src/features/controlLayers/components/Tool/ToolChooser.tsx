import { ButtonGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { ToolBboxButton } from 'features/controlLayers/components/Tool/ToolBboxButton';
import { ToolBrushButton } from 'features/controlLayers/components/Tool/ToolBrushButton';
import { ToolEyeDropperButton } from 'features/controlLayers/components/Tool/ToolEyeDropperButton';
import { ToolMoveButton } from 'features/controlLayers/components/Tool/ToolMoveButton';
import { ToolRectButton } from 'features/controlLayers/components/Tool/ToolRectButton';
import { useCanvasDeleteLayerHotkey } from 'features/controlLayers/hooks/useCanvasDeleteLayerHotkey';
import { useCanvasResetLayerHotkey } from 'features/controlLayers/hooks/useCanvasResetLayerHotkey';

import { ToolEraserButton } from './ToolEraserButton';
import { ToolTransformButton } from './ToolTransformButton';
import { ToolViewButton } from './ToolViewButton';

export const ToolChooser: React.FC = () => {
  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  const isTransforming = useAppSelector((s) => s.canvasV2.tool.isTransforming);

  return (
    <>
      <ButtonGroup isAttached isDisabled={isTransforming}>
        <ToolBrushButton />
        <ToolEraserButton />
        <ToolRectButton />
        <ToolMoveButton />
        <ToolViewButton />
        <ToolBboxButton />
        <ToolEyeDropperButton />
      </ButtonGroup>
      <ToolTransformButton />
    </>
  );
};

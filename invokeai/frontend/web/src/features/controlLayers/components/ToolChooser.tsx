import { ButtonGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { BboxToolButton } from 'features/controlLayers/components/BboxToolButton';
import { BrushToolButton } from 'features/controlLayers/components/BrushToolButton';
import { EraserToolButton } from 'features/controlLayers/components/EraserToolButton';
import { MoveToolButton } from 'features/controlLayers/components/MoveToolButton';
import { RectToolButton } from 'features/controlLayers/components/RectToolButton';
import { TransformToolButton } from 'features/controlLayers/components/TransformToolButton';
import { ViewToolButton } from 'features/controlLayers/components/ViewToolButton';
import { useCanvasDeleteLayerHotkey } from 'features/controlLayers/hooks/useCanvasDeleteLayerHotkey';
import { useCanvasResetLayerHotkey } from 'features/controlLayers/hooks/useCanvasResetLayerHotkey';

export const ToolChooser: React.FC = () => {
  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  const isCanvasSessionActive = useAppSelector((s) => s.canvasV2.session.isActive);
  const isTransforming = useAppSelector((s) => s.canvasV2.tool.isTransforming);

  if (isCanvasSessionActive) {
    return (
      <>
        <ButtonGroup isAttached isDisabled={isTransforming}>
          <BrushToolButton />
          <EraserToolButton />
          <RectToolButton />
          <MoveToolButton />
          <ViewToolButton />
          <BboxToolButton />
        </ButtonGroup>
        <TransformToolButton />
      </>
    );
  }

  return (
    <ButtonGroup isAttached isDisabled={isTransforming}>
      <BrushToolButton />
      <EraserToolButton />
      <RectToolButton />
      <MoveToolButton />
      <ViewToolButton />
    </ButtonGroup>
  );
};

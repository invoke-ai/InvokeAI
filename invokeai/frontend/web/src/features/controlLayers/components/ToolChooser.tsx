import { ButtonGroup } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { BboxToolButton } from 'features/controlLayers/components/BboxToolButton';
import { BrushToolButton } from 'features/controlLayers/components/BrushToolButton';
import { EraserToolButton } from 'features/controlLayers/components/EraserToolButton';
import { MoveToolButton } from 'features/controlLayers/components/MoveToolButton';
import { RectToolButton } from 'features/controlLayers/components/RectToolButton';
import { ViewToolButton } from 'features/controlLayers/components/ViewToolButton';
import { useCanvasDeleteLayerHotkey } from 'features/controlLayers/hooks/useCanvasDeleteLayerHotkey';
import { useCanvasResetLayerHotkey } from 'features/controlLayers/hooks/useCanvasResetLayerHotkey';

export const ToolChooser: React.FC = () => {
  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  const isCanvasSessionActive = useAppSelector((s) => s.canvasV2.session.isActive);

  if (isCanvasSessionActive) {
    return (
      <ButtonGroup isAttached>
        <BrushToolButton />
        <EraserToolButton />
        <RectToolButton />
        <MoveToolButton />
        <ViewToolButton />
        <BboxToolButton />
      </ButtonGroup>
    );
  }

  return (
    <ButtonGroup isAttached>
      <BrushToolButton />
      <EraserToolButton />
      <RectToolButton />
      <MoveToolButton />
      <ViewToolButton />
    </ButtonGroup>
  );
};

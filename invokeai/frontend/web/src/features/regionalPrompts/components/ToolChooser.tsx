import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $tool } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutCardinalBold, PiEraserBold, PiPaintBrushBold } from 'react-icons/pi';

export const ToolChooser: React.FC = () => {
  const { t } = useTranslation();
  const tool = useStore($tool);

  const setToolToBrush = useCallback(() => {
    $tool.set('brush');
  }, []);
  useHotkeys('b', setToolToBrush, []);
  const setToolToEraser = useCallback(() => {
    $tool.set('eraser');
  }, []);
  useHotkeys('e', setToolToEraser, []);
  const setToolToMove = useCallback(() => {
    $tool.set('move');
  }, []);
  useHotkeys('v', setToolToMove, []);

  return (
    <ButtonGroup isAttached>
      <IconButton
        aria-label={`${t('unifiedCanvas.brush')} (B)`}
        tooltip={`${t('unifiedCanvas.brush')} (B)`}
        icon={<PiPaintBrushBold />}
        variant={tool === 'brush' ? 'solid' : 'outline'}
        onClick={setToolToBrush}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.eraser')} (E)`}
        tooltip={`${t('unifiedCanvas.eraser')} (E)`}
        icon={<PiEraserBold />}
        variant={tool === 'eraser' ? 'solid' : 'outline'}
        onClick={setToolToEraser}
      />
      <IconButton
        aria-label={`${t('unifiedCanvas.move')} (V)`}
        tooltip={`${t('unifiedCanvas.move')} (V)`}
        icon={<PiArrowsOutCardinalBold />}
        variant={tool === 'move' ? 'solid' : 'outline'}
        onClick={setToolToMove}
      />
    </ButtonGroup>
  );
};

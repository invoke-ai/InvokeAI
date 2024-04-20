import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $tool } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowsOutCardinalBold, PiEraserBold, PiPaintBrushBold } from 'react-icons/pi';

export const ToolChooser: React.FC = () => {
  const { t } = useTranslation();
  const tool = useStore($tool);
  const setToolToBrush = useCallback(() => {
    $tool.set('brush');
  }, []);
  const setToolToEraser = useCallback(() => {
    $tool.set('eraser');
  }, []);
  const setToolToMove = useCallback(() => {
    $tool.set('move');
  }, []);

  return (
    <ButtonGroup isAttached>
      <IconButton
        aria-label={t('unifiedCanvas.brush')}
        tooltip={t('unifiedCanvas.brush')}
        icon={<PiPaintBrushBold />}
        variant={tool === 'brush' ? 'solid' : 'outline'}
        onClick={setToolToBrush}
      />
      <IconButton
        aria-label={t('unifiedCanvas.eraser')}
        tooltip={t('unifiedCanvas.eraser')}
        icon={<PiEraserBold />}
        variant={tool === 'eraser' ? 'solid' : 'outline'}
        onClick={setToolToEraser}
      />
      <IconButton
        aria-label={t('unifiedCanvas.move')}
        tooltip={t('unifiedCanvas.move')}
        icon={<PiArrowsOutCardinalBold />}
        variant={tool === 'move' ? 'solid' : 'outline'}
        onClick={setToolToMove}
      />
    </ButtonGroup>
  );
};

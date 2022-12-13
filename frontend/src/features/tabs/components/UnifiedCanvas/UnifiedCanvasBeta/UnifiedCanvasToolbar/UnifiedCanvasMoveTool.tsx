import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setTool } from 'features/canvas/store/canvasSlice';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaArrowsAlt } from 'react-icons/fa';

export default function UnifiedCanvasMoveTool() {
  const tool = useAppSelector((state: RootState) => state.canvas.tool);
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  const handleSelectMoveTool = () => dispatch(setTool('move'));

  return (
    <IAIIconButton
      aria-label="Move Tool (V)"
      tooltip="Move Tool (V)"
      icon={<FaArrowsAlt />}
      data-selected={tool === 'move' || isStaging}
      onClick={handleSelectMoveTool}
    />
  );
}

import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { CSSProperties, memo } from 'react';
import { MiniMap } from 'reactflow';

const MinimapStyle: CSSProperties = {
  background: 'var(--invokeai-colors-base-500)',
};

const MinimapPanel = () => {
  const currentTheme = useAppSelector(
    (state: RootState) => state.ui.currentTheme
  );

  return (
    <MiniMap
      nodeStrokeWidth={3}
      pannable
      zoomable
      nodeBorderRadius={30}
      style={MinimapStyle}
      nodeColor={
        currentTheme === 'light'
          ? 'var(--invokeai-colors-accent-700)'
          : currentTheme === 'green'
          ? 'var(--invokeai-colors-accent-600)'
          : 'var(--invokeai-colors-accent-700)'
      }
      maskColor="var(--invokeai-colors-base-700)"
    />
  );
};

export default memo(MinimapPanel);

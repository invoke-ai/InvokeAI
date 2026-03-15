import { memo } from 'react';

import { StageViewer } from './StageViewer';

export const StageViewerPanel = memo(() => {
  return <StageViewer />;
});

StageViewerPanel.displayName = 'StageViewerPanel';

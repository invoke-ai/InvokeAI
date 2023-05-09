import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { RootState } from 'app/store/store';
import UnifiedCanvasContentBeta from './UnifiedCanvasBeta/UnifiedCanvasContentBeta';
import UnifiedCanvasContent from './UnifiedCanvasContent';

const CanvasWorkspace = () => {
  const shouldUseCanvasBetaLayout = useAppSelector(
    (state: RootState) => state.ui.shouldUseCanvasBetaLayout
  );

  return shouldUseCanvasBetaLayout ? (
    <UnifiedCanvasContentBeta />
  ) : (
    <UnifiedCanvasContent />
  );
};

export default memo(CanvasWorkspace);

import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import InvokeWorkarea from 'features/ui/components/common/InvokeWorkarea';
import UnifiedCanvasContentBeta from './UnifiedCanvasBeta/UnifiedCanvasContentBeta';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';

export default function UnifiedCanvasWorkarea() {
  const shouldUseCanvasBetaLayout = useAppSelector(
    (state: RootState) => state.ui.shouldUseCanvasBetaLayout
  );
  return (
    <InvokeWorkarea parametersPanel={<UnifiedCanvasParameters />}>
      {shouldUseCanvasBetaLayout ? (
        <UnifiedCanvasContentBeta />
      ) : (
        <UnifiedCanvasContent />
      )}
    </InvokeWorkarea>
  );
}

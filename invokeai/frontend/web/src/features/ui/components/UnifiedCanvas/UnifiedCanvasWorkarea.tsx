import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import UnifiedCanvasDisplayBeta from './UnifiedCanvasBeta/UnifiedCanvasDisplayBeta';
import UnifiedCanvasDisplay from './UnifiedCanvasDisplay';
import UnifiedCanvasPanel from './UnifiedCanvasPanel';

export default function UnifiedCanvasWorkarea() {
  const shouldUseCanvasBetaLayout = useAppSelector(
    (state: RootState) => state.ui.shouldUseCanvasBetaLayout
  );
  return (
    <InvokeWorkarea optionsPanel={<UnifiedCanvasPanel />}>
      {shouldUseCanvasBetaLayout ? (
        <UnifiedCanvasDisplayBeta />
      ) : (
        <UnifiedCanvasDisplay />
      )}
    </InvokeWorkarea>
  );
}

import UnifiedCanvasPanel from './UnifiedCanvasPanel';
import UnifiedCanvasDisplay from './UnifiedCanvasDisplay';
import InvokeWorkarea from 'features/tabs/components/InvokeWorkarea';
import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import UnifiedCanvasDisplayBeta from './UnifiedCanvasBeta/UnifiedCanvasDisplayBeta';

export default function UnifiedCanvasWorkarea() {
  const shouldUseCanvasBetaLayout = useAppSelector(
    (state: RootState) => state.options.shouldUseCanvasBetaLayout
  );
  return (
    <InvokeWorkarea
      optionsPanel={<UnifiedCanvasPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      {shouldUseCanvasBetaLayout ? (
        <UnifiedCanvasDisplayBeta />
      ) : (
        <UnifiedCanvasDisplay />
      )}
    </InvokeWorkarea>
  );
}

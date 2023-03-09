import { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import InvokeWorkarea from 'features/ui/components/InvokeWorkarea';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import UnifiedCanvasContentBeta from './UnifiedCanvasBeta/UnifiedCanvasContentBeta';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';

export default function UnifiedCanvasWorkarea() {
  const shouldUseCanvasBetaLayout = useAppSelector(
    (state: RootState) => state.ui.shouldUseCanvasBetaLayout
  );

  const activeTabName = useAppSelector(activeTabNameSelector);

  return (
    <InvokeWorkarea parametersPanelContent={<UnifiedCanvasParameters />}>
      {activeTabName === 'unifiedCanvas' &&
        (shouldUseCanvasBetaLayout ? (
          <UnifiedCanvasContentBeta />
        ) : (
          <UnifiedCanvasContent />
        ))}
    </InvokeWorkarea>
  );
}

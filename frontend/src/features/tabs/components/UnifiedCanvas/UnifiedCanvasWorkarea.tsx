import UnifiedCanvasPanel from './UnifiedCanvasPanel';
import UnifiedCanvasDisplay from './UnifiedCanvasDisplay';
import InvokeWorkarea from 'features/tabs/components/InvokeWorkarea';

export default function UnifiedCanvasWorkarea() {
  return (
    <InvokeWorkarea
      optionsPanel={<UnifiedCanvasPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <UnifiedCanvasDisplay />
    </InvokeWorkarea>
  );
}

import InpaintingPanel from './InpaintingPanel';
import InpaintingDisplay from './InpaintingDisplay';
import InvokeWorkarea from '../InvokeWorkarea';

export default function InpaintingWorkarea() {
  return (
    <InvokeWorkarea
      optionsPanel={<InpaintingPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <InpaintingDisplay />
    </InvokeWorkarea>
  );
}

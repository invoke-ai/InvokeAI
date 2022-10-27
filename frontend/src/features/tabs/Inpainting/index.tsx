import InpaintingPanel from './InpaintingPanel';
import InpaintingDisplay from './InpaintingDisplay';
import InvokeWorkarea from '../InvokeWorkarea';

export default function InpaintingWorkarea() {
  return (
    <InvokeWorkarea
      optionsPanel={<InpaintingPanel />}
      className="inpainting-workarea-container"
    >
      <InpaintingDisplay />
    </InvokeWorkarea>
  );
}

import InpaintingPanel from './InpaintingPanel';
import InpaintingDisplay from './InpaintingDisplay';
import InvokeWorkarea from 'features/tabs/InvokeWorkarea';

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

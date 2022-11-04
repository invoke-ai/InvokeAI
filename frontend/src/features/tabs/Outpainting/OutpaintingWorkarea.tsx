import OutpaintingPanel from './OutpaintingPanel';
import OutpaintingDisplay from './OutpaintingDisplay';
import InvokeWorkarea from 'features/tabs/InvokeWorkarea';

export default function OutpaintingWorkarea() {
  return (
    <InvokeWorkarea
      optionsPanel={<OutpaintingPanel />}
      styleClass="inpainting-workarea-overrides"
    >
      <OutpaintingDisplay />
    </InvokeWorkarea>
  );
}

import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ControlNetTab from './ControlNetTab/ControlNetTab';
import { setControlNet1 } from './store/controlnetSlice';

export default function ControlNetTab1() {
  const controlnet1 = useAppSelector(
    (state: RootState) => state.controlnet.controlnet1
  );
  return (
    <ControlNetTab
      label="ControlNet 1"
      controlnet={controlnet1}
      setControlnet={setControlNet1}
    />
  );
}

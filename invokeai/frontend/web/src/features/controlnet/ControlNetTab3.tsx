import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ControlNetTab from './ControlNetTab/ControlNetTab';
import { setControlNet3 } from './store/controlnetSlice';

export default function ControlNetTab3() {
  const controlnet3 = useAppSelector(
    (state: RootState) => state.controlnet.controlnet3
  );
  return (
    <ControlNetTab
      label="ControlNet 3"
      controlnet={controlnet3}
      setControlnet={setControlNet3}
    />
  );
}

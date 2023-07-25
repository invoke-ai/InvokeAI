import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { useIsRefinerAvailable } from 'features/sdxl/hooks/useIsRefinerAvailable';
import { setShouldUseSDXLRefiner } from 'features/sdxl/store/sdxlSlice';
import { ChangeEvent } from 'react';

export default function ParamUseSDXLRefiner() {
  const shouldUseSDXLRefiner = useAppSelector(
    (state: RootState) => state.sdxl.shouldUseSDXLRefiner
  );
  const isRefinerAvailable = useIsRefinerAvailable();

  const dispatch = useAppDispatch();

  const handleUseSDXLRefinerChange = (e: ChangeEvent<HTMLInputElement>) => {
    dispatch(setShouldUseSDXLRefiner(e.target.checked));
  };

  return (
    <IAISwitch
      label="Use Refiner"
      isChecked={shouldUseSDXLRefiner}
      onChange={handleUseSDXLRefinerChange}
      isDisabled={!isRefinerAvailable}
    />
  );
}

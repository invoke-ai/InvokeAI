import { ChangeEvent } from 'react';
import React from 'react';

import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRandomizeSeed } from 'features/options/store/optionsSlice';

export default function RandomizeSeed() {
  const dispatch = useAppDispatch();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  return (
    <IAISwitch
      label="Randomize Seed"
      isChecked={shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );
}

import React, { ChangeEvent } from 'react';

import {
  RootState,
  useAppDispatch,
  useAppSelector,
} from '../../../../app/store';
import SDSwitch from '../../../../common/components/SDSwitch';
import { setShouldRandomizeSeed } from '../../optionsSlice';

export default function RandomizeSeed() {
  const dispatch = useAppDispatch();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.options.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  return (
    <SDSwitch
      label="Randomize Seed"
      isChecked={shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );
}

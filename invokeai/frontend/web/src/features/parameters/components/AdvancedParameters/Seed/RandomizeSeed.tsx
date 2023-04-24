import { ChangeEvent, memo } from 'react';

import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { Switch } from '@chakra-ui/react';

// export default function RandomizeSeed() {
//   const dispatch = useAppDispatch();
//   const { t } = useTranslation();

//   const shouldRandomizeSeed = useAppSelector(
//     (state: RootState) => state.generation.shouldRandomizeSeed
//   );

//   const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
//     dispatch(setShouldRandomizeSeed(e.target.checked));

//   return (
//     <Switch
//       aria-label={t('parameters.randomizeSeed')}
//       isChecked={shouldRandomizeSeed}
//       onChange={handleChangeShouldRandomizeSeed}
//     />
//   );
// }

const SeedToggle = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(!e.target.checked));

  return (
    <Switch
      aria-label={t('parameters.randomizeSeed')}
      isChecked={!shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );
};

export default memo(SeedToggle);

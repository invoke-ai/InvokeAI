import { ChangeEvent, memo } from 'react';

import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { FormControl, FormLabel, Switch } from '@chakra-ui/react';

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

const ParamSeedRandomize = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  return (
    <FormControl
      sx={{
        display: 'flex',
        gap: 4,
        alignItems: 'center',
      }}
    >
      <FormLabel
        sx={{
          mb: 0,
          flexGrow: 1,
          fontSize: 'sm',
          fontWeight: 600,
          color: 'base.100',
        }}
      >
        {t('parameters.randomizeSeed')}
      </FormLabel>
      <Switch
        isChecked={shouldRandomizeSeed}
        onChange={handleChangeShouldRandomizeSeed}
      />
    </FormControl>
  );
};

export default memo(ParamSeedRandomize);

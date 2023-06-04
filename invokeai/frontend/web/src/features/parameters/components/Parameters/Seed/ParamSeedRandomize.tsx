import { ChangeEvent, memo } from 'react';

import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldRandomizeSeed } from 'features/parameters/store/generationSlice';
import { useTranslation } from 'react-i18next';
import { FormControl, FormLabel, Switch, Tooltip } from '@chakra-ui/react';
import IAISwitch from 'common/components/IAISwitch';

const ParamSeedRandomize = () => {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const shouldRandomizeSeed = useAppSelector(
    (state: RootState) => state.generation.shouldRandomizeSeed
  );

  const handleChangeShouldRandomizeSeed = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setShouldRandomizeSeed(e.target.checked));

  return (
    <IAISwitch
      label={t('common.random')}
      isChecked={shouldRandomizeSeed}
      onChange={handleChangeShouldRandomizeSeed}
    />
  );

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

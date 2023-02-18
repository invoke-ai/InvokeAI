import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAISwitch from 'common/components/IAISwitch';
import { setSeamless } from 'features/parameters/store/generationSlice';
import { ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Seamless tiling toggle
 */
const SeamlessSettings = () => {
  const dispatch = useAppDispatch();

  const seamless = useAppSelector(
    (state: RootState) => state.generation.seamless
  );

  const handleChangeSeamless = (e: ChangeEvent<HTMLInputElement>) =>
    dispatch(setSeamless(e.target.checked));

  const { t } = useTranslation();

  return (
    <Flex gap={2} direction="column">
      <IAISwitch
        label={t('parameters.seamlessTiling')}
        fontSize="md"
        isChecked={seamless}
        onChange={handleChangeSeamless}
      />
    </Flex>
  );
};

export default SeamlessSettings;

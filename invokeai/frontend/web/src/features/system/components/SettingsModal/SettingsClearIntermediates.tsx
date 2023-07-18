import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { StyledFlex } from './SettingsModal';
import { Heading, Text } from '@chakra-ui/react';
import IAIButton from '../../../../common/components/IAIButton';
import { useClearIntermediatesMutation } from '../../../../services/api/endpoints/images';
import { addToast } from '../../store/systemSlice';

export default function SettingsClearIntermediates() {
  const dispatch = useAppDispatch();
  const [isDisabled, setIsDisabled] = useState(false);

  const [clearIntermediates, { isLoading: isLoadingClearIntermediates, data }] =
    useClearIntermediatesMutation();

  const handleClickClearIntermediates = useCallback(() => {
    clearIntermediates({});
  }, [clearIntermediates]);

  useEffect(() => {
    if (data >= 0) {
      dispatch(
        addToast({
          title:
            data === 0
              ? `No intermediates to clear`
              : `Successfully cleared ${data} intermediates`,
          status: 'info',
        })
      );
      if (data < 100) {
        setIsDisabled(true);
      }
    }
  }, [data, dispatch]);

  return (
    <StyledFlex>
      <Heading size="sm">Clear Intermediates</Heading>
      <IAIButton
        colorScheme="error"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={isDisabled}
      >
        {isDisabled ? 'No more intermedates to clear' : 'Clear Intermediates'}
      </IAIButton>
      <Text>
        Will permanently delete first 100 intermediates found on disk and in
        database
      </Text>
    </StyledFlex>
  );
}

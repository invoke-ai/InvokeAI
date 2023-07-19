import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useCallback, useEffect, useState } from 'react';
import { StyledFlex } from './SettingsModal';
import { Heading, Text } from '@chakra-ui/react';
import IAIButton from '../../../../common/components/IAIButton';
import { useClearIntermediatesMutation } from '../../../../services/api/endpoints/images';
import { addToast } from '../../store/systemSlice';
import { resetCanvas } from '../../../canvas/store/canvasSlice';

export default function SettingsClearIntermediates() {
  const dispatch = useAppDispatch();
  const [isDisabled, setIsDisabled] = useState(false);

  const [clearIntermediates, { isLoading: isLoadingClearIntermediates }] =
    useClearIntermediatesMutation();

  const handleClickClearIntermediates = useCallback(() => {
    clearIntermediates({})
      .unwrap()
      .then((response) => {
        dispatch(resetCanvas());
        dispatch(
          addToast({
            title:
              response === 0
                ? `No intermediates to clear`
                : `Successfully cleared ${response} intermediates`,
            status: 'info',
          })
        );
        if (response < 100) {
          setIsDisabled(true);
        }
      });
  }, [clearIntermediates, dispatch]);

  return (
    <StyledFlex>
      <Heading size="sm">Clear Intermediates</Heading>
      <IAIButton
        colorScheme="error"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={isDisabled}
      >
        {isDisabled ? 'Intermediates Cleared' : 'Clear 100 Intermediates'}
      </IAIButton>
      <Text>
        Will permanently delete first 100 intermediates found on disk and in
        database
      </Text>
      <Text fontWeight="bold">This will also clear your canvas state.</Text>
      <Text>
        Intermediate images are byproducts of generation, different from the
        result images in the gallery. Purging intermediates will free disk
        space. Your gallery images will not be deleted.
      </Text>
    </StyledFlex>
  );
}

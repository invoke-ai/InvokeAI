import { Heading, Text } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useCallback, useEffect } from 'react';
import IAIButton from '../../../../common/components/IAIButton';
import {
  useClearIntermediatesMutation,
  useGetIntermediatesCountQuery,
} from '../../../../services/api/endpoints/images';
import { resetCanvas } from '../../../canvas/store/canvasSlice';
import { addToast } from '../../store/systemSlice';
import { StyledFlex } from './SettingsModal';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';

export default function SettingsClearIntermediates() {
  const dispatch = useAppDispatch();

  const { data: intermediatesCount, refetch: updateIntermediatesCount } =
    useGetIntermediatesCountQuery();

  const [clearIntermediates, { isLoading: isLoadingClearIntermediates }] =
    useClearIntermediatesMutation();

  const handleClickClearIntermediates = useCallback(() => {
    clearIntermediates()
      .unwrap()
      .then((response) => {
        dispatch(controlNetReset());
        dispatch(resetCanvas());
        dispatch(
          addToast({
            title: `Cleared ${response} intermediates`,
            status: 'info',
          })
        );
      });
  }, [clearIntermediates, dispatch]);

  useEffect(() => {
    // update the count on mount
    updateIntermediatesCount();
  }, [updateIntermediatesCount]);

  return (
    <StyledFlex>
      <Heading size="sm">Clear Intermediates</Heading>
      <IAIButton
        colorScheme="warning"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={!intermediatesCount}
      >
        {intermediatesCount
          ? `Clear ${
              intermediatesCount >= 100 ? 100 : intermediatesCount
            } Intermediates`
          : 'Intermediates Cleared'}
      </IAIButton>
      <Text>
        Permanently delete the first 100 intermediates found on disk and in
        database. There are currently {intermediatesCount ?? 0} stored
        intermediates.
      </Text>
      <Text fontWeight="bold">
        Clearing intermediates will reset your Canvas and ControlNet state.
      </Text>
      <Text sx={{ fontSize: 'sm' }} variant="subtext">
        Intermediate images are byproducts of generation, different from the
        result images in the gallery. Clearing intermediates will free disk
        space. Your gallery images will not be deleted.
      </Text>
    </StyledFlex>
  );
}

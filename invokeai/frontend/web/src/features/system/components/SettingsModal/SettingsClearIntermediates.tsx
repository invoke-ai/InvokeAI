import { Heading, Text } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { controlNetReset } from 'features/controlNet/store/controlNetSlice';
import { useCallback, useEffect } from 'react';
import IAIButton from '../../../../common/components/IAIButton';
import {
  useClearIntermediatesMutation,
  useGetIntermediatesCountQuery,
} from '../../../../services/api/endpoints/images';
import { resetCanvas } from '../../../canvas/store/canvasSlice';
import { addToast } from '../../store/systemSlice';
import StyledFlex from './StyledFlex';

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

  const buttonText = intermediatesCount
    ? `Clear ${intermediatesCount} Intermediate${
        intermediatesCount > 1 ? 's' : ''
      }`
    : 'No Intermediates to Clear';

  return (
    <StyledFlex>
      <Heading size="sm">Clear Intermediates</Heading>
      <IAIButton
        colorScheme="warning"
        onClick={handleClickClearIntermediates}
        isLoading={isLoadingClearIntermediates}
        isDisabled={!intermediatesCount}
      >
        {buttonText}
      </IAIButton>
      <Text fontWeight="bold">
        Clearing intermediates will reset your Canvas and ControlNet state.
      </Text>
      <Text variant="subtext">
        Intermediate images are byproducts of generation, different from the
        result images in the gallery. Clearing intermediates will free disk
        space.
      </Text>
      <Text variant="subtext">Your gallery images will not be deleted.</Text>
    </StyledFlex>
  );
}

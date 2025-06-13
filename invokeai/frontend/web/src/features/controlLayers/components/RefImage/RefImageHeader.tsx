import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { refImageDeleted } from 'features/controlLayers/store/refImagesSlice';
import { memo, useCallback } from 'react';
import { PiTrashBold } from 'react-icons/pi';

export const RefImageHeader = memo(() => {
  const id = useRefImageIdContext();
  const dispatch = useAppDispatch();
  const entity = useRefImageEntity(id);
  const deleteRefImage = useCallback(() => {
    dispatch(refImageDeleted({ id }));
  }, [dispatch, id]);

  return (
    <Flex justifyContent="space-between" alignItems="center" w="full">
      {entity.config.image !== null && (
        <Text fontWeight="semibold" color="base.300">
          Reference Image
        </Text>
      )}
      {entity.config.image === null && (
        <Text fontWeight="semibold" color="base.300">
          Reference Image - No Image Selected
        </Text>
      )}
      <IconButton
        size="xs"
        variant="link"
        alignSelf="stretch"
        icon={<PiTrashBold />}
        onClick={deleteRefImage}
        aria-label="Delete reference image"
        colorScheme="error"
      />
    </Flex>
  );
});
RefImageHeader.displayName = 'RefImageHeader';

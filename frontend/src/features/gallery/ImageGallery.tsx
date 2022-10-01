import { Button, Flex, Text } from '@chakra-ui/react';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch } from '../../app/store';
import { useAppSelector } from '../../app/store';
import HoverableImage from './HoverableImage';

/**
 * Simple image gallery.
 */
const ImageGallery = () => {
  const { images, currentImageUuid, areMoreImagesAvailable } = useAppSelector(
    (state: RootState) => state.gallery
  );
  const dispatch = useAppDispatch();
  /**
   * I don't like that this needs to rerender whenever the current image is changed.
   * What if we have a large number of images? I suppose pagination (planned) will
   * mitigate this issue.
   *
   * TODO: Refactor if performance complaints, or after migrating to new API which supports pagination.
   */

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  return (
    <Flex direction={'column'} gap={2} pb={2}>
      {images.length ? (
        <Flex gap={2} wrap="wrap">
          {images.map((image) => {
            const { uuid } = image;
            const isSelected = currentImageUuid === uuid;
            return (
              <HoverableImage
                key={uuid}
                image={image}
                isSelected={isSelected}
              />
            );
          })}
        </Flex>
      ) : (
        <Text size={'xl'} padding={5} textAlign={'center'}>
          No images in gallery
        </Text>
      )}
      <Button
        onClick={handleClickLoadMore}
        isDisabled={!areMoreImagesAvailable}
      >
        {areMoreImagesAvailable ? 'Load more' : 'All images loaded'}
      </Button>
    </Flex>
  );
};

export default ImageGallery;

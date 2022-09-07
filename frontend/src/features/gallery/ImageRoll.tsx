import { Image, VStack } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { setCurrentImage } from './gallerySlice';

const ImageRoll = () => {
    const { images, currentImageUuid } = useAppSelector(
        (state: RootState) => state.gallery
    );
    const dispatch = useAppDispatch();

    return (
        <VStack pb={2}>
            {[...images].reverse().map((image) => {
                const { url, uuid } = image;
                return (
                    <Image
                        border={currentImageUuid === uuid ? '3px' : 0}
                        borderColor='green'
                        rounded={'md'}
                        key={uuid}
                        onClick={() => dispatch(setCurrentImage(uuid))}
                        src={url}
                    />
                );
            })}
        </VStack>
    );
};

export default ImageRoll;

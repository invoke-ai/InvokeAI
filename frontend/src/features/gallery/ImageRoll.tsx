import { Box, IconButton, Image, VStack } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { MdDeleteForever } from 'react-icons/md';
import { setCurrentImage } from './gallerySlice';
import { useSocketIOEmitters } from '../../context/socket';

const ImageRoll = () => {
    const { images } = useAppSelector((state: RootState) => state.gallery);
    const dispatch = useAppDispatch();
    const { deleteImage } = useSocketIOEmitters();

    return (
        <VStack>
            {[...images].reverse().map((image) => {
                const { url, uuid } = image;
                return (
                    <Image
                        borderRadius={'md'}
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

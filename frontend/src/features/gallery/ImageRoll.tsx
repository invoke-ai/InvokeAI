import { Box, IconButton, Image, VStack } from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { MdDeleteForever } from 'react-icons/md';
import { deleteImage, setCurrentImage } from '../../app/sdSlice';

const ImageRoll = () => {
    const { images } = useAppSelector((state: RootState) => state.sd);
    const dispatch = useAppDispatch();

    return (
        <VStack>
            {images.map((image, i) => (
                <Box key={image.url} position={'relative'}>
                    <IconButton
                        position={'absolute'}
                        right={0}
                        top={0}
                        aria-label='Delete image'
                        icon={<MdDeleteForever />}
                        onClick={() => dispatch(deleteImage(i))}
                    />
                    <Image
                        onClick={() => dispatch(setCurrentImage(i))}
                        src={image.url}
                    />
                </Box>
            ))}
        </VStack>
    );
};

export default ImageRoll;

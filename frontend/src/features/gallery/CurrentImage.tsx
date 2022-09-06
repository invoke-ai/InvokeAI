import { Center, Flex, IconButton, Image } from '@chakra-ui/react';
import { MdDeleteForever } from 'react-icons/md';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../context/socket';

const height = 'calc(100vh - 176px)';

const CurrentImage = () => {
    const { currentImageUuid, images } = useAppSelector(
        (state: RootState) => state.sd
    );
    const imageToDisplay = images.find(
        (image) => image.uuid === currentImageUuid
    );
    const { deleteImage } = useSocketIOEmitters();
    return (
        <Center height={height}>
            {imageToDisplay && (
                <Flex>
                    <Image
                        maxHeight={height}
                        fit='contain'
                        src={imageToDisplay?.url}
                    />
                    <IconButton
                        aria-label='Delete image'
                        icon={<MdDeleteForever />}
                        onClick={() => deleteImage(imageToDisplay?.uuid)}
                    />
                </Flex>
            )}
        </Center>
    );
};

export default CurrentImage;

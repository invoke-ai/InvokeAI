import { Center, Image } from '@chakra-ui/react';
import { useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import fallbackImgUrl from '../../assets/images/rick.jpeg';

const height = 'calc(100vh - 176px)';

const CurrentImage = () => {
    const { currentImageIndex, images } = useAppSelector(
        (state: RootState) => state.sd
    );
    const imageToDisplay = images[currentImageIndex];

    return (
        <Center height={height}>
            <Image
                maxHeight={height}
                fit='contain'
                src={imageToDisplay?.url}
                fallbackSrc={fallbackImgUrl}
            />
        </Center>
    );
};

export default CurrentImage;

import { Center, Flex, Image, useColorModeValue } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { setInitialImagePath } from '../sd/sdSlice';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import DeleteImageModalButton from './DeleteImageModalButton';
import SDButton from '../../components/SDButton';
import { runESRGAN, runGFPGAN } from '../../app/socketio';

const height = 'calc(100vh - 270px)';

const CurrentImage = () => {
    const { currentImage, intermediateImage } = useAppSelector(
        (state: RootState) => state.gallery
    );

    const { isGFPGANAvailable, isESRGANAvailable } = useAppSelector(
        (state: RootState) => state.system
    );
    const dispatch = useAppDispatch();

    const bgColor = useColorModeValue(
        'rgba(255, 255, 255, 0.85)',
        'rgba(0, 0, 0, 0.8)'
    );

    const [shouldShowImageDetails, setShouldShowImageDetails] =
        useState<boolean>(false);

    const imageToDisplay = intermediateImage || currentImage;

    return (
        <Flex direction={'column'} rounded={'md'} borderWidth={1} p={2} gap={2}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <SDButton
                        label='Use as initial image'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        onClick={() =>
                            dispatch(setInitialImagePath(imageToDisplay.url))
                        }
                    />
                    <SDButton
                        label='Details'
                        colorScheme={'gray'}
                        variant={shouldShowImageDetails ? 'solid' : 'outline'}
                        flexGrow={1}
                        onClick={() =>
                            setShouldShowImageDetails(!shouldShowImageDetails)
                        }
                    />
                    <SDButton
                        label='Upscale'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        isDisabled={
                            !isESRGANAvailable || Boolean(intermediateImage)
                        }
                        onClick={() => dispatch(runESRGAN(imageToDisplay))}
                    />
                    <SDButton
                        label='Fix faces'
                        colorScheme={'gray'}
                        flexGrow={1}
                        variant={'outline'}
                        isDisabled={
                            !isGFPGANAvailable || Boolean(intermediateImage)
                        }
                        onClick={() => dispatch(runGFPGAN(imageToDisplay))}
                    />
                    <DeleteImageModalButton image={imageToDisplay}>
                        <SDButton
                            label='Delete'
                            colorScheme={'red'}
                            flexGrow={1}
                            variant={'outline'}
                            isDisabled={Boolean(intermediateImage)}
                        />
                    </DeleteImageModalButton>
                </Flex>
            )}
            <Center height={height} position={'relative'}>
                {imageToDisplay && (
                    <Image
                        src={imageToDisplay.url}
                        fit='contain'
                        maxWidth={'100%'}
                        maxHeight={'100%'}
                    />
                )}
                {imageToDisplay && shouldShowImageDetails && (
                    <Flex
                        width={'100%'}
                        height={'100%'}
                        position={'absolute'}
                        top={0}
                        left={0}
                        p={3}
                        boxSizing='border-box'
                        backgroundColor={bgColor}
                        overflow='scroll'
                    >
                        <ImageMetadataViewer image={imageToDisplay} />
                    </Flex>
                )}
            </Center>
        </Flex>
    );
};

export default CurrentImage;

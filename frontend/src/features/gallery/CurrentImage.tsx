import {
    Box,
    Center,
    Flex,
    IconButton,
    Image,
    Link,
    Tooltip,
    useColorModeValue,
    VStack,
} from '@chakra-ui/react';
import { FaCopy, FaRecycle } from 'react-icons/fa';
import { RiBracesFill } from 'react-icons/ri';
import { GiResize } from 'react-icons/gi';
import { IoOpen } from 'react-icons/io5';
import { MdFaceRetouchingNatural } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { setAllParameters, setInitialImagePath } from '../sd/sdSlice';
import { useState } from 'react';
import ImageMetadataViewer from './ImageMetadataViewer';
import DeleteImageModalButton from './DeleteImageModalButton';

const height = 'calc(100vh - 216px)';

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
        <Center height={height}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <Box position={'relative'}>
                        <Image
                            height={height}
                            src={imageToDisplay.url}
                            fit='contain'
                            // minWidth={512}
                            // minHeight={512}
                        />
                        {shouldShowImageDetails && (
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
                    </Box>
                    <VStack>
                        <Tooltip label='Use as initial image'>
                            <IconButton
                                fontSize={18}
                                onClick={() =>
                                    dispatch(
                                        setInitialImagePath(imageToDisplay.url)
                                    )
                                }
                                aria-label='Use as initial image'
                                icon={<FaRecycle />}
                            />
                        </Tooltip>
                        <Tooltip label='Use all parameters'>
                            <IconButton
                                fontSize={18}
                                aria-label='Use all parameters'
                                icon={<FaCopy />}
                                onClick={() =>
                                    dispatch(
                                        setAllParameters(
                                            imageToDisplay.metadata
                                        )
                                    )
                                }
                            />
                        </Tooltip>
                        <Tooltip label='Image details'>
                            <IconButton
                                fontSize={20}
                                colorScheme={
                                    shouldShowImageDetails ? 'green' : undefined
                                }
                                aria-label='Image details'
                                icon={<RiBracesFill />}
                                onClick={() =>
                                    setShouldShowImageDetails(
                                        !shouldShowImageDetails
                                    )
                                }
                            />
                        </Tooltip>
                        <Tooltip label='Open in new tab' shouldWrapChildren>
                            <Link isExternal href={imageToDisplay.url}>
                                <IconButton
                                    fontSize={20}
                                    aria-label='Open in new tab'
                                    icon={<IoOpen />}
                                />
                            </Link>
                        </Tooltip>
                        <Tooltip label='Upscale (ESRGAN)' shouldWrapChildren>
                            <IconButton
                                isDisabled={!isESRGANAvailable}
                                fontSize={20}
                                aria-label='Upscale (ESRGAN)'
                                icon={<GiResize />}
                            />
                        </Tooltip>
                        <Tooltip label='Fix faces (GFPGAN)' shouldWrapChildren>
                            <IconButton
                                isDisabled={!isGFPGANAvailable}
                                fontSize={20}
                                aria-label='Fix faces (GFPGAN)'
                                icon={<MdFaceRetouchingNatural />}
                            />
                        </Tooltip>
                        {!intermediateImage && (
                            <DeleteImageModalButton
                                image={imageToDisplay}
                                // uuid={imageToDisplay.uuid}
                                // url={imageToDisplay.url}
                            />
                        )}
                    </VStack>
                </Flex>
            )}
        </Center>
    );
};

export default CurrentImage;

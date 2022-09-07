import {
    Box,
    Center,
    Flex,
    IconButton,
    Image,
    List,
    ListItem,
    Text,
    Tooltip,
    useColorModeValue,
    VStack,
} from '@chakra-ui/react';
import { FaCopy, FaPlus, FaRecycle } from 'react-icons/fa';
import { RiBracesFill } from 'react-icons/ri';
import { GiResize } from 'react-icons/gi';
import { MdDeleteForever, MdFaceRetouchingNatural } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../app/socket';
import {
    setAllParameters,
    setInitialImagePath,
    setParameter,
} from '../sd/sdSlice';
import { useState } from 'react';
import { SDMetadata } from './gallerySlice';
import { PARAMETERS } from '../../app/constants';
import ImageParameters from './ImageParameters';

const height = 'calc(100vh - 176px)';

const CurrentImage = () => {
    const { emitDeleteImage } = useSocketIOEmitters();
    const { currentImageUuid, images } = useAppSelector(
        (state: RootState) => state.gallery
    );
    const { isGFPGANAvailable, isESRGANAvailable } = useAppSelector(
        (state: RootState) => state.system
    );
    const dispatch = useAppDispatch();

    const [shouldShowImageDetails, setShouldShowImageDetails] =
        useState<boolean>(false);

    const imageToDisplay = images.find(
        (image) => image.uuid === currentImageUuid
    );

    const bgColor = useColorModeValue(
        'rgba(255, 255, 255, 0.85)',
        'rgba(0, 0, 0, 0.8)'
    );

    return (
        <Center height={height}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <Box position={'relative'}>
                        <Image maxHeight={height} src={imageToDisplay?.url} />
                        {shouldShowImageDetails && (
                            <Flex
                                width={'100%'}
                                height={'100%'}
                                position={'absolute'}
                                top={0}
                                left={0}
                                p={5}
                                backgroundColor={bgColor}
                            >
                                <ImageParameters image={imageToDisplay} />
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
                        <Tooltip label='Upscale (ESRGAN)' shouldWrapChildren>
                            <IconButton
                                isDisabled={isESRGANAvailable ? false : true}
                                fontSize={20}
                                aria-label='Upscale (ESRGAN)'
                                icon={<GiResize />}
                            />
                        </Tooltip>
                        <Tooltip label='Fix faces (GFPGAN)' shouldWrapChildren>
                            <IconButton
                                isDisabled={isGFPGANAvailable ? false : true}
                                fontSize={20}
                                aria-label='Fix faces (GFPGAN)'
                                icon={<MdFaceRetouchingNatural />}
                            />
                        </Tooltip>
                        <Tooltip label='Delete'>
                            <IconButton
                                aria-label='Delete'
                                icon={<MdDeleteForever />}
                                fontSize={24}
                                onClick={() =>
                                    emitDeleteImage(imageToDisplay?.uuid)
                                }
                            />
                        </Tooltip>
                    </VStack>
                </Flex>
            )}
        </Center>
    );
};

export default CurrentImage;

import {
    Center,
    Flex,
    IconButton,
    Image,
    Tooltip,
    VStack,
} from '@chakra-ui/react';
import { FaCopy, FaPaintBrush, FaRecycle, FaSeedling } from 'react-icons/fa';
import { RiBracesFill } from 'react-icons/ri';
import { GiResize } from 'react-icons/gi';
import { MdDeleteForever, MdFaceRetouchingNatural } from 'react-icons/md';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../app/socket';
import { setInitialImagePath } from '../sd/sdSlice';

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

    const imageToDisplay = images.find(
        (image) => image.uuid === currentImageUuid
    );

    return (
        <Center height={height}>
            {imageToDisplay && (
                <Flex gap={2}>
                    <Image maxHeight={height} src={imageToDisplay?.url} />
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
                            />
                        </Tooltip>
                        <Tooltip label='Use individual parameters'>
                            <IconButton
                                fontSize={20}
                                aria-label='Use individual parameters'
                                icon={<RiBracesFill />}
                            />
                        </Tooltip>
                        <Tooltip label='Use seed'>
                            <IconButton
                                fontSize={18}
                                aria-label='Use seed'
                                icon={<FaSeedling />}
                            />
                        </Tooltip>
                        <Tooltip label='Create inpainting mask'>
                            <IconButton
                                fontSize={18}
                                aria-label='Create inpainting mask'
                                icon={<FaPaintBrush />}
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

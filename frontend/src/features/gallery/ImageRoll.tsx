import {
    Box,
    Flex,
    Icon,
    Image,
    useColorModeValue,
} from '@chakra-ui/react';
import { RootState } from '../../app/store';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { setCurrentImage } from './gallerySlice';
import { FaCheck } from 'react-icons/fa';
import DeleteImageModalButton from './DeleteImageModalButton';

const ImageRoll = () => {
    const { images, currentImageUuid } = useAppSelector(
        (state: RootState) => state.gallery
    );

    const bgColor = useColorModeValue('gray.200', 'gray.700');
    const overlayColor = useColorModeValue(
        'rgba(255,255,255,0.7)',
        'rgba(0,0,0,0.7)'
    );
    const checkColor = useColorModeValue('green.600', 'green.300');
    const dispatch = useAppDispatch();

    return (
        <Flex gap={2} wrap='wrap' pb={2}>
            {[...images].reverse().map((image) => {
                const { url, uuid } = image;
                const isSelected = currentImageUuid === uuid;
                return (
                    <Box position={'relative'}>
                        <Image
                            width={120}
                            height={120}
                            objectFit='cover'
                            rounded={'md'}
                            key={uuid}
                            src={url}
                            loading={'lazy'}
                            backgroundColor={bgColor}
                        />
                        <Flex
                            cursor={'pointer'}
                            position={'absolute'}
                            top={0}
                            left={0}
                            rounded={'md'}
                            width='100%'
                            height='100%'
                            alignItems={'center'}
                            justifyContent={'center'}
                            backgroundColor={
                                isSelected ? overlayColor : undefined
                            }
                            onClick={() => dispatch(setCurrentImage(uuid))}
                        >
                            {isSelected && (
                                <Icon
                                    fill={checkColor}
                                    width={'50%'}
                                    height={'50%'}
                                    as={FaCheck}
                                />
                            )}
                            <DeleteImageModalButton
                                position={'absolute'}
                                top={1}
                                right={1}
                                uuid={uuid}
                                size='xs'
                                fontSize={18}
                                colorScheme='red'
                            />
                        </Flex>
                    </Box>
                );
            })}
        </Flex>
    );
};

export default ImageRoll;

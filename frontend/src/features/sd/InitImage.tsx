import {
  Box,
  Center,
  Flex,
  Icon,
  IconButton,
  Image,
  Text,
  Tooltip,
  useColorModeValue,
  useToast,
} from '@chakra-ui/react';
import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaMask, FaUpload } from 'react-icons/fa';
import { RiCloseFill } from 'react-icons/ri';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { resetInitialImagePath } from '../../features/sd/sdSlice';
import MaskUploader from './MaskUploader';
import './InitImage.css';
import { uploadInitialImage } from '../../app/socketio';

const InitImage = () => {
  const toast = useToast();
  const dispatch = useAppDispatch();
  const iconColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('gray.300', 'gray.500');
  const bgColor = useColorModeValue('gray.100', 'gray.700');
  const { initialImagePath, maskPath } = useAppSelector(
    (state: RootState) => state.sd
  );

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: any) => {
      fileRejections.forEach((rejection: any) => {
        const msg = rejection.errors.reduce(
          (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
          ''
        );

        toast({
          title: 'Upload failed',
          description: msg,
          status: 'error',
          isClosable: true,
        });
      });

      acceptedFiles.forEach((file: File) => {
        dispatch(uploadInitialImage(file));
      });
    },
    []
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg', '.png'],
    },
  });

  const [shouldShowMask, setShouldShowMask] = useState<boolean>(false);

  const handleMouseOver = () => setShouldShowMask(true);
  const handleMouseOut = () => setShouldShowMask(false);

  return (
    <div
      {...getRootProps({
        onClick: initialImagePath ? (e) => e.stopPropagation() : undefined,
      })}
    >
      <input {...getInputProps({ multiple: false })} />
      <Box
        rounded={'md'}
        border={initialImagePath ? undefined : '1px'}
        borderColor={iconColor}
        mt={2}
        backgroundColor={isDragActive ? bgColor : undefined}
      >
        <Center height={280} width={280}>
          <Flex
            direction={'column'}
            alignItems={'center'}
            position={'relative'}
          >
            {initialImagePath ? (
              <>
                <Flex
                  direction={'column'}
                  position={'absolute'}
                  top={2}
                  right={2}
                  gap={2}
                  zIndex={2}
                >
                  <Tooltip label={'Reset initial image & mask'}>
                    <IconButton
                      aria-label='Reset initial image & mask'
                      icon={<RiCloseFill />}
                      fontSize={24}
                      colorScheme='red'
                      onClick={() => dispatch(resetInitialImagePath())}
                    />
                  </Tooltip>
                  <Tooltip label='Upload new initial image'>
                    <IconButton
                      aria-label='Upload new initial image'
                      icon={<FaUpload />}
                      fontSize={20}
                      colorScheme='blue'
                      onClick={open}
                    />
                  </Tooltip>
                  <MaskUploader>
                    <Tooltip
                      label={
                        maskPath ? 'Upload new mask image' : 'Upload mask image'
                      }
                    >
                      <IconButton
                        onMouseOver={handleMouseOver}
                        onMouseOut={handleMouseOut}
                        aria-label='Upload mask'
                        icon={<FaMask />}
                        fontSize={20}
                        colorScheme='blue'
                        onClick={open}
                      />
                    </Tooltip>
                  </MaskUploader>
                </Flex>
                <Image
                  maxHeight={270}
                  maxWidth={270}
                  src={initialImagePath}
                  rounded={'md'}
                />
                {!shouldShowMask && maskPath && (
                  <Image
                    position={'absolute'}
                    top={0}
                    left={0}
                    maxHeight={270}
                    maxWidth={270}
                    src={maskPath}
                    rounded={'md'}
                    zIndex={1}
                  />
                )}
              </>
            ) : (
              <>
                <Text textColor={textColor}>Upload initial image</Text>
                <Icon fontSize={136} color={iconColor} as={FaUpload} pt={7} />
              </>
            )}
          </Flex>
        </Center>
      </Box>
    </div>
  );
};

export default InitImage;

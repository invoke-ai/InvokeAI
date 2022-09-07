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
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaUpload } from 'react-icons/fa';
import { RiCloseFill } from 'react-icons/ri';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { RootState } from '../../app/store';
import { useSocketIOEmitters } from '../../app/socket';
import { resetInitialImagePath } from '../../features/sd/sdSlice';

const InitialImage = () => {
  const toast = useToast();
  const dispatch = useAppDispatch();
  const iconColor = useColorModeValue('gray.200', 'gray.600');
  const textColor = useColorModeValue('gray.300', 'gray.500');
  const bgColor = useColorModeValue('gray.100', 'gray.700');
  const { initialImagePath } = useAppSelector((state: RootState) => state.sd);
  const { emitUploadInitialImage } = useSocketIOEmitters();

  const onDrop = useCallback(
    (acceptedFiles: Array<File>, fileRejections: any) => {
      fileRejections.forEach((rejection: any) => {
        const msg = rejection.errors.reduce(
          (acc: string, cur: { message: string }) => acc + '\n' + cur.message,
          ''
        );

        toast({
          title: 'Upload failed.',
          description: msg,
          status: 'error',
          isClosable: true,
        });
      });

      acceptedFiles.forEach((file: File) => {
        emitUploadInitialImage(file, file.name);
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

  return (
    <div
      {...getRootProps({
        onClick: initialImagePath ? (e) => e.stopPropagation() : undefined,
      })}
    >
      <input {...getInputProps()} />
      <Box
        rounded={'md'}
        border={initialImagePath ? undefined : '1px'}
        borderColor={iconColor}
        m={'10px'}
        backgroundColor={isDragActive ? bgColor : undefined}
      >
        <Center height={280} width={280} pr={2}>
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
                >
                  <Tooltip label='Clear initial image'>
                    <IconButton
                      aria-label='Clear initial image'
                      icon={<RiCloseFill />}
                      fontSize={24}
                      onClick={() => dispatch(resetInitialImagePath())}
                    />
                  </Tooltip>
                  <Tooltip label='Upload initial image'>
                    <IconButton
                      aria-label='Upload initial image'
                      icon={<FaUpload />}
                      fontSize={20}
                      onClick={open}
                    />
                  </Tooltip>
                </Flex>
                <Image
                  maxHeight={270}
                  maxWidth={270}
                  src={initialImagePath}
                  rounded={'md'}
                />
              </>
            ) : (
              <>
                <Text textColor={textColor}>Upload initial image</Text>
                <Icon fontSize={136} color={iconColor} as={FaUpload} pt={5} />
              </>
            )}
          </Flex>
        </Center>
      </Box>
    </div>
  );
};

export default InitialImage;

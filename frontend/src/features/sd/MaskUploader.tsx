import { IconButton, Tooltip, useToast } from '@chakra-ui/react';
import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaMask } from 'react-icons/fa';
import { useAppDispatch, useAppSelector } from '../../app/hooks';
import { uploadMaskImage } from '../../app/socketio';
import { RootState } from '../../app/store';

type Props = {
    setShouldShowMask: (shouldShowMask: boolean) => void;
};

const MaskUploader = ({ setShouldShowMask }: Props) => {
    const dispatch = useAppDispatch();
    const toast = useToast();
    const { maskPath } = useAppSelector((state: RootState) => state.sd);

    const onDrop = useCallback(
        (acceptedFiles: Array<File>, fileRejections: any) => {
            fileRejections.forEach((rejection: any) => {
                const msg = rejection.errors.reduce(
                    (acc: string, cur: { message: string }) =>
                        acc + '\n' + cur.message,
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
                dispatch(uploadMaskImage(file));
            });
        },
        []
    );

    const { getRootProps, getInputProps, open } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg', '.png'],
        },
    });

    const handleMouseOver = () => setShouldShowMask(true);
    const handleMouseOut = () => setShouldShowMask(false);

    return (
        <div {...getRootProps()}>
            <input {...getInputProps({ multiple: false })} />
            <Tooltip
                label={maskPath ? 'Upload new mask image' : 'Upload mask image'}
            >
                <IconButton
                    onMouseOver={maskPath ? handleMouseOver : undefined}
                    onMouseOut={maskPath ? handleMouseOut : undefined}
                    aria-label='Upload mask'
                    icon={<FaMask />}
                    fontSize={20}
                    colorScheme='blue'
                    onClick={open}
                />
            </Tooltip>
        </div>
    );
};

export default MaskUploader;

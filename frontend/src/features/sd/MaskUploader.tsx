import { useToast } from '@chakra-ui/react';
import { ReactElement, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useAppDispatch } from '../../app/hooks';
import { uploadMaskImage } from '../../app/socketio';

type Props = {
    children: ReactElement;
};

const MaskUploader = ({ children }: Props) => {
    const dispatch = useAppDispatch();
    const toast = useToast();

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

    const { getRootProps, getInputProps } = useDropzone({
        onDrop,
        accept: {
            'image/jpeg': ['.jpg', '.jpeg', '.png'],
        },
    });

    return (
        <div {...getRootProps()}>
            <input {...getInputProps({ multiple: false })} />
            {children}
        </div>
    );
};

export default MaskUploader;

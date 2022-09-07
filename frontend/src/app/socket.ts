import { useToast } from '@chakra-ui/react';
import { useEffect } from 'react';
import { io } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { useAppDispatch, useAppSelector } from '../app/hooks';
import {
    setIsConnected,
    setIsProcessing,
    setCurrentStep,
    setSocketId,
} from '../features/system/systemSlice';
import { RootState } from '../app/store';
import {
    addImage,
    deleteImage,
    SDMetadata,
    setGalleryImages,
} from '../features/gallery/gallerySlice';
import { setInitialImagePath, setMaskPath } from '../features/sd/sdSlice';

export const socket = io('http://localhost:9090');

interface SocketIOResponse {
    status: 'OK' | 'ERROR';
    message?: string;
    data?: any;
}

/*
Sets up socketio listeners which interact with state.
Manages sync between server and UI, namely the image gallery.
Called only once in App.tsx.

Socketio does no error handling on the client (here) so must catch all errors.
See: https://socket.io/docs/v4/listening-to-events/#error-handling
Very fun figuring that out.
*/
export const useSocketIOListeners = () => {
    const dispatch = useAppDispatch();
    // Makes a toast alert on change in connection status, not sure how to implement this elsewhere
    const toast = useToast();

    useEffect(() => {
        socket.on('connect', () => {
            try {
                dispatch(setIsConnected(true));
                dispatch(setSocketId(socket.id));

                // maintain sync with local images
                socket.emit(
                    'requestAllImages',
                    (response: SocketIOResponse) => {
                        try {
                            response.status === 'OK' &&
                                dispatch(setGalleryImages(response.data));
                        } catch (e) {
                            console.error(e);
                        }
                    }
                );

                toast({
                    title: 'Connected',
                    status: 'success',
                    isClosable: true,
                });
            } catch (e) {
                console.error(e);
            }
        });

        socket.on('disconnect', () => {
            try {
                dispatch(setIsConnected(false));
                dispatch(setIsProcessing(false));
                toast({
                    title: 'Disconnected',
                    status: 'error',
                    isClosable: true,
                });
            } catch (e) {
                console.error(e);
            }
        });

        socket.on('progress', (data: { step: number }) => {
            try {
                dispatch(setIsProcessing(true));
                dispatch(setCurrentStep(data.step + 1));
            } catch (e) {
                console.error(e);
            }
        });

        socket.on('result', (data: { url: string; metadata: SDMetadata }) => {
            try {
                const uuid = uuidv4();
                const { url, metadata } = data;
                dispatch(
                    addImage({
                        uuid,
                        url,
                        metadata,
                    })
                );
                dispatch(setIsProcessing(false));
            } catch (e) {
                console.error(e);
            }
        });

        // clean up all listeners
        return () => {
            socket.off('connect');
            socket.off('disconnect');
            socket.off('progress');
            socket.off('result');
        };
    }, []);
};

/*
Provides emitters which interact with state.
*/
export const useSocketIOEmitters = () => {
    const dispatch = useAppDispatch();

    const {
        prompt,
        imagesToGenerate,
        steps,
        cfgScale,
        height,
        width,
        sampler,
        seed,
        img2imgStrength,
        gfpganStrength,
        upscalingLevel,
        upscalingStrength,
        initialImagePath,
        maskPath,
        seamless,
        shouldFitToWidthHeight,
    } = useAppSelector((state: RootState) => state.sd);

    const { images } = useAppSelector((state: RootState) => state.gallery);

    return {
        emitGenerateImage: () => {
            dispatch(setIsProcessing(true));
            dispatch(setCurrentStep(-1));
            socket.emit('generateImage', {
                prompt,
                imagesToGenerate,
                steps,
                cfgScale,
                height,
                width,
                sampler,
                seed,
                img2imgStrength,
                gfpganStrength,
                upscalingLevel,
                upscalingStrength,
                initialImagePath,
                maskPath,
                shouldFitToWidthHeight,
                seamless,
            });
        },
        emitDeleteImage: (uuid: string) => {
            const imageToDelete = images.find((image) => image.uuid === uuid);
            imageToDelete &&
                socket.emit(
                    'deleteImage',
                    imageToDelete.url,
                    (response: SocketIOResponse) => {
                        response.status === 'OK' && dispatch(deleteImage(uuid));
                    }
                );
        },
        emitCancel: () => {
            socket.emit('cancel', (response: SocketIOResponse) => {
                response.status === 'OK' && dispatch(setIsProcessing(false));
            });
        },
        emitUploadInitialImage: (file: File, name: string) => {
            socket.emit(
                'uploadInitialImage',
                file,
                name,
                (response: SocketIOResponse) => {
                    response.status === 'OK' &&
                        dispatch(setInitialImagePath(response.data));
                }
            );
        },
        emitUploadMask: (file: File, name: string) => {
            socket.emit(
                'uploadMask',
                file,
                name,
                (response: SocketIOResponse) => {
                    response.status === 'OK' &&
                        dispatch(setMaskPath(response.data));
                }
            );
        },
        emitRequestAllImages: () =>
            socket.emit('requestAllImages', (response: SocketIOResponse) => {
                dispatch(setGalleryImages(response.data));
            }),
    };
};

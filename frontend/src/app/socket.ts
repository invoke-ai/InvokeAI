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
    addLogEntry,
} from '../features/system/systemSlice';
import { RootState } from '../app/store';
import {
    addImage,
    deleteImage,
    SDMetadata,
    setGalleryImages,
    setIntermediateImage,
    clearIntermediateImage,
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
                dispatch(addLogEntry(`Connected to server: ${socket.id}`));
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
                dispatch(addLogEntry(`Disconnected from server`));
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
                dispatch(setCurrentStep(data.step));
            } catch (e) {
                console.error(e);
            }
        });

        socket.on(
            'intermediateResult',
            (data: { url: string; metadata: SDMetadata }) => {
                try {
                    const uuid = uuidv4();
                    const { url, metadata } = data;
                    dispatch(
                        setIntermediateImage({
                            uuid,
                            url,
                            metadata,
                        })
                    );
                    dispatch(
                        addLogEntry(`Intermediate image generated: ${url}`)
                    );
                } catch (e) {
                    console.error(e);
                }
            }
        );

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
                dispatch(addLogEntry(`Image generated: ${url}`));
            } catch (e) {
                console.error(e);
            }
        });

        // clean up all listeners
        return () => {
            socket.off('connect');
            socket.off('disconnect');
            socket.off('progress');
            socket.off('intermediateResult');
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

    const { shouldDisplayInProgress } = useAppSelector(
        (state: RootState) => state.system
    );

    const { images, intermediateImage } = useAppSelector(
        (state: RootState) => state.gallery
    );

    const generationParameters = {
        // from sd slice
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
        // from system slid
        shouldDisplayInProgress,
    };

    return {
        emitGenerateImage: () => {
            dispatch(setIsProcessing(true));
            dispatch(setCurrentStep(-1));
            socket.emit('generateImage', generationParameters);
            dispatch(
                addLogEntry(
                    `Image generation requested ${JSON.stringify(
                        generationParameters
                    )}`
                )
            );
        },
        emitDeleteImage: (uuid: string) => {
            const imageToDelete = images.find((image) => image.uuid === uuid);
            imageToDelete &&
                socket.emit(
                    'deleteImage',
                    imageToDelete.url,
                    (response: SocketIOResponse) => {
                        if (response.status === 'OK') {
                            dispatch(deleteImage(uuid));
                            dispatch(
                                addLogEntry(
                                    `Image deleted ${imageToDelete.url}`
                                )
                            );
                        }
                    }
                );
        },
        emitCancel: () => {
            socket.emit('cancel', (response: SocketIOResponse) => {
                if (response.status === 'OK') {
                    dispatch(setIsProcessing(false));
                    if (intermediateImage) {
                        dispatch(addImage(intermediateImage));
                        dispatch(
                            addLogEntry(
                                `Intermediate image saved ${intermediateImage.url}`
                            )
                        );

                        dispatch(clearIntermediateImage());
                    }
                    dispatch(addLogEntry(`Image generation canceled`));
                }
            });
        },
        emitUploadInitialImage: (file: File, name: string) => {
            socket.emit(
                'uploadInitialImage',
                file,
                name,
                (response: SocketIOResponse) => {
                    if (response.status === 'OK') {
                        dispatch(setInitialImagePath(response.data));
                        dispatch(
                            addLogEntry(
                                `Initial image uploaded ${response.data}`
                            )
                        );
                    }
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
                    dispatch(
                        addLogEntry(`Mask image uploaded ${response.data}`)
                    );
                }
            );
        },
        emitRequestAllImages: () =>
            socket.emit('requestAllImages', (response: SocketIOResponse) => {
                dispatch(setGalleryImages(response.data));
                dispatch(addLogEntry(`Syncing gallery`));
            }),
    };
};

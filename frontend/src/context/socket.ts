/*
Socket.io setup.
Context used only for socket.io activities that do not touch redux.
*/

import { useToast } from '@chakra-ui/react';
import { createContext, useEffect } from 'react';
import { io } from 'socket.io-client';
import { v4 as uuidv4 } from 'uuid';
import { useAppDispatch, useAppSelector } from '../app/hooks';
import {
    setIsConnected,
    setIsProcessing,
    setProgress,
} from '../features/system/systemSlice';
import { RootState } from '../app/store';
import {
    addImage,
    deleteImage,
    setGalleryImages,
} from '../features/gallery/gallerySlice';
import { setInitialImagePath } from '../features/sd/sdSlice';

// Single instance of the client, shared across the app via two hooks and Context
export const socket = io('http://localhost:9090');
// In case a component needs to communicate without touching redux, it can useContext()
// to to access the single client instance
export const SocketContext = createContext(socket);

interface SocketIOResponse {
    status: 'OK' | 'ERROR';
    message?: string;
    data?: any;
}

// Hook sets up socketio listeners that touch redux and initializes gallery state, called only once in App.tsx
export const useSocketIOListeners = () => {
    const dispatch = useAppDispatch();
    const toast = useToast();

    useEffect(() => {
        socket.on('connect', () => {
            dispatch(setIsConnected(true));
            // maintain gallery sync
            socket.emit('requestAllImages', (response: SocketIOResponse) => {
                dispatch(setGalleryImages(response.data));
            });
            toast({
                title: 'Connected',
                status: 'success',
                isClosable: true,
            });
        });

        socket.on('disconnect', () => {
            dispatch(setIsConnected(false));
            toast({
                title: 'Disconnected',
                status: 'error',
                isClosable: true,
            });
        });

        socket.on('progress', (data: { step: number; steps: number }) =>
            dispatch(setProgress(Math.round((data.step / data.steps) * 100)))
        );

        socket.on('result', (data: { url: string }) => {
            const uuid = uuidv4();
            dispatch(
                addImage({
                    uuid,
                    url: data.url,
                    metadata: {
                        prompt: 'test',
                    },
                })
            );
            dispatch(setIsProcessing(false));
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

// Hook provides emitters that interact with redux. Ex:
// const { generateImage } = useSocketIOEmitters()
// const config = {...} // prompt, steps, etc
// generateImage(config)
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
    } = useAppSelector((state: RootState) => state.sd);

    const { images } = useAppSelector((state: RootState) => state.gallery);

    return {
        emitGenerateImage: () => {
            dispatch(setIsProcessing(true));
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
        emitRequestAllImages: () =>
            socket.emit('requestAllImages', (response: SocketIOResponse) => {
                dispatch(setGalleryImages(response.data));
            }),
    };
};

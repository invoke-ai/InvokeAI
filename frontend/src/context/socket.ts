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

// Single instance of the client, shared across the app via two hooks and Context
export const socket = io('http://localhost:9090');

// In case a component needs to communicate without touching redux, it can useContext()
// to to access the single client instance
export const SocketContext = createContext(socket);

// Hook sets up socketio listeners that touch redux and initializes gallery state, called only once in App.tsx
export const useSocketIOInitialize = () => {
    const dispatch = useAppDispatch();
    const toast = useToast();
    socket.emit('requestAllImages', (data: Array<string>) => {
        dispatch(setGalleryImages(data));
    });

    useEffect(() => {
        socket.on('connect', () => {
            dispatch(setIsConnected(true));
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

        socket.on('progress', (value: number) =>
            dispatch(setProgress(Math.round(value * 100)))
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
        // images,
    } = useAppSelector((state: RootState) => state.sd);

    const { images } = useAppSelector((state: RootState) => state.gallery);

    return {
        generateImage: () => {
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
            });
        },
        deleteImage: (uuid: string) => {
            const imageToDelete = images.find((image) => image.uuid === uuid);
            imageToDelete &&
                socket.emit(
                    'deleteImage',
                    imageToDelete.url,
                    (response: string) => {
                        response === 'ok' && dispatch(deleteImage(uuid));
                    }
                );
        },
        cancel: () => {
            socket.emit('cancel', (response: string) => {
                response === 'ok' && dispatch(setIsProcessing(false));
            });
        },
    };
};

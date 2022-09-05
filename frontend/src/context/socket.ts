/*
Socket.io setup.
Context used only for socket.io activities that do not touch redux.
*/

import { createContext, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useAppDispatch, useAppSelector } from '../app/hooks';
import {
    addImage,
    deleteImage,
    setGalleryImages,
    setIsConnected,
    setIsProcessing,
    setProgress,
} from '../app/sdSlice';
import { RootState } from '../app/store';

// Single instance of the client, shared across the app via two hooks and Context
export const socket = io('http://localhost:9090');

// In case a component needs to communicate without touching redux, it can useContext()
// to to access the single client instance
export const SocketContext = createContext(socket);

// Hook provides listeners that touch redux and initializes gallery state, called only once in App.tsx
export const useSocketIOListeners = () => {
    const dispatch = useAppDispatch();

    useEffect(() => {
        socket.on('connect', () => {
            dispatch(setIsConnected(true));
        });

        socket.on('disconnect', () => {
            dispatch(setIsConnected(false));
        });

        socket.on('progress', (value: number) =>
            dispatch(setProgress(Math.round(value * 100)))
        );

        socket.on('result', (data: { url: string }) => {
            dispatch(
                addImage({
                    url: data.url,
                    metadata: {
                        prompt: 'test',
                    },
                })
            );
        });

        socket.on('sendAllImages', (data: Array<string>) => {
            console.log(data);
            dispatch(setGalleryImages(data));
        });

        // clean up all listeners
        return () => {
            socket.off('connect');
            socket.off('disconnect');
            socket.off('progress');
            socket.off('result');
        };
    }, []);

    socket.emit('requestAllImages');
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
        images,
    } = useAppSelector((state: RootState) => state.sd);

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
        deleteImage: (id: number) => {
            socket.emit('deleteImage', images[id].url, (response: string) => {
                response === 'ok' && dispatch(deleteImage(id));
            });
        },
    };
};

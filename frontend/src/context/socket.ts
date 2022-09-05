import { createContext, useEffect } from 'react';
import { io } from 'socket.io-client';
import { useAppDispatch } from '../app/hooks';
import {
    addImage,
    setGalleryImages,
    setIsConnected,
    setProgress,
} from '../app/sdSlice';

export const socket = io('http://localhost:9090');
export const SocketContext = createContext(socket);

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

        return () => {
            socket.off('connect');
            socket.off('disconnect');
            socket.off('progress');
            socket.off('result');
        };
    }, []);
};

export const useSocketIOEmitters = () => {
    return {
        generateImage: (data: any) => socket.emit('generateImage', data),
    };
};

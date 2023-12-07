import { isRejectedWithValue } from '@reduxjs/toolkit'
import type { MiddlewareAPI, Middleware } from '@reduxjs/toolkit'
import { addToast } from 'features/system/store/systemSlice';

export const authToastMiddleware: Middleware =
    (api: MiddlewareAPI) => (next) => (action) => {
        if (isRejectedWithValue(action)) {
            if (action.payload.status === 403) {
                const { dispatch } = api;
                const customMessage = action.payload.data.detail !== "Forbidden" ? action.payload.data.detail : undefined;
                dispatch(
                    addToast({
                        title: 'Something went wrong',
                        status: 'error',
                        description: customMessage,
                    })
                );
            }
        }

        return next(action)
    }
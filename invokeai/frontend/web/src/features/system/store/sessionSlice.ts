// TODO: split system slice inot this

// import type { PayloadAction } from '@reduxjs/toolkit';
// import { createSlice } from '@reduxjs/toolkit';
// import { socketSubscribed, socketUnsubscribed } from 'services/events/actions';

// export type SessionState = {
//   /**
//    * The current socket session id
//    */
//   sessionId: string;
//   /**
//    * Whether the current session is a canvas session. Needed to manage the staging area.
//    */
//   isCanvasSession: boolean;
//   /**
//    * When a session is canceled, its ID is stored here until a new session is created.
//    */
//   canceledSessionId: string;
// };

// export const initialSessionState: SessionState = {
//   sessionId: '',
//   isCanvasSession: false,
//   canceledSessionId: '',
// };

// export const sessionSlice = createSlice({
//   name: 'session',
//   initialState: initialSessionState,
//   reducers: {
//     sessionIdChanged: (state, action: PayloadAction<string>) => {
//       state.sessionId = action.payload;
//     },
//     isCanvasSessionChanged: (state, action: PayloadAction<boolean>) => {
//       state.isCanvasSession = action.payload;
//     },
//   },
//   extraReducers: (builder) => {
//     /**
//      * Socket Subscribed
//      */
//     builder.addCase(socketSubscribed, (state, action) => {
//       state.sessionId = action.payload.sessionId;
//       state.canceledSessionId = '';
//     });

//     /**
//      * Socket Unsubscribed
//      */
//     builder.addCase(socketUnsubscribed, (state) => {
//       state.sessionId = '';
//     });
//   },
// });

// export const { sessionIdChanged, isCanvasSessionChanged } =
//   sessionSlice.actions;

// export default sessionSlice.reducer;

export default {};

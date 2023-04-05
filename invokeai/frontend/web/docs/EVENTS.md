# Events

Events via `socket.io`

## `actions.ts`

Redux actions for all socket events. Payloads all include a timestamp, and optionally some other data.

Any reducer (or middleware) can respond to the actions.

## `middleware.ts`

Redux middleware for events.

Handles dispatching the event actions. Only put logic here if it can't really go anywhere else.

For example, on connect we want to load images to the gallery if it's not populated. This requires dispatching a thunk, so we need to directly dispatch this in the middleware.

## `types.ts`

Hand-written types for the socket events. Cannot generate these from the server, but fortunately they are few and simple.

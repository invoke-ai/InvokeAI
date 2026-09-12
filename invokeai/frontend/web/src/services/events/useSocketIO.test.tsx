// @vitest-environment happy-dom
/**
 * Mounted-DOM coverage for the socket's lifecycle against the auth session.
 *
 * The rule this pins down is which auth changes may replace the live socket. A sliding-window
 * token refresh must not: it lands about once a minute during any activity, and rebuilding the
 * socket for it disposed the event listeners and glitched the preview mid-generation. A change of
 * user must, or the socket keeps the previous account's rooms and private events.
 */
import { Buffer } from 'node:buffer';

import { createStore } from 'app/store/store';
import {
  currentUserUpdated,
  externalTokenAdopted,
  logout,
  setCredentials,
  tokenRefreshed,
} from 'features/auth/store/authSlice';
import { act } from 'react';
import type { Root } from 'react-dom/client';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useSocketIO } from './useSocketIO';

type SocketIOOptions = Partial<ManagerOptions & SocketOptions>;

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const createFakeSocket = () => {
  const handlers = new Map<string, Set<(...args: unknown[]) => void>>();
  return {
    connect: vi.fn(),
    disconnect: vi.fn(),
    on: vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      handlers.set(event, (handlers.get(event) ?? new Set()).add(handler));
    }),
    off: vi.fn((event: string, handler: (...args: unknown[]) => void) => {
      handlers.get(event)?.delete(handler);
    }),
    emit: vi.fn(),
    /** Drive the socket's own listeners, as socket.io does when the connection ends. */
    fire: (event: string, ...args: unknown[]) => {
      handlers.get(event)?.forEach((handler) => handler(...args));
    },
  };
};

const sockets: ReturnType<typeof createFakeSocket>[] = [];
const io = vi.fn((_url: string, _options: SocketIOOptions) => {
  const socket = createFakeSocket();
  sockets.push(socket);
  return socket;
});

vi.mock('socket.io-client', () => ({ io: (url: string, options: SocketIOOptions) => io(url, options) }));
vi.mock('services/events/setEventListeners', () => ({ setEventListeners: () => () => {} }));

const user = {
  user_id: 'user-1',
  email: 'user@example.com',
  display_name: null,
  is_admin: false,
  is_active: true,
};

/**
 * A token for `userId`. `nonce` varies the bytes without varying the identity, as a sliding-window
 * refresh does; `epoch` is the revocation epoch the server bumps when a password change kills the
 * account's earlier sessions.
 */
const tokenFor = (userId: string, nonce = 0, epoch = 0) =>
  `header.${Buffer.from(JSON.stringify({ user_id: userId, nonce, token_epoch: epoch })).toString('base64url')}.signature`;

const Probe = () => {
  useSocketIO();
  return null;
};

/** The `auth` option as socket.io will invoke it: a callback, resolved at each connection attempt. */
const resolveAuth = (options: SocketIOOptions | undefined): object | undefined => {
  const auth = options?.auth;
  if (typeof auth !== 'function') {
    return auth;
  }
  let resolved: object | undefined;
  auth((data) => {
    resolved = data;
  });
  return resolved;
};

describe('useSocketIO (mounted)', () => {
  let container: HTMLDivElement;
  let root: Root;
  let store: ReturnType<typeof createStore>;

  const mount = () => {
    act(() => {
      root.render(
        <Provider store={store}>
          <Probe />
        </Provider>
      );
    });
  };

  beforeEach(() => {
    vi.useFakeTimers();
    sockets.length = 0;
    io.mockClear();
    localStorage.clear();
    store = createStore();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.useRealTimers();
  });

  it('keeps the live socket across a sliding-window token refresh', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();
    expect(io).toHaveBeenCalledTimes(1);

    act(() => {
      store.dispatch(tokenRefreshed(tokenFor(user.user_id, 2)));
    });

    expect(io).toHaveBeenCalledTimes(1);
    expect(sockets[0]?.disconnect).not.toHaveBeenCalled();
  });

  it('presents the current token on every connection attempt, not the one it was built with', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    const options = io.mock.calls[0]?.[1];
    expect(resolveAuth(options)).toEqual({ token: tokenFor(user.user_id, 1) });

    act(() => {
      store.dispatch(tokenRefreshed(tokenFor(user.user_id, 2)));
    });

    // Same socket, same options object — but a reconnect now sends the refreshed token.
    expect(resolveAuth(options)).toEqual({ token: tokenFor(user.user_id, 2) });
  });

  // Note this one is the hydration gate's, not the session key's: `externalTokenAdopted` nulls
  // `auth.user`, which closes the gate on its own. The session key's account-switch case is the
  // same-tab `setCredentials` test below, where the gate never closes.
  it('waits for the new account to hydrate before reconnecting when another tab takes over', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    act(() => {
      store.dispatch(externalTokenAdopted(tokenFor('user-2', 1)));
    });

    // The adopted token nulls the current user, so no socket may connect until /me rehydrates:
    // events arriving before that would be attributed to the wrong owner.
    expect(sockets[0]?.disconnect).toHaveBeenCalledTimes(1);
    expect(io).toHaveBeenCalledTimes(1);

    act(() => {
      store.dispatch(currentUserUpdated({ ...user, user_id: 'user-2' }));
    });

    expect(io).toHaveBeenCalledTimes(2);
    expect(resolveAuth(io.mock.calls[1]?.[1])).toEqual({ token: tokenFor('user-2', 1) });
  });

  it('rebuilds the socket when a password change revokes the earlier session', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    // A password change bumps the revocation epoch and the server force-disconnects every socket
    // that authenticated under the old one. socket.io never retries a server-initiated
    // disconnect, so the replacement token — which carries the new epoch — has to bring a new
    // socket with it, or this session has no events until the page is reloaded.
    act(() => {
      store.dispatch(tokenRefreshed(tokenFor(user.user_id, 2, 1)));
    });

    expect(sockets[0]?.disconnect).toHaveBeenCalledTimes(1);
    expect(io).toHaveBeenCalledTimes(2);
    expect(resolveAuth(io.mock.calls[1]?.[1])).toEqual({ token: tokenFor(user.user_id, 2, 1) });
  });

  it('rebuilds the socket when a different account signs in without an intervening logout', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    // One action swaps token and user together, so the hydration gate never closes — only the
    // session key separates this from a refresh.
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor('user-2', 1), user: { ...user, user_id: 'user-2' } }));
    });

    expect(sockets[0]?.disconnect).toHaveBeenCalledTimes(1);
    expect(io).toHaveBeenCalledTimes(2);
    expect(resolveAuth(io.mock.calls[1]?.[1])).toEqual({ token: tokenFor('user-2', 1) });
  });

  it('reconnects after a server-initiated disconnect, which socket.io never retries', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();
    expect(sockets[0]?.connect).toHaveBeenCalledTimes(1);

    act(() => {
      sockets[0]?.fire('disconnect', 'io server disconnect');
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });

    // The same socket, reconnected — not a rebuild, so listeners and rooms survive.
    expect(sockets[0]?.connect).toHaveBeenCalledTimes(2);
    expect(io).toHaveBeenCalledTimes(1);
  });

  it('leaves transport-level drops to socket.io', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    act(() => {
      sockets[0]?.fire('disconnect', 'transport close');
    });
    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    // socket.io's own backoff owns this one; a second driver would race it.
    expect(sockets[0]?.connect).toHaveBeenCalledTimes(1);
  });

  it('gives up after a bounded number of server disconnects', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    for (let attempt = 0; attempt < 8; attempt++) {
      act(() => {
        sockets[0]?.fire('disconnect', 'io server disconnect');
      });
      act(() => {
        vi.advanceTimersByTime(60_000);
      });
    }

    // 1 mount + 5 retries: a server that keeps dropping this socket must not be retried forever.
    expect(sockets[0]?.connect).toHaveBeenCalledTimes(6);
  });

  it('does not reconnect a socket the session has already discarded', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    act(() => {
      sockets[0]?.fire('disconnect', 'io server disconnect');
    });
    // Logout lands inside the retry delay: the pending timer must not revive the old socket.
    act(() => {
      store.dispatch(logout());
    });
    act(() => {
      vi.advanceTimersByTime(60_000);
    });

    expect(sockets[0]?.connect).toHaveBeenCalledTimes(1);
  });

  it('tears the socket down on logout', () => {
    act(() => {
      store.dispatch(setCredentials({ token: tokenFor(user.user_id, 1), user }));
    });
    mount();

    act(() => {
      store.dispatch(logout());
    });

    expect(sockets[0]?.disconnect).toHaveBeenCalledTimes(1);
    // No token is single-user mode, where an unauthenticated socket is the correct one.
    expect(io).toHaveBeenCalledTimes(2);
    expect(resolveAuth(io.mock.calls[1]?.[1])).toBeUndefined();
  });
});

import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext } from 'react';

const CanvasInstanceContext = createContext<string | null>(null);

export const CanvasInstanceContextProvider = memo(({ canvasId, children }: PropsWithChildren<{ canvasId: string }>) => {
  return <CanvasInstanceContext.Provider value={canvasId}>{children}</CanvasInstanceContext.Provider>;
});
CanvasInstanceContextProvider.displayName = 'CanvasInstanceContextProvider';

export const useScopedCanvasIdSafe = () => {
  return useContext(CanvasInstanceContext);
};

import { Middleware } from '@reduxjs/toolkit';
import { store } from 'app/store/store';
import { PartialAppConfig } from 'app/types/invokeai';
import React, {
  lazy,
  memo,
  PropsWithChildren,
  ReactNode,
  useEffect,
} from 'react';
import { Provider } from 'react-redux';
import { addMiddleware, resetMiddlewares } from 'redux-dynamic-middlewares';
import { socketMiddleware } from 'services/events/middleware';
import Loading from '../../common/components/Loading/Loading';
import '../../i18n';
import AppDndContext from '../../features/dnd/components/AppDndContext';

const App = lazy(() => import('./App'));
const ThemeLocaleProvider = lazy(() => import('./ThemeLocaleProvider'));

interface Props extends PropsWithChildren {
  apiUrl?: string;
  token?: string;
  config?: PartialAppConfig;
  headerComponent?: ReactNode;
  middleware?: Middleware[];
  projectId?: string;
  selectedImage?: {
    imageName: string;
    action: 'sendToImg2Img' | 'sendToCanvas' | 'useAllParameters';
  };
}

const InvokeAIUI = ({
  apiUrl,
  token,
  config,
  headerComponent,
  middleware,
  projectId,
  selectedImage,
}: Props) => {
  useEffect(() => {
    // reset dynamically added middlewares
    resetMiddlewares();

    // TODO: at this point, after resetting the middleware, we really ought to clean up the socket
    // stuff by calling `dispatch(socketReset())`. but we cannot dispatch from here as we are
    // outside the provider. it's not needed until there is the possibility that we will change
    // the `apiUrl`/`token` dynamically.

    // rebuild socket middleware with token and apiUrl
    if (middleware && middleware.length > 0) {
      addMiddleware(socketMiddleware(), ...middleware);
    } else {
      addMiddleware(socketMiddleware());
    }
  }, [middleware]);

  return (
    <React.StrictMode>
      <Provider store={store}>
        <React.Suspense fallback={<Loading />}>
          <ThemeLocaleProvider>
            <AppDndContext>
              <App
                config={config}
                headerComponent={headerComponent}
                selectedImage={selectedImage}
                apiUrl={apiUrl}
                token={token}
                projectId={projectId}
              />
            </AppDndContext>
          </ThemeLocaleProvider>
        </React.Suspense>
      </Provider>
    </React.StrictMode>
  );
};

export default memo(InvokeAIUI);

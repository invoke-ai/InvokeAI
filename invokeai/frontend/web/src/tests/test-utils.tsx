import { ChakraProvider, extendTheme, theme as baseTheme, TOAST_OPTIONS } from '@invoke-ai/ui-library';
import { render } from '@testing-library/react';
import type { RenderOptions, RenderResult } from '@testing-library/react';
import { createStore } from 'app/store/store';
import type { AppStore } from 'app/store/store';
import type { ReactNode } from 'react';
import { Provider } from 'react-redux';

const defaultTheme = extendTheme({ ...baseTheme });

interface RenderWithProvidersOptions extends Omit<RenderOptions, 'wrapper'> {
  store?: AppStore;
}

interface RenderWithProvidersResult extends RenderResult {
  store: AppStore;
}

export function renderWithProviders(
  ui: ReactNode,
  { store = createStore(), ...renderOptions }: RenderWithProvidersOptions = {}
): RenderWithProvidersResult {
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <Provider store={store}>
        <ChakraProvider theme={defaultTheme} toastOptions={TOAST_OPTIONS}>
          {children}
        </ChakraProvider>
      </Provider>
    );
  }

  const result = render(ui, { wrapper: Wrapper, ...renderOptions });
  return { ...result, store };
}

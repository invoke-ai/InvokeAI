import { ChakraProvider, Menu } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it } from 'vitest';

import { CanvasGlobalContextMenu } from './CanvasGlobalContextMenu';

const englishCatalogModules = import.meta.glob('../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const enCatalog = Object.values(englishCatalogModules)[0] as Record<string, unknown>;
const testI18n = createInstance();
await testI18n.init({
  initImmediate: false,
  lng: 'en',
  resources: { en: { translation: enCatalog } },
});
const target = { layerId: null, x: 10, y: 20 } as const;
const noop = () => undefined;

describe('CanvasGlobalContextMenu', () => {
  it('labels empty-canvas actions as Canvas', () => {
    const markup = renderToStaticMarkup(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={testI18n}>
          <CanvasGlobalContextMenu target={target} onClose={noop}>
            <Menu.Item value="test-action">Test action</Menu.Item>
          </CanvasGlobalContextMenu>
        </I18nextProvider>
      </ChakraProvider>
    );

    expect(markup).toContain('Test action');
    expect(markup).toContain('>Canvas<');
    expect(markup.indexOf('>Canvas<')).toBeLessThan(markup.indexOf('Test action'));
  });
});

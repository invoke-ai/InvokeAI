import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { atom } from 'nanostores';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Global atom storing the language direction, to be consumed by the Chakra theme.
 *
 * Why do we need this? We have a kind of catch-22:
 * - The Chakra theme needs to know the language direction to apply the correct styles.
 * - The language direction is determined by i18n and the language selection.
 * - We want our error boundary to be themed.
 * - It's possible that i18n can throw if the language selection is invalid or not supported.
 *
 * Previously, we had the logic in this file in the theme provider, which wrapped the error boundary. The error
 * was properly themed. But then, if i18n threw in the theme provider, the error boundary does not catch the
 * error. The app would crash to a white screen.
 *
 * We tried swapping the component hierarchy so that the error boundary wraps the theme provider, but then the
 * error boundary isn't themed!
 *
 * The solution is to move this i18n direction logic out of the theme provider and into a hook that we can use
 * within the error boundary. The error boundary will be themed, _and_ catch any i18n errors.
 */
export const $direction = atom<'ltr' | 'rtl'>('ltr');

export const useSyncLangDirection = () => {
  useAssertSingleton('useSyncLangDirection');
  const { i18n, t } = useTranslation();

  useEffect(() => {
    const direction = i18n.dir();
    $direction.set(direction);
    document.body.dir = direction;
  }, [i18n, t]);
};

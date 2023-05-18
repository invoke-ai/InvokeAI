import { LightboxState } from './lightboxSlice';

/**
 * Lightbox slice persist denylist
 */
export const lightboxPersistDenylist: (keyof LightboxState)[] = [
  'isLightboxOpen',
];

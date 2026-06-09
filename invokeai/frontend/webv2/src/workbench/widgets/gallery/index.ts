import type { WidgetManifest } from '../../types';
import { GalleryWidgetView } from './GalleryWidgetView';

export const galleryWidgetManifest: WidgetManifest = {
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: 'lucide-react:image',
  id: 'gallery',
  label: 'Gallery',
  labelText: 'Gallery',
  regions: ['right', 'center', 'bottom'],
  version: 1,
  view: GalleryWidgetView,
};

import { Icon, type IconProps } from '@chakra-ui/react';
import { SquareIcon } from 'lucide-react';
import { lazy, Suspense, type ComponentType } from 'react';

import type { WidgetIconId } from './types';

type IconImporter = () => Promise<{ default: ComponentType }>;

const iconImporters = {
  'lucide-react:box': () => import('lucide-react/dist/esm/icons/box.mjs'),
  'lucide-react:bug': () => import('lucide-react/dist/esm/icons/bug.mjs'),
  'lucide-react:cloud-check': () => import('lucide-react/dist/esm/icons/cloud-check.mjs'),
  'lucide-react:bell': () => import('lucide-react/dist/esm/icons/bell.mjs'),
  'lucide-react:eye': () => import('lucide-react/dist/esm/icons/eye.mjs'),
  'lucide-react:folder-cog': () => import('lucide-react/dist/esm/icons/folder-cog.mjs'),
  'lucide-react:image': () => import('lucide-react/dist/esm/icons/image.mjs'),
  'lucide-react:info': () => import('lucide-react/dist/esm/icons/info.mjs'),
  'lucide-react:layers': () => import('lucide-react/dist/esm/icons/layers.mjs'),
  'lucide-react:list-ordered': () => import('lucide-react/dist/esm/icons/list-ordered.mjs'),
  'lucide-react:panel-bottom': () => import('lucide-react/dist/esm/icons/panel-bottom.mjs'),
  'lucide-react:plug-zap': () => import('lucide-react/dist/esm/icons/plug-zap.mjs'),
  'lucide-react:sliders-horizontal': () => import('lucide-react/dist/esm/icons/sliders-horizontal.mjs'),
  'lucide-react:undo-2': () => import('lucide-react/dist/esm/icons/undo-2.mjs'),
  'lucide-react:users': () => import('lucide-react/dist/esm/icons/users.mjs'),
  'lucide-react:wand-sparkles': () => import('lucide-react/dist/esm/icons/wand-sparkles.mjs'),
  'lucide-react:workflow': () => import('lucide-react/dist/esm/icons/workflow.mjs'),
} satisfies Partial<Record<WidgetIconId, IconImporter>>;

const iconCache = new Map<WidgetIconId, ComponentType>();

const getIconImporter = (iconId: WidgetIconId): IconImporter | undefined =>
  iconImporters[iconId as keyof typeof iconImporters];

export const isSupportedIconId = (iconId: string): iconId is WidgetIconId => {
  return getIconImporter(iconId as WidgetIconId) !== undefined;
};

const resolveLazyIcon = (iconId: WidgetIconId): ComponentType => {
  const cachedIcon = iconCache.get(iconId);

  if (cachedIcon) {
    return cachedIcon;
  }

  const LazyIcon = lazy(getIconImporter(iconId) ?? (() => Promise.resolve({ default: SquareIcon })));

  iconCache.set(iconId, LazyIcon);

  return LazyIcon;
};

export const WidgetIcon = ({ iconId, ...props }: IconProps & { iconId: WidgetIconId }) => {
  const ResolvedIcon = resolveLazyIcon(iconId);

  return (
    <Suspense fallback={<Icon as={SquareIcon} {...props} />}>
      <Icon as={ResolvedIcon} {...props} />
    </Suspense>
  );
};

import { Text } from '@invoke-ai/ui-library';
import type { Layer } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  type: Layer['type'];
};

export const LayerTitle = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const title = useMemo(() => {
    if (type === 'regional_guidance_layer') {
      return t('controlLayers.maskedGuidance');
    } else if (type === 'control_adapter_layer') {
      return t('common.controlNet');
    } else if (type === 'ip_adapter_layer') {
      return t('common.ipAdapter');
    }
  }, [t, type]);

  return (
    <Text size="sm" fontWeight="semibold" pointerEvents="none" color="base.300">
      {title}
    </Text>
  );
});

LayerTitle.displayName = 'LayerTitle';
